"""OpenAPI guard: no untyped response bodies (Epic 21.1, TASK-006).

Every documented operation must declare a JSON schema for each 2xx response
that actually types the body: ``$ref``s must resolve to a component with a
non-empty ``properties`` block — which catches the ``@model_serializer``
schema collapse (``{"type": "object", "additionalProperties": true}``) that
the byte-compat suite structurally cannot see (D4) — and inline schemas are
walked recursively so an untyped object (``-> dict[str, Any]``) or an untyped
array (``-> list[Any]``) cannot slip through on the strength of a bare
``type`` key. ``204`` no-content responses are exempt, narrowly: they must
carry no ``content`` block at all (Phase 21.2's DELETE lands as a 204 without
a response_model).

Hermetic: importing the app reads no env (settings load in the lifespan).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI

from rag_recipes.api.app import app
from rag_recipes.api.schemas.health import HealthResponse
from rag_recipes.api.schemas.search import SearchResponse

_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options", "trace")

# Component names legitimately documented without `properties` (e.g. a bare
# dict[str, Any] passthrough response model). None exist today — add a name
# here only with a comment justifying it.
_COMPONENT_EXEMPTIONS: frozenset[str] = frozenset()


_COMPOSITION_KEYS = ("allOf", "anyOf", "oneOf")


def _assert_typed_body(
    schema: dict[str, Any], components: dict[str, Any], where: str
) -> None:
    """Assert one response-body schema types something.

    Recurses through the *envelope* only — ``$ref``, composition branches and
    array items — and stops at the first component. It deliberately does not
    walk a component's own properties: fields like
    ``KnowledgeItemDetail.structured_data`` are untyped ``dict[str, Any]``
    passthroughs by design (the full recipe.v1 payload), which is a different
    question from "does this endpoint document its body at all".
    """
    ref = schema.get("$ref")
    if ref is not None:
        name = ref.rsplit("/", 1)[-1]
        assert name in components, f"{where}: {ref} does not resolve"
        if name in _COMPONENT_EXEMPTIONS:
            return
        # Load-bearing (D4): a @model_serializer collapses a model to
        # {"type": "object", "additionalProperties": true} — non-empty by a
        # shallow check, yet it types nothing.
        assert components[name].get("properties"), (
            f"{where}: component {name!r} has no properties "
            "(schema collapsed or untyped)"
        )
        return

    for key in _COMPOSITION_KEYS:
        branches = schema.get(key)
        if branches:
            for branch in branches:
                if branch.get("type") == "null":
                    continue
                _assert_typed_body(branch, components, f"{where} [{key}]")
            return

    schema_type = schema.get("type")
    assert schema_type or schema.get("properties"), (
        f"{where}: schema is empty/untyped: {schema!r}"
    )
    if schema_type == "object":
        if not schema.get("properties"):
            # A typed mapping (`-> dict[str, str]`) documents its body through
            # `additionalProperties`. `additionalProperties: true` or `{}` (a
            # `-> dict[str, Any]` / `response_model=dict` handler) is the same
            # untyped shape the $ref branch rejects, and `type` alone would wave
            # it through.
            additional = schema.get("additionalProperties")
            assert isinstance(additional, dict) and additional, (
                f"{where}: inline object schema types nothing "
                f"(dict passthrough): {schema!r}"
            )
            _assert_typed_body(
                additional, components, f"{where} [additionalProperties]"
            )
    elif schema_type == "array":
        # `-> list[Any]` documents `{"type": "array", "items": {}}`.
        items = schema.get("items")
        assert items, f"{where}: array schema has untyped items: {schema!r}"
        _assert_typed_body(items, components, f"{where} [items]")


def _assert_typed_success_responses(openapi: dict[str, Any]) -> None:
    """Walk every documented operation and assert its 2xx bodies are typed."""
    components = openapi.get("components", {}).get("schemas", {})
    for path, path_item in openapi["paths"].items():
        for method in _HTTP_METHODS:
            operation = path_item.get(method)
            if operation is None:
                continue
            where = f"{method.upper()} {path}"
            for status, response in operation.get("responses", {}).items():
                if not status.startswith("2"):
                    continue
                if status == "204":
                    # No-content by design; the exemption is narrow — a 204
                    # must not document a body at all.
                    assert "content" not in response, (
                        f"{where}: 204 must have no content block"
                    )
                    continue
                content = response.get("content")
                assert content, f"{where}: {status} response has no content block"
                json_content = content.get("application/json")
                assert json_content, f"{where}: {status} response is not JSON"
                schema = json_content.get("schema") or {}
                # A bare {} or {"title": ...} (an -> Any handler without a
                # response_model) fails inside the recursive check.
                _assert_typed_body(schema, components, f"{where}: {status}")


def test_every_documented_success_response_is_typed() -> None:
    _assert_typed_success_responses(app.openapi())


def test_debug_routes_stay_out_of_the_schema() -> None:
    """include_in_schema=False on the debug router remains load-bearing."""
    openapi = app.openapi()
    paths = set(openapi["paths"])
    assert not any("extraction-runs" in p for p in paths)
    assert not any("source-spans" in p for p in paths)
    # Baseline: 10 documented paths / 12 operations (GET+POST on /documents;
    # GET+DELETE on /documents/{document_id} since Phase 21.2's delete route;
    # +1/+1 for Phase 21.3's GET /review-items).
    assert len(paths) == 10
    operations = sum(
        1
        for item in openapi["paths"].values()
        for method in _HTTP_METHODS
        if method in item
    )
    assert operations == 12


def test_204_no_content_operation_is_exempted_not_failed() -> None:
    """A 204 route without a response_model must pass the guard (21.2's DELETE)."""
    throwaway = FastAPI(separate_input_output_schemas=False)

    @throwaway.delete("/things/{thing_id}", status_code=204)
    async def delete_thing(thing_id: str) -> None:  # pragma: no cover - schema only
        return None

    _assert_typed_success_responses(throwaway.openapi())


# --- Red direction: the guard must actually fail on untyped bodies ---
#
# Without these, gutting `_assert_typed_success_responses` to `pass` would leave
# the whole module green — the guard would assert nothing and no test would say so.


def _guard_message(app_: FastAPI) -> str:
    with pytest.raises(AssertionError) as excinfo:
        _assert_typed_success_responses(app_.openapi())
    return str(excinfo.value)


def test_guard_fails_on_route_without_response_model() -> None:
    """The regression this guard exists for: `-> Any` and no response_model."""
    throwaway = FastAPI(separate_input_output_schemas=False)

    @throwaway.get("/things")
    async def list_things() -> Any:  # pragma: no cover - schema only
        return {}

    assert "empty/untyped" in _guard_message(throwaway)


def test_guard_fails_on_untyped_dict_response() -> None:
    """`-> dict[str, Any]` documents an object that types nothing."""
    throwaway = FastAPI(separate_input_output_schemas=False)

    @throwaway.get("/things")
    async def list_things() -> dict[str, Any]:  # pragma: no cover - schema only
        return {}

    assert "types nothing" in _guard_message(throwaway)


def test_guard_accepts_a_typed_mapping_body() -> None:
    """`-> dict[str, str]` has no properties but is still a fully typed body."""
    throwaway = FastAPI(separate_input_output_schemas=False)

    @throwaway.get("/things")
    async def list_things() -> dict[str, str]:  # pragma: no cover - schema only
        return {}

    _assert_typed_success_responses(throwaway.openapi())


def test_guard_fails_on_mapping_of_untyped_values() -> None:
    """The mapping allowance is not a bypass — its value schema is walked too."""
    throwaway = FastAPI(separate_input_output_schemas=False)

    @throwaway.get("/things")
    async def list_things() -> dict[str, list[Any]]:  # pragma: no cover - schema only
        return {}

    assert "untyped items" in _guard_message(throwaway)


def test_guard_fails_on_untyped_array_response() -> None:
    """`-> list[Any]` documents `{"type": "array", "items": {}}`."""
    throwaway = FastAPI(separate_input_output_schemas=False)

    @throwaway.get("/things")
    async def list_things() -> list[Any]:  # pragma: no cover - schema only
        return []

    assert "untyped items" in _guard_message(throwaway)


def test_guard_accepts_a_typed_list_body() -> None:
    """The array rule must not fire on a legitimately typed collection."""
    throwaway = FastAPI(separate_input_output_schemas=False)

    @throwaway.get("/things")
    async def list_things() -> list[HealthResponse]:  # pragma: no cover - schema only
        return []

    _assert_typed_success_responses(throwaway.openapi())


def test_guard_fails_on_serializer_collapsed_component() -> None:
    """The D4 failure mode: `separate_input_output_schemas` left at its default
    collapses a `@model_serializer` model to a property-less object behind a
    `$ref`, with byte-perfect responses. Proves `api/app.py`'s flag is
    load-bearing and that the guard, not luck, is what holds it."""
    throwaway = FastAPI()  # i.e. separate_input_output_schemas=True (default)

    @throwaway.post("/search", response_model=SearchResponse)
    async def search() -> Any:  # pragma: no cover - schema only
        return None

    assert "has no properties" in _guard_message(throwaway)

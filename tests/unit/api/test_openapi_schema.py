"""OpenAPI guard: no untyped response bodies (Epic 21.1, TASK-006).

Every documented operation must declare a non-empty JSON schema for each 2xx
response, and every ``$ref`` must resolve to a component with a non-empty
``properties`` block — the second half catches the ``@model_serializer``
schema collapse (``{"type": "object", "additionalProperties": true}``) that
the byte-compat suite structurally cannot see (D4). ``204`` no-content
responses are exempt, narrowly: they must carry no ``content`` block at all
(Phase 21.2's DELETE lands as a 204 without a response_model).

Hermetic: importing the app reads no env (settings load in the lifespan).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from rag_recipes.api.app import app

_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options", "trace")

# Component names legitimately documented without `properties` (e.g. a bare
# dict[str, Any] passthrough response model). None exist today — add a name
# here only with a comment justifying it.
_COMPONENT_EXEMPTIONS: frozenset[str] = frozenset()


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
                ref = schema.get("$ref")
                if ref is None:
                    # Inline schema: must actually type something. A bare {}
                    # or {"title": ...} (an -> Any handler without a
                    # response_model) fails here.
                    assert any(
                        key in schema for key in ("type", "properties", "allOf", "anyOf")
                    ), f"{where}: {status} schema is empty/untyped: {schema!r}"
                    continue
                name = ref.rsplit("/", 1)[-1]
                assert name in components, f"{where}: {ref} does not resolve"
                if name in _COMPONENT_EXEMPTIONS:
                    continue
                properties = components[name].get("properties") or {}
                # Load-bearing (D4): a @model_serializer collapses a model to
                # {"type": "object", "additionalProperties": true} — non-empty
                # by a shallow check, yet it types nothing.
                assert properties, (
                    f"{where}: component {name!r} has no properties "
                    "(schema collapsed or untyped)"
                )


def test_every_documented_success_response_is_typed() -> None:
    _assert_typed_success_responses(app.openapi())


def test_debug_routes_stay_out_of_the_schema() -> None:
    """include_in_schema=False on the debug router remains load-bearing."""
    openapi = app.openapi()
    paths = set(openapi["paths"])
    assert not any("extraction-runs" in p for p in paths)
    assert not any("source-spans" in p for p in paths)
    # Baseline: 9 documented paths / 10 operations (GET+POST on /documents).
    assert len(paths) == 9
    operations = sum(
        1
        for item in openapi["paths"].values()
        for method in _HTTP_METHODS
        if method in item
    )
    assert operations == 10


def test_204_no_content_operation_is_exempted_not_failed() -> None:
    """A 204 route without a response_model must pass the guard (21.2's DELETE)."""
    throwaway = FastAPI(separate_input_output_schemas=False)

    @throwaway.delete("/things/{thing_id}", status_code=204)
    async def delete_thing(thing_id: str) -> None:  # pragma: no cover - schema only
        return None

    _assert_typed_success_responses(throwaway.openapi())

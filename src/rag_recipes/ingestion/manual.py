"""Hand-typed recipes: the shelf they live on, and the row one becomes.

Every other knowledge item in this system is the output of an extraction run
over a PDF. A manual recipe has no PDF and no run, but it still has to be a
``KnowledgeItem`` — anything else would leave it out of search, out of the
library, out of favourites and out of the edit endpoint. So it becomes an
ordinary row, and this module supplies the two things it cannot supply itself:

- ``ensure_manual_shelf`` — the source chain the FKs demand (SourceAsset →
  Document → ExtractionRun), created once and shared by every manual recipe.
- ``authored_recipe`` — the column values a typed recipe produces, composed by
  the *edit* layer rather than by a second implementation of the ``recipe.v1``
  shape.

That second point is the load-bearing one. A created recipe and an edited one
are the same act — a human writing lines into a form — so ``authored_recipe``
runs ``apply_edit`` over an empty ``recipe.v1`` payload instead of assembling
ingredient and step rows by hand. The human-authored row shape (parse fields
nulled, confidences 1.0, ``edited: true``), the ``ingredients_text`` /
``steps_text`` refresh and the ``body_text`` composition are therefore defined
exactly once, in ``ingestion/editing``, and a manual recipe cannot drift into a
shape the edit endpoint would not produce.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.editing import (
    RecipeEdit,
    apply_edit,
    notes_for_item,
    warnings_for_item,
)
from rag_recipes.ingestion.pipeline.composition import compose_body_text
from rag_recipes.ingestion.pipeline.extraction import SCHEMA_VERSION
from rag_recipes.ingestion.validation import SoftValidationThresholds
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.source_asset import SourceAsset

__all__ = [
    "MANUAL_SHELF_DOCUMENT_ID",
    "MANUAL_SHELF_TITLE",
    "AuthoredRecipe",
    "ManualShelf",
    "authored_recipe",
    "ensure_manual_shelf",
    "human_confidence",
]

# Fixed ids, not ULIDs: they are what makes the bootstrap below idempotent
# without a lookup-then-insert race. Only ever three rows carry them.
MANUAL_SHELF_ASSET_ID = "asset_manual_shelf"
MANUAL_SHELF_DOCUMENT_ID = "doc_manual_shelf"
MANUAL_SHELF_RUN_ID = "run_manual_shelf"
#: The one generation the shelf ever has (see ``ensure_manual_shelf``).
MANUAL_SHELF_SOURCE_VERSION = 1
MANUAL_SHELF_TITLE = "Handwritten"
MANUAL_SHELF_CATEGORY = "recipes"
#: ``source_assets.content_hash`` is UNIQUE and NOT NULL. This is not a hash of
#: anything — there is no file — it is a sentinel that cannot collide with a
#: real SHA-256 (wrong alphabet, wrong length).
MANUAL_SHELF_CONTENT_HASH = "manual:handwritten-shelf"


@dataclass(frozen=True)
class ManualShelf:
    """The three ids a manual ``KnowledgeItem`` needs to be insertable."""

    document_id: str
    extraction_run_id: str
    source_version: int


@dataclass(frozen=True)
class AuthoredRecipe:
    """The column values a typed recipe produces, ready for a ``KnowledgeItem``."""

    title: str
    normalized_title: str
    summary: str | None
    body_text: str
    structured_data: dict[str, Any]
    confidence: dict[str, Any]


def human_confidence() -> dict[str, Any]:
    """The item-level confidence a hand-typed recipe carries: 1.0 throughout.

    A fresh dict per call — it is written into a JSONB column and must not be
    shared between rows.

    Not a flattering default but the honest one, and the same claim the edit
    layer already makes for a line a human retyped (``_HUMAN_INGREDIENT_CONFIDENCE``):
    these floats measure how sure the *extractor* is that it read the page
    correctly, and nothing read a page here. Storing it explicitly rather than
    leaving the column NULL also keeps a later edit's ``warnings_for_item``
    from having to assume it, which is the only other way the same numbers
    arrive.
    """
    return {
        "overall": 1.0,
        "boundary": 1.0,
        "fields": {
            "title": 1.0,
            "summary": 1.0,
            "yield": 1.0,
            "ingredients": 1.0,
            "steps": 1.0,
        },
    }


async def ensure_manual_shelf(session: AsyncSession) -> ManualShelf:
    """Get-or-create the handwritten shelf: asset → document → run.

    A ``KnowledgeItem`` cannot stand alone. ``document_id`` and
    ``extraction_run_id`` are both NOT NULL, and the composite FK
    ``fk_knowledge_items_extraction_run_document_version`` additionally forces
    the run to agree with the item on the document *and* the source version. So
    a hand-typed recipe needs a source chain, and this is it: one Document
    titled "Handwritten", one ExtractionRun that never ran, and one SourceAsset
    that exists only because ``documents.asset_id`` is NOT NULL.

    Idempotent by construction: all three rows carry fixed ids and are inserted
    ``ON CONFLICT DO NOTHING``, so a second call writes nothing and two
    concurrent creates cannot both win. Flush-level only — the caller owns the
    transaction, like every other write in this package.

    **One shelf for every manual recipe**, not one document per recipe: the
    library lists documents, so per-recipe documents would fill the shelf with
    one-recipe spines and make every per-book count meaningless.

    ``active_source_version`` is set to the shelf's version up front rather than
    left NULL for ``index_knowledge_item``'s handoff to fill in. Search requires
    ``source_version == active_source_version`` (``retrieval/_sql.py``) and
    every manual item is written at that one version, so the handoff has nothing
    to decide and sees equal-and-no-op forever. Nothing here ever creates a
    second generation — there is no re-extraction that could — which is also why
    ``POST /documents/{id}/reprocess`` refuses a ``MANUAL`` document.

    The asset's values are the honest ones available: no storage provider, no
    key, and ``UPLOADED`` only because ``UploadStatus`` has no "there was never
    a file" member. Its one reader is ``delete_document_cascade``'s post-commit
    file delete, which is handed a key it will not find — already its
    best-effort path.
    """
    # No index_elements on the asset: it has TWO unique keys (id and
    # content_hash), and naming one would turn a conflict on the other into an
    # IntegrityError instead of the no-op this function promises.
    await session.execute(
        pg_insert(SourceAsset)
        .values(
            id=MANUAL_SHELF_ASSET_ID,
            source_type=SourceType.MANUAL,
            original_filename=MANUAL_SHELF_TITLE,
            storage_provider="none",
            storage_key="",
            content_hash=MANUAL_SHELF_CONTENT_HASH,
            upload_status=UploadStatus.UPLOADED,
        )
        .on_conflict_do_nothing()
    )
    await session.execute(
        pg_insert(Document)
        .values(
            id=MANUAL_SHELF_DOCUMENT_ID,
            asset_id=MANUAL_SHELF_ASSET_ID,
            category=MANUAL_SHELF_CATEGORY,
            subcategory=None,
            title=MANUAL_SHELF_TITLE,
            author="",
            source_type=SourceType.MANUAL,
            language=None,
            active_source_version=MANUAL_SHELF_SOURCE_VERSION,
            status=DocumentStatus.READY,
        )
        .on_conflict_do_nothing()
    )
    await session.execute(
        pg_insert(ExtractionRun)
        .values(
            id=MANUAL_SHELF_RUN_ID,
            document_id=MANUAL_SHELF_DOCUMENT_ID,
            source_version=MANUAL_SHELF_SOURCE_VERSION,
            # Not a provider name — no provider was called. All three identity
            # fields say "manual" so a cache lookup or an eval keyed on them can
            # never mistake this row for a real extraction run.
            provider="manual",
            model="manual",
            prompt_version="manual",
            schema_version=SCHEMA_VERSION,
            input_source_span_ids=[],
            input_hash="manual",
            status=ExtractionRunStatus.SUCCESS,
            output_json=None,
        )
        .on_conflict_do_nothing()
    )
    await session.flush()
    return ManualShelf(
        document_id=MANUAL_SHELF_DOCUMENT_ID,
        extraction_run_id=MANUAL_SHELF_RUN_ID,
        source_version=MANUAL_SHELF_SOURCE_VERSION,
    )


def authored_recipe(
    *,
    title: str,
    summary: str | None,
    yield_: str | None,
    prep_time: str | None,
    cook_time: str | None,
    total_time: str | None,
    ingredients: list[str],
    steps: list[str],
    thresholds: SoftValidationThresholds,
) -> AuthoredRecipe:
    """Compose the row values for a recipe a human typed. Pure.

    ``apply_edit`` over an empty ``recipe.v1`` payload does the work: every
    submitted line is unmatched against the (absent) existing rows, so each one
    takes the human-authored shape the edit layer defines, and
    ``ingredients_text`` / ``steps_text`` are refreshed from them. The identity
    fields ride in as the *base* (``title`` / ``summary``) and the payload
    fields as the *edit*, which is the split those two arguments already mean.

    ``body_text`` is composed rather than carried: unlike an edit, there is no
    extracted prose to preserve, and ``apply_edit`` leaves the base body
    untouched when neither list changes — which for an empty base is the empty
    string, and an empty ``body_text`` costs the item its ``recipe_full`` chunk.

    Warnings are derived by the same ``warnings_for_item`` an edit uses, so a
    typed recipe with no method carries ``no_steps`` exactly as an extracted one
    would. They do NOT gate the status: an authored recipe is not a candidate
    for review — there is no extraction to second-guess — so the caller shelves
    it either way. The codes are stored because a later edit recomputes them
    from the content regardless, and a row whose warnings disagreed with its
    text would be a lie waiting to surface.
    """
    edited = apply_edit(
        title=title,
        summary=summary,
        body_text="",
        structured_data={"schema": SCHEMA_VERSION},
        edit=RecipeEdit(
            yield_=yield_,
            prep_time=prep_time,
            cook_time=cook_time,
            total_time=total_time,
            ingredients=list(ingredients),
            steps=list(steps),
        ),
    )
    structured = deepcopy(edited.structured_data)
    body_text = compose_body_text(title=edited.title, structured=structured)
    confidence = human_confidence()
    structured["warnings"] = warnings_for_item(
        title=edited.title,
        summary=edited.summary,
        body_text=body_text,
        source_span_ids=[],
        structured_data=structured,
        confidence=confidence,
        thresholds=thresholds,
    )
    structured["validation_notes"] = notes_for_item(
        title=edited.title,
        summary=edited.summary,
        body_text=body_text,
        source_span_ids=[],
        structured_data=structured,
        confidence=confidence,
        thresholds=thresholds,
    )
    structured["validation_thresholds"] = thresholds.to_record()
    return AuthoredRecipe(
        title=edited.title,
        normalized_title=edited.normalized_title,
        summary=edited.summary,
        body_text=body_text,
        structured_data=structured,
        confidence=confidence,
    )

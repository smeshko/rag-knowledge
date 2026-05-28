from __future__ import annotations

import pytest

from rag_recipes.ingestion.status import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    InvalidTransitionError,
    is_terminal,
)
from rag_recipes.storage.enums import DocumentStatus


def test_terminal_statuses_match_doc2() -> None:
    expected = frozenset(
        {DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW, DocumentStatus.FAILED}
    )
    assert expected == TERMINAL_STATUSES


@pytest.mark.parametrize("status", list(DocumentStatus))
def test_is_terminal_for_terminal_and_non_terminal(status: DocumentStatus) -> None:
    expected = status in {
        DocumentStatus.READY,
        DocumentStatus.NEEDS_REVIEW,
        DocumentStatus.FAILED,
    }
    assert is_terminal(status) is expected


def test_valid_transitions_keys_cover_every_status() -> None:
    # Missing keys would KeyError inside transition_to; guard at the matrix
    # level so an enum addition forces a matrix update.
    assert set(VALID_TRANSITIONS.keys()) == set(DocumentStatus)


@pytest.mark.parametrize(
    "current,expected_next",
    [
        (DocumentStatus.QUEUED, DocumentStatus.EXTRACTING_TEXT),
        (DocumentStatus.EXTRACTING_TEXT, DocumentStatus.CREATING_SOURCE_SPANS),
        (DocumentStatus.CREATING_SOURCE_SPANS, DocumentStatus.EXTRACTING_ITEMS),
        (DocumentStatus.EXTRACTING_ITEMS, DocumentStatus.VALIDATING_ITEMS),
        (DocumentStatus.VALIDATING_ITEMS, DocumentStatus.CREATING_CHUNKS),
        (DocumentStatus.CREATING_CHUNKS, DocumentStatus.EMBEDDING_CHUNKS),
        (DocumentStatus.EMBEDDING_CHUNKS, DocumentStatus.INDEXING),
        (DocumentStatus.INDEXING, DocumentStatus.READY),
        (DocumentStatus.INDEXING, DocumentStatus.NEEDS_REVIEW),
    ],
)
def test_valid_transitions_covers_linear_progression(
    current: DocumentStatus, expected_next: DocumentStatus
) -> None:
    assert expected_next in VALID_TRANSITIONS[current]


@pytest.mark.parametrize(
    "current",
    [s for s in DocumentStatus if s not in TERMINAL_STATUSES],
)
def test_valid_transitions_includes_universal_failure_edge(
    current: DocumentStatus,
) -> None:
    assert DocumentStatus.FAILED in VALID_TRANSITIONS[current]


@pytest.mark.parametrize(
    "current",
    [DocumentStatus.FAILED, DocumentStatus.NEEDS_REVIEW, DocumentStatus.READY],
)
def test_valid_transitions_includes_retry_edges(current: DocumentStatus) -> None:
    assert VALID_TRANSITIONS[current] == frozenset({DocumentStatus.QUEUED})


def test_valid_transitions_rejects_random_illegal_edge() -> None:
    assert DocumentStatus.INDEXING not in VALID_TRANSITIONS[DocumentStatus.QUEUED]
    assert DocumentStatus.READY not in VALID_TRANSITIONS[DocumentStatus.QUEUED]


def test_invalid_transition_error_message_includes_all_three() -> None:
    err = InvalidTransitionError(
        current=DocumentStatus.QUEUED,
        attempted=DocumentStatus.INDEXING,
        allowed=frozenset({DocumentStatus.EXTRACTING_TEXT, DocumentStatus.FAILED}),
    )
    rendered = str(err)
    assert "queued" in rendered
    assert "indexing" in rendered
    assert "extracting_text" in rendered
    assert "failed" in rendered
    assert err.current is DocumentStatus.QUEUED
    assert err.attempted is DocumentStatus.INDEXING
    assert err.allowed == frozenset(
        {DocumentStatus.EXTRACTING_TEXT, DocumentStatus.FAILED}
    )

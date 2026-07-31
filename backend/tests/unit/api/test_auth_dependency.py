"""Unit tests for require_api_token (fail-closed bearer auth)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from rag_recipes.api.dependencies import require_api_token
from rag_recipes.api.errors import ApiError, ErrorCode


def _settings(token: str | None) -> Any:
    class _S:
        personal_api_token = token

    return _S()


def _creds(scheme: str, token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=token)


def test_unset_token_fails_closed() -> None:
    with pytest.raises(ApiError) as excinfo:
        require_api_token(
            credentials=_creds("Bearer", "anything"),
            settings=_settings(None),
        )
    assert excinfo.value.status_code == 401
    assert excinfo.value.code is ErrorCode.UNAUTHORIZED


def test_unset_token_fails_closed_even_with_no_credentials() -> None:
    with pytest.raises(ApiError) as excinfo:
        require_api_token(credentials=None, settings=_settings(None))
    assert excinfo.value.status_code == 401
    assert excinfo.value.code is ErrorCode.UNAUTHORIZED


def test_valid_token_passes() -> None:
    require_api_token(
        credentials=_creds("Bearer", "sekret"),
        settings=_settings("sekret"),
    )


def test_valid_token_accepts_lowercase_scheme() -> None:
    require_api_token(
        credentials=_creds("bearer", "sekret"),
        settings=_settings("sekret"),
    )


def test_missing_credentials_when_token_configured_rejects() -> None:
    with pytest.raises(ApiError) as excinfo:
        require_api_token(credentials=None, settings=_settings("sekret"))
    assert excinfo.value.status_code == 401
    assert excinfo.value.code is ErrorCode.UNAUTHORIZED


def test_wrong_token_rejects() -> None:
    with pytest.raises(ApiError) as excinfo:
        require_api_token(
            credentials=_creds("Bearer", "nope"),
            settings=_settings("sekret"),
        )
    assert excinfo.value.status_code == 401
    assert excinfo.value.code is ErrorCode.UNAUTHORIZED


def test_non_bearer_scheme_rejects() -> None:
    with pytest.raises(ApiError) as excinfo:
        require_api_token(
            credentials=_creds("Basic", "sekret"),
            settings=_settings("sekret"),
        )
    assert excinfo.value.status_code == 401
    assert excinfo.value.code is ErrorCode.UNAUTHORIZED

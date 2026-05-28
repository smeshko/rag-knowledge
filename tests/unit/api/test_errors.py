from __future__ import annotations

import pytest

from rag_recipes.api.errors import ApiError, ErrorCode, error_body


class TestErrorCode:
    def test_members_have_expected_string_values(self) -> None:
        assert ErrorCode.INVALID_REQUEST.value == "invalid_request"
        assert ErrorCode.UNSUPPORTED_FILE_TYPE.value == "unsupported_file_type"
        assert ErrorCode.DUPLICATE_SOURCE_ASSET.value == "duplicate_source_asset"
        assert ErrorCode.INTERNAL_ERROR.value == "internal_error"

    def test_is_string_enum(self) -> None:
        # StrEnum members must compare equal to their string value.
        assert ErrorCode.INVALID_REQUEST == "invalid_request"
        assert isinstance(ErrorCode.INVALID_REQUEST, str)


class TestApiError:
    def test_stores_all_attributes(self) -> None:
        err = ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="bad input",
            details={"field": "file"},
        )
        assert err.status_code == 400
        assert err.code is ErrorCode.INVALID_REQUEST
        assert err.message == "bad input"
        assert err.details == {"field": "file"}

    def test_details_defaults_to_empty_dict(self) -> None:
        err = ApiError(status_code=500, code=ErrorCode.INTERNAL_ERROR, message="boom")
        assert err.details == {}

    def test_is_an_exception(self) -> None:
        err = ApiError(status_code=415, code=ErrorCode.UNSUPPORTED_FILE_TYPE, message="nope")
        assert isinstance(err, Exception)
        with pytest.raises(ApiError):
            raise err


class TestErrorBody:
    def test_renders_exact_doc_6_shape(self) -> None:
        body = error_body(
            code=ErrorCode.INVALID_REQUEST,
            message="missing file",
            details={"field": "file"},
        )
        assert body == {
            "error": {
                "code": "invalid_request",
                "message": "missing file",
                "details": {"field": "file"},
            }
        }

    def test_serialises_code_to_string_value_not_enum_repr(self) -> None:
        body = error_body(
            code=ErrorCode.UNSUPPORTED_FILE_TYPE,
            message="not a pdf",
            details={},
        )
        assert body["error"]["code"] == "unsupported_file_type"
        assert isinstance(body["error"]["code"], str)

    def test_defaults_details_to_empty_dict_when_none(self) -> None:
        body = error_body(
            code=ErrorCode.INTERNAL_ERROR,
            message="server error",
            details=None,
        )
        assert body == {
            "error": {
                "code": "internal_error",
                "message": "server error",
                "details": {},
            }
        }


class TestApiErrorToBody:
    def test_matches_error_body_helper(self) -> None:
        err = ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="missing file",
            details={"field": "file"},
        )
        assert err.to_body() == error_body(
            code=ErrorCode.INVALID_REQUEST,
            message="missing file",
            details={"field": "file"},
        )

    def test_default_details_yields_empty_details_dict_in_body(self) -> None:
        err = ApiError(status_code=500, code=ErrorCode.INTERNAL_ERROR, message="boom")
        body = err.to_body()
        assert body == {
            "error": {
                "code": "internal_error",
                "message": "boom",
                "details": {},
            }
        }

import pytest

from deepface_server.errors import BadRequest
from deepface_server.schemas import (
    serialize_actions,
    validate_analyze_request,
    validate_batch_request,
)


def test_validate_analyze_rejects_non_object():
    with pytest.raises(BadRequest):
        validate_analyze_request(["not", "an", "object"])


def test_validate_analyze_rejects_missing_image():
    with pytest.raises(BadRequest):
        validate_analyze_request({})


def test_validate_analyze_rejects_non_string_image():
    with pytest.raises(BadRequest):
        validate_analyze_request({"image": 42})


def test_validate_analyze_returns_image_string():
    assert validate_analyze_request({"image": "abc"}) == "abc"


def test_validate_batch_rejects_non_list():
    with pytest.raises(BadRequest):
        validate_batch_request({"images": "abc"}, max_items=4)


def test_validate_batch_rejects_empty():
    with pytest.raises(BadRequest):
        validate_batch_request({"images": []}, max_items=4)


def test_validate_batch_rejects_oversize():
    with pytest.raises(BadRequest):
        validate_batch_request({"images": ["a", "b", "c", "d", "e"]}, max_items=4)


def test_validate_batch_rejects_non_string_entries():
    with pytest.raises(BadRequest):
        validate_batch_request({"images": ["a", 1]}, max_items=4)


def test_serialize_actions_coerces_to_list_of_str():
    assert serialize_actions(("a", "b")) == ["a", "b"]

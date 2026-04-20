import pytest

from deepface_server.errors import ImageDecodeError, ImageTooLarge
from deepface_server.services.image_service import ImageService


def test_decode_b64_rejects_empty():
    svc = ImageService(max_bytes=1024)
    with pytest.raises(ImageDecodeError):
        svc.decode_b64("")


def test_decode_b64_rejects_huge_estimated_size():
    svc = ImageService(max_bytes=10)
    huge = "A" * 1024
    with pytest.raises(ImageTooLarge):
        svc.decode_b64(huge)


def test_decode_b64_rejects_invalid_chars():
    svc = ImageService(max_bytes=1024)
    with pytest.raises(ImageDecodeError):
        svc.decode_b64("***")


def test_fingerprint_is_stable_and_distinct():
    svc = ImageService()
    a = svc.fingerprint(b"hello")
    b = svc.fingerprint(b"hello")
    c = svc.fingerprint(b"world")
    assert a == b
    assert a != c
    assert len(a) == 64  # sha256 hex length


def test_to_ndarray_rejects_undecodable_bytes():
    svc = ImageService(max_bytes=1024)
    with pytest.raises(ImageDecodeError):
        svc.to_ndarray(b"\x00\x01\x02\x03")

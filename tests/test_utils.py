from deepface_server.utils.base64_utils import looks_like_base64, strip_data_uri
from deepface_server.utils.image_utils import detect_format, estimate_dimensions


def test_strip_data_uri_passthrough_when_not_a_uri():
    assert strip_data_uri("AAAA") == "AAAA"


def test_strip_data_uri_removes_prefix():
    payload = "data:image/png;base64,ABCD"
    assert strip_data_uri(payload) == "ABCD"


def test_looks_like_base64_handles_padding():
    assert looks_like_base64("QUJDR") is False  # length not multiple of 4
    assert looks_like_base64("QUJDRA==") is True
    assert looks_like_base64("not base64") is False


def test_detect_format_recognises_known_magics():
    assert detect_format(b"\xff\xd8\xff something") == "jpeg"
    assert detect_format(b"\x89PNG\r\n\x1a\n more") == "png"
    assert detect_format(b"RIFF1234WEBPxxxx") == "webp"
    assert detect_format(b"unknown") == "unknown"


def test_estimate_dimensions_reads_png_header():
    width = 7
    height = 11
    header = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00" * 8  # length + IHDR magic placeholder
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )
    assert estimate_dimensions(header) == (width, height)
    assert estimate_dimensions(b"\xff\xd8\xff") is None

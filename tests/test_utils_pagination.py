from __future__ import annotations

import pytest

from deepface_server.utils import pagination as p


def test_encode_decode_round_trip():
    cur = p.encode_cursor(42, secret="s")
    assert p.decode_cursor(cur, secret="s") == 42


def test_decode_with_wrong_secret_fails():
    cur = p.encode_cursor(42, secret="s")
    with pytest.raises(ValueError):
        p.decode_cursor(cur, secret="other")


def test_encode_negative_offset():
    with pytest.raises(ValueError):
        p.encode_cursor(-1)


def test_decode_empty():
    with pytest.raises(ValueError):
        p.decode_cursor("")


def test_decode_malformed():
    with pytest.raises(ValueError):
        p.decode_cursor("nodot")


def test_paginate_basic():
    page = p.paginate(list(range(10)), limit=3)
    assert page.items == (0, 1, 2)
    assert page.next_cursor is not None


def test_paginate_with_cursor():
    items = list(range(10))
    page = p.paginate(items, limit=3)
    page2 = p.paginate(items, limit=3, cursor=page.next_cursor)
    assert page2.items == (3, 4, 5)


def test_paginate_last_page_no_cursor():
    page = p.paginate(list(range(3)), limit=10)
    assert page.next_cursor is None
    assert len(page.items) == 3


def test_paginate_invalid_limit():
    with pytest.raises(ValueError):
        p.paginate([], limit=0)
    with pytest.raises(ValueError):
        p.paginate([], limit=2000)


def test_split_into_pages():
    pages = p.split_into_pages(list(range(7)), page_size=3)
    assert len(pages) == 3
    assert pages[0] == (0, [0, 1, 2])
    assert pages[2] == (2, [6])


def test_split_into_pages_invalid():
    with pytest.raises(ValueError):
        p.split_into_pages([], page_size=0)


def test_page_to_dict():
    page = p.Page(items=(1, 2), next_cursor=None, total=2)
    d = page.to_dict()
    assert d == {"items": [1, 2], "next_cursor": None, "total": 2}

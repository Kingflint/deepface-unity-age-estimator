import pytest

from deepface_server.services.cache import LRUCache


def test_lru_cache_returns_none_on_miss():
    cache = LRUCache(max_entries=2)
    assert cache.get("missing") is None


def test_lru_cache_evicts_oldest_entry():
    cache = LRUCache(max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_lru_cache_promotes_on_access():
    cache = LRUCache(max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # promotes 'a' to most-recent
    cache.set("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_lru_cache_reports_stats():
    cache = LRUCache(max_entries=4)
    cache.set("k", "v")
    cache.get("k")
    cache.get("missing")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert 0.0 <= stats["hit_rate"] <= 1.0


def test_lru_cache_rejects_zero_capacity():
    with pytest.raises(ValueError):
        LRUCache(max_entries=0)

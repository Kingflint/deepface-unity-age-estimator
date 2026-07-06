from __future__ import annotations

import pytest

from deepface_server.jobs import priority as pq


def test_pop_lowest_priority_first():
    q = pq.PriorityQueue()
    q.push("a", priority=10)
    q.push("b", priority=1)
    q.push("c", priority=5)
    assert q.pop() == "b"
    assert q.pop() == "c"
    assert q.pop() == "a"


def test_fifo_within_priority():
    q = pq.PriorityQueue()
    q.push("first", priority=1)
    q.push("second", priority=1)
    q.push("third", priority=1)
    assert q.pop() == "first"
    assert q.pop() == "second"
    assert q.pop() == "third"


def test_pop_empty_raises():
    q = pq.PriorityQueue()
    with pytest.raises(pq.PriorityQueueEmpty):
        q.pop()


def test_peek_does_not_consume():
    q = pq.PriorityQueue()
    q.push("a", priority=1)
    assert q.peek() == "a"
    assert q.peek() == "a"
    assert len(q) == 1


def test_peek_empty_raises():
    q = pq.PriorityQueue()
    with pytest.raises(pq.PriorityQueueEmpty):
        q.peek()


def test_cancel_skips_payload():
    q = pq.PriorityQueue()
    t1 = q.push("a", priority=1)
    q.push("b", priority=2)
    assert q.cancel(t1)
    assert q.pop() == "b"


def test_cancel_unknown_returns_false():
    q = pq.PriorityQueue()
    assert not q.cancel(999)


def test_len_excludes_cancelled():
    q = pq.PriorityQueue()
    t = q.push("a", priority=1)
    q.push("b", priority=2)
    q.cancel(t)
    assert len(q) == 1


def test_bool():
    q = pq.PriorityQueue()
    assert not q
    q.push("a", priority=1)
    assert q


def test_drain_yields_all_in_order():
    q = pq.PriorityQueue()
    for i, p in enumerate([3, 1, 2]):
        q.push(i, priority=p)
    assert list(q.drain()) == [1, 2, 0]


def test_snapshot_does_not_consume():
    q = pq.PriorityQueue()
    q.push("a", priority=2)
    q.push("b", priority=1)
    snap = q.snapshot()
    assert snap == ["b", "a"]
    assert len(q) == 2

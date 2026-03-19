def test_lru_ttl_cache_hits_and_eviction():
    from evo_rhyme.cache import LRUTTLCache

    t = {"now": 0.0}

    def now():
        return t["now"]

    c = LRUTTLCache(max_size=2, ttl_seconds=10.0, time_fn=now)
    assert c.get("missing") is None

    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1  # hit, and 'a' becomes most-recent
    c.set("c", 3)  # should evict 'b'

    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3

    snap = c.snapshot()
    assert snap["evictions"] >= 1
    assert snap["hits"] >= 2


def test_lru_ttl_cache_expires():
    from evo_rhyme.cache import LRUTTLCache

    t = {"now": 0.0}

    def now():
        return t["now"]

    c = LRUTTLCache(max_size=10, ttl_seconds=1.0, time_fn=now)
    c.set("x", "y")
    assert c.get("x") == "y"
    t["now"] = 2.0
    assert c.get("x") is None
    snap = c.snapshot()
    assert snap["expired"] >= 1


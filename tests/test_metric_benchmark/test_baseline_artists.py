from evo_rhyme.metric_benchmark.baseline_artists import (
    classify_track_group,
    default_quota_fractions,
    parse_artists_field,
)


def test_parse_artists_field():
    raw = '["Kendrick Lamar", "Nas"]'
    assert parse_artists_field(raw) == ["Kendrick Lamar", "Nas"]


def test_classify_technical_wins_over_semantic():
    assert classify_track_group(["Nas", "Kendrick Lamar"]) == "technical"


def test_classify_semantic():
    assert classify_track_group(["Nas"]) == "semantic"


def test_classify_control():
    assert classify_track_group(["Ice Spice"]) == "control"


def test_classify_control_expanded_tier5():
    assert classify_track_group(["Fetty Wap"]) == "control"
    assert classify_track_group(["Desiigner"]) == "control"


def test_playboi_carti_is_control_not_modern():
    """Tier 5 sampling: structural/timing variance; was previously modern-only."""
    assert classify_track_group(["Playboi Carti"]) == "control"


def test_classify_none():
    assert classify_track_group(["Drake"]) is None


def test_default_quotas_sum_one():
    q = default_quota_fractions()
    assert abs(sum(q.values()) - 1.0) < 1e-9

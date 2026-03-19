"""Tests for evo_rhyme.lm_proposer: ProposerConfig, BarRequest, _build_system_prompt, BarProposer, propose_bars, propose_verse_pool, _clean_line, _filter_bars. Area 9."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from evo_rhyme.lm_proposer import (
    BarProposer,
    BarRequest,
    ProposerConfig,
    ROLE_DESCRIPTIONS,
    _build_system_prompt,
    _clean_line,
    _default_roles,
    _filter_bars,
    _rhyme_groups,
)


def test_proposer_config_defaults():
    cfg = ProposerConfig()
    assert cfg.backend == "openai"
    assert cfg.bars_per_slot == 80
    assert cfg.temperature == 0.95


def test_bar_request_dataclass():
    req = BarRequest(
        theme_keywords=["flow", "money"],
        rhyme_target="cat",
        syllable_range=(8, 14),
        role="flex",
    )
    assert req.theme_keywords == ["flow", "money"]
    assert req.rhyme_target == "cat"
    assert req.syllable_range == (8, 14)
    assert req.role == "flex"


def test_build_system_prompt():
    req = BarRequest(theme_keywords=["pressure"], syllable_range=(6, 12), role="punchline")
    prompt = _build_system_prompt(3, req)
    assert "3" in prompt
    assert "pressure" in prompt
    assert "6" in prompt and "12" in prompt
    assert "punchline" in prompt or ROLE_DESCRIPTIONS["punchline"][:10] in prompt


def test_build_system_prompt_with_rhyme_target():
    req = BarRequest(
        theme_keywords=["flow"],
        rhyme_target="dream",
        syllable_range=(8, 14),
    )
    prompt = _build_system_prompt(5, req)
    assert "dream" in prompt


def test_bar_proposer_construction():
    cfg = ProposerConfig(model="gpt-4.1-nano", bars_per_slot=10)
    proposer = BarProposer(config=cfg)
    assert proposer.cfg.model == "gpt-4.1-nano"
    assert proposer.cfg.bars_per_slot == 10


def test_clean_line_strips_numbering_and_bullets():
    assert _clean_line("1. First bar here") == "First bar here"
    assert _clean_line("  - bullet bar") == "bullet bar"
    assert _clean_line('"quoted bar"') == "quoted bar"


def test_filter_bars_accepts_valid_bars():
    """_filter_bars keeps lines in syllable range with pronounceable end words."""
    req = BarRequest(theme_keywords=["flow"], syllable_range=(8, 14))
    raw = [
        "I got the flow when I step in the spot",
        "You know I rock it hard when I hit the block",
    ]
    accepted = _filter_bars(raw, req)
    assert len(accepted) >= 1


def test_filter_bars_rejects_duplicates():
    req = BarRequest(theme_keywords=["flow"], syllable_range=(6, 18))
    raw = [
        "The burns of life bring many concerns",
        "The burns of life bring many concerns",
    ]
    accepted = _filter_bars(raw, req)
    assert len(accepted) <= 1


def test_rhyme_groups_scheme_aabb():
    assert _rhyme_groups("AABB", 4) == [(0, 1), (2, 3)]


def test_rhyme_groups_scheme_abab():
    assert _rhyme_groups("ABAB", 4) == [(0, 2), (1, 3)]


def test_default_roles():
    assert _default_roles(4) == ["setup", "flex", "flex", "punchline"]
    assert len(_default_roles(8)) == 8


def test_propose_bars_with_mocked_api():
    """propose_bars returns bars when _call_api is mocked to return valid lines."""
    valid_bars = [
        "I got the flow when I step in the spot",
        "You know I rock it hard when I hit the block",
    ]

    async def mock_call_api(system: str, n: int):
        return valid_bars[:n]

    proposer = BarProposer(config=ProposerConfig(bars_per_slot=5, batch_size=5))
    with patch.object(proposer, "_call_api", new_callable=AsyncMock, side_effect=mock_call_api):
        result = proposer.propose_bars(
            BarRequest(theme_keywords=["flow"], syllable_range=(8, 14))
        )
    assert isinstance(result, list)
    assert len(result) >= 1


def test_propose_bars_empty_response_returns_empty_list():
    """When API returns empty, propose_bars returns empty list."""

    async def mock_call_api_empty(system: str, n: int):
        return []

    proposer = BarProposer(config=ProposerConfig(bars_per_slot=5, batch_size=5))
    with patch.object(proposer, "_call_api", new_callable=AsyncMock, side_effect=mock_call_api_empty):
        result = proposer.propose_bars(
            BarRequest(theme_keywords=["flow"], syllable_range=(8, 14))
        )
    assert result == []


def test_propose_verse_pool_with_mocked_api():
    """propose_verse_pool returns slot -> list of bars when API is mocked."""
    valid_bars = [
        "I got the flow when I step in the spot",
        "You know I rock it hard when I hit the block",
    ]

    async def mock_propose_bars_async(request):
        return valid_bars

    proposer = BarProposer(config=ProposerConfig(bars_per_slot=10, batch_size=5))
    with patch.object(proposer, "_propose_bars_async", new_callable=AsyncMock, side_effect=mock_propose_bars_async):
        pool = proposer.propose_verse_pool(
            theme_keywords=["flow"],
            scheme="AABB",
            num_lines=4,
        )
    assert isinstance(pool, dict)
    assert set(pool.keys()) == {0, 1, 2, 3}
    for idx in range(4):
        assert isinstance(pool[idx], list)
        assert len(pool[idx]) >= 1


def test_propose_verse_pool_batch_exception_handled():
    """When one batch raises, propose_verse_pool still returns pool with other slots."""
    valid_bars = ["I got the flow when I step in the spot"]

    call_count = [0]

    async def mock_propose_bars_async(request):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("simulated API failure")
        return valid_bars

    proposer = BarProposer(config=ProposerConfig(bars_per_slot=10, batch_size=5))
    with patch.object(proposer, "_propose_bars_async", new_callable=AsyncMock, side_effect=mock_propose_bars_async):
        pool = proposer.propose_verse_pool(
            theme_keywords=["flow"],
            scheme="AABB",
            num_lines=4,
        )
    assert isinstance(pool, dict)
    assert 0 in pool and 1 in pool and 2 in pool and 3 in pool

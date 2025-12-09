#!/usr/bin/env python
"""
generate_story_verse.py

Story-focused verse generator built on top of generate_rhymed_verse.py.

- Uses the same QLoRA Qwen model + LoRA adapter.
- Reuses rhyme memory, rhyme filtering, and candidate scoring logic.
- Adds story beats per bar to keep a clear narrative arc.
- Still enforces a rhyme scheme (e.g. AAAA, AABB, ABAB).
- In this version, RHYME STRUCTURE is weighted significantly more
  than story content, but story is still present.

Example:

  python generate_story_verse.py \
    --artist kendrick \
    --seed "demons in the mirror and pressure on my chest" \
    --scheme AAAA \
    --num_bars 12 \
    --candidates 8 \
    --max_new_tokens 32 \
    --temperature 0.8 \
    --top_p 0.9
"""

import argparse
import math
from collections import defaultdict
from typing import List, Tuple

import generate_rhymed_verse as grv


# ---------------------------------------------------------------------------
# Story beat planning
# ---------------------------------------------------------------------------

def build_story_beats(theme: str, num_bars: int) -> List[str]:
    """
    Create a list of story beats for each bar, tied to the theme.
    If num_bars > template, the beats repeat; if <, they are truncated.
    """
    base_beats = [
        f"Set the scene and atmosphere around: {theme}",
        f"Describe your mental and emotional state related to: {theme}",
        f"Introduce a conflict or main problem connected to: {theme}",
        f"Show how you cope (good or bad) with: {theme}",
        f"Flash back to the origin or cause behind: {theme}",
        f"Describe the lowest point or biggest pressure related to: {theme}",
        f"Show a turning point or realization about: {theme}",
        f"Show consequences or fallout that follow from: {theme}",
        f"Describe growth, resilience, or defiance in spite of: {theme}",
        f"Connect the struggle to a larger context (society, family, legacy) about: {theme}",
        f"Show a glimpse of hope or future vision beyond: {theme}",
        f"End with a reflective or quotable line that sums up: {theme}",
    ]

    beats: List[str] = []
    i = 0
    while len(beats) < num_bars:
        beats.append(base_beats[i % len(base_beats)])
        i += 1
    return beats[:num_bars]


# ---------------------------------------------------------------------------
# Story-aware candidate scoring (heavily rhyme-weighted)
# ---------------------------------------------------------------------------

def story_candidate_score(
    letter: str,
    bar: str,
    rhyme_memory,
    prev_bars: List[str],
    seed: str,
) -> float:
    """
    Composite score for STORY MODE where RHYME is much more important
    than pure story content.

    Components:
      - End-rhyme consistency with this scheme letter (A/B/…)
      - Internal rhymes / multis (more is better)
      - Theme similarity to the seed (Siamese)
      - Coherence with recent bars (Siamese)

    Weighting:
      rhyme_term  (end + internal) gets a high weight.
      story_term  (theme + coherence) gets a smaller weight.
    """
    if not bar or not bar.strip():
        return -1e9

    # Hard filters
    if grv.is_meta_bar(bar):
        return -1e9

    filler_penalty = grv.FILLER_PENALTY if grv.is_filler_bar(bar) else 1.0

    # --- Rhyme components ---
    end_score = grv.end_rhyme_consistency_score(letter, bar, rhyme_memory)

    internal_hits = grv.count_internal_hits(bar)
    internal_score = math.log(1 + internal_hits)

    # --- Story / coherence components ---
    theme_sim = 0.0
    if grv.SIAMESE_SCORER.enabled and seed:
        # Similarity to the main theme line (not the beat text)
        theme_sim = grv.SIAMESE_SCORER.score_pair(seed, bar)

    coh = grv.coherence_score(bar, prev_bars, seed)

    # Use the same base weights from grv but modulate them.
    rhyme_term = (
        grv.END_RHYME_WEIGHT * end_score +
        grv.INTERNAL_RHYME_WEIGHT * internal_score
    )

    story_term = (
        grv.COHERENCE_WEIGHT * coh +
        grv.THEME_SIM_WEIGHT * theme_sim
    )

    # Heavily prioritize rhyme, but keep story in the mix.
    # You can tweak these two multipliers if you want even more/less story.
    total = 2.0 * rhyme_term + 0.6 * story_term

    return total * filler_penalty


def select_bar_with_story(
    letter: str,
    candidates: List[str],
    rhyme_memory,
    prev_bars: List[str],
    seed: str,
) -> str:
    """
    Story-aware candidate selector:

    1) Remove meta / filler bars.
    2) Enforce rhyme scheme via filter_candidates_by_rhyme_letter.
    3) Score via story_candidate_score (rhyme >> story).
    4) Choose highest scoring bar.
    """
    # First pass: remove obvious meta / filler
    filtered = [
        c for c in candidates
        if c and not grv.is_meta_bar(c) and not grv.is_filler_bar(c)
    ]

    # If everything gets filtered out, fall back to only meta-filter
    if not filtered:
        filtered = [c for c in candidates if c and not grv.is_meta_bar(c)]

    if not filtered:
        return ""

    # Enforce rhyme-group / suffix consistency for this scheme letter
    rhyme_filtered = grv.filter_candidates_by_rhyme_letter(
        letter=letter,
        candidates=filtered,
        rhyme_memory=rhyme_memory,
    )

    pool = rhyme_filtered if rhyme_filtered else filtered

    scored = []
    for bar in pool:
        sc = story_candidate_score(
            letter=letter,
            bar=bar,
            rhyme_memory=rhyme_memory,
            prev_bars=prev_bars,
            seed=seed,
        )
        scored.append((bar, sc))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else ""


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a story-focused rap verse with a rhyme scheme."
    )
    p.add_argument("--artist", type=str, required=True)
    p.add_argument("--section", type=str, default="VERSE")
    p.add_argument("--seed", type=str, required=True, help="Theme / central line.")
    p.add_argument("--scheme", type=str, default="AAAA")
    p.add_argument("--num_bars", type=int, default=12)

    p.add_argument("--candidates", type=int, default=grv.DEFAULT_CANDIDATES_PER_BAR)
    p.add_argument("--max_new_tokens", type=int, default=grv.DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=grv.DEFAULT_TEMPERATURE)
    p.add_argument(
        "--top_p",
        type=float,
        default=grv.DEFAULT_TOP_P,
    )
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=grv.DEFAULT_REPETITION_PENALTY,
    )

    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--verse_accept_threshold", type=float, default=0.55)
    return p.parse_args()


def generate_story_verse(
    artist: str,
    section: str,
    seed: str,
    scheme: str,
    num_bars: int,
    candidates_per_bar: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    attempts: int,
    verse_accept_threshold: float,
):
    """
    Full story-verse generation loop:

    - Multiple attempts, keep best verse by Siamese coherence score.
    - Within each attempt, use story beats, rhyme memory, and heavily
      rhyme-weighted scoring.
    """
    artist_token = artist.strip().lower().replace(" ", "_")
    section_token = section.strip().upper()
    seed = seed.strip()
    scheme = scheme.strip().upper()

    scheme_letters = grv.build_scheme_letters(scheme, num_bars)
    story_beats = build_story_beats(seed, num_bars)

    print("=== STORY VERSE CONFIG ===")
    print(f"Artist   : {artist}")
    print(f"Section  : {section_token}")
    print(f"Theme    : {seed}")
    print(f"Scheme   : {''.join(scheme_letters)}")
    print(f"Bars     : {num_bars}")
    print("==========================\n")

    # Load model + adapter once
    print("__ Loading rap model + LoRA (shared with generate_rhymed_verse) ...")
    tokenizer, model = grv.load_rap_model(grv.ADAPTER_DIR)

    best_verse: List[Tuple[str, str]] = []
    best_score = -1e9

    for attempt in range(1, attempts + 1):
        print(f"\n===== STORY VERSE ATTEMPT {attempt}/{attempts} =====")
        rhyme_memory = defaultdict(list)
        verse_bars: List[Tuple[str, str]] = []
        prev_bars_text: List[str] = []

        for idx, (letter, beat) in enumerate(zip(scheme_letters, story_beats), start=1):
            print(f"[Bar {idx}/{len(scheme_letters)}] Rhyme letter: {letter}")
            print(f"  Story beat: {beat}")

            # For generation, fold theme + beat into the "seed" for this bar
            story_seed = f"{seed} || {beat}"

            bar_candidates = grv.generate_bar_candidates(
                artist_token=artist_token,
                section_token=section_token,
                seed=story_seed,
                tokenizer=tokenizer,
                model=model,
                num_candidates=candidates_per_bar,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                bar_index=idx,
                scheme=scheme,
            )

            bar = select_bar_with_story(
                letter=letter,
                candidates=bar_candidates,
                rhyme_memory=rhyme_memory,
                prev_bars=prev_bars_text,
                seed=seed,  # main theme (not the beat string)
            )

            # If everything failed, fall back to first raw candidate to avoid empty bars
            if not bar and bar_candidates:
                bar = bar_candidates[0].strip()

            grv.update_rhyme_memory(letter, bar, rhyme_memory)
            verse_bars.append((letter, bar))
            prev_bars_text.append(bar)

            print(f"  -> Selected: {bar}\n")

        verse_score = grv.verse_siamese_rhyme_score(verse_bars, grv.SIAMESE_SCORER)
        print(f"--> Verse-level Siamese rhyme/story score: {verse_score:.3f}")

        if verse_score > best_score:
            best_score = verse_score
            best_verse = verse_bars

        if verse_score >= verse_accept_threshold:
            print(
                f"✅ Verse accepted (score {verse_score:.3f} "
                f"≥ threshold {verse_accept_threshold:.3f})"
            )
            break
        else:
            print(
                f"❌ Verse rejected (score {verse_score:.3f} "
                f"< threshold {verse_accept_threshold:.3f})"
            )

    if not best_verse:
        print("[WARN] No verse generated.")
        return

    print("\n=== STORY VERSE ===")
    for i, (letter, bar) in enumerate(best_verse, start=1):
        print(f"{i:02d} [{letter}] {bar}")

    grv.analyse_verse(best_verse)


def main():
    args = parse_args()

    print(f"__ Loaded {len(grv.RHYME_GROUPS)} rhyme-group entries from {grv.RHYME_GROUP_CSV}")
    print(f"__ Loading Siamese rhyme model from {grv.SIAMESE_MODEL_DIR} ...")
    if grv.SIAMESE_SCORER.enabled:
        print("__ Siamese rhyme scorer ready.\n")
    else:
        print("[WARN] Siamese scorer disabled; story coherence scoring will be weaker.\n")

    generate_story_verse(
        artist=args.artist,
        section=args.section,
        seed=args.seed,
        scheme=args.scheme,
        num_bars=args.num_bars,
        candidates_per_bar=args.candidates,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        attempts=args.attempts,
        verse_accept_threshold=args.verse_accept_threshold,
    )


if __name__ == "__main__":
    main()

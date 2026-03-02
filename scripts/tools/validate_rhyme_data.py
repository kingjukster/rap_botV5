#!/usr/bin/env python
"""
validate_rhyme_data.py

Validation tool to check existing rhyme data files and report issues.
Do not trust existing data without validation.

Checks:
- rhymes_grouped.csv format and quality
- Siamese model directory (if exists)
- Sample rhyme group quality using known rhyme pairs
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import pronouncing

try:
    from rapbot.rhyme_scorer import SiameseRhymeScorer, load_rhyme_groups
except ImportError:
    SiameseRhymeScorer = None
    load_rhyme_groups = None


WORD_RE = re.compile(r"[A-Za-z']+")


def extract_last_word(text: str) -> str:
    """Extract the last word from a line."""
    words = WORD_RE.findall(text.lower())
    return words[-1] if words else ""


def validate_csv_format(csv_path: Path) -> Tuple[bool, List[str]]:
    """
    Check CSV format correctness.
    Returns: (is_valid, list_of_issues)
    """
    issues = []
    
    if not csv_path.exists():
        return False, [f"CSV file does not exist: {csv_path}"]
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, [f"Failed to read CSV: {e}"]
    
    # Check required columns
    required_cols = {"word", "group"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    if "word" not in df.columns or "group" not in df.columns:
        return False, issues
    
    # Check for empty dataframe
    if len(df) == 0:
        issues.append("CSV is empty")
        return len(issues) == 0, issues
    
    # Check for duplicate word entries (same word, different groups)
    word_to_groups = defaultdict(set)
    for _, row in df.iterrows():
        word = str(row["word"]).strip().lower()
        try:
            group = int(row["group"])
            word_to_groups[word].add(group)
        except (ValueError, KeyError):
            continue
    
    duplicate_words = {w: list(gs) for w, gs in word_to_groups.items() if len(gs) > 1}
    if duplicate_words:
        sample = dict(list(duplicate_words.items())[:10])
        issues.append(f"Found {len(duplicate_words)} words with multiple groups. Sample: {sample}")
    
    # Check for invalid group IDs (non-numeric)
    invalid_groups = []
    for _, row in df.iterrows():
        try:
            group = int(row["group"])
            if group < 0:
                invalid_groups.append((row["word"], group))
        except (ValueError, TypeError):
            invalid_groups.append((row.get("word", "?"), row.get("group", "?")))
    
    if invalid_groups:
        issues.append(f"Found {len(invalid_groups)} entries with invalid group IDs. Sample: {invalid_groups[:10]}")
    
    # Check for empty words
    empty_words = df[df["word"].astype(str).str.strip() == ""]
    if len(empty_words) > 0:
        issues.append(f"Found {len(empty_words)} entries with empty words")
    
    return len(issues) == 0, issues


def test_known_rhyme_pairs(rhyme_groups: Dict[str, int]) -> Tuple[int, int, List[str]]:
    """
    Test known rhyme pairs to check data quality.
    Returns: (correct_count, total_count, list_of_issues)
    """
    # Known exact rhymes
    known_rhymes = [
        ("cat", "hat"),
        ("time", "rhyme"),
        ("love", "dove"),
        ("night", "light"),
        ("day", "way"),
        ("man", "can"),
        ("see", "me"),
        ("go", "show"),
        ("know", "flow"),
        ("mind", "find"),
    ]
    
    # Known non-rhymes
    known_non_rhymes = [
        ("cat", "dog"),
        ("time", "space"),
        ("love", "hate"),
        ("night", "day"),
    ]
    
    issues = []
    correct = 0
    total = 0
    
    # Test known rhymes
    for word1, word2 in known_rhymes:
        total += 1
        g1 = rhyme_groups.get(word1.lower())
        g2 = rhyme_groups.get(word2.lower())
        
        if g1 is not None and g2 is not None:
            if g1 == g2:
                correct += 1
            else:
                issues.append(f"Known rhyme pair ('{word1}', '{word2}') has different groups: {g1} vs {g2}")
        elif g1 is None and g2 is None:
            # Both missing - can't validate, but not necessarily wrong
            issues.append(f"Known rhyme pair ('{word1}', '{word2}') both missing from rhyme groups")
        elif g1 is None:
            issues.append(f"Known rhyme word '{word1}' missing from rhyme groups")
        elif g2 is None:
            issues.append(f"Known rhyme word '{word2}' missing from rhyme groups")
    
    # Test known non-rhymes (should have different groups if both present)
    for word1, word2 in known_non_rhymes:
        total += 1
        g1 = rhyme_groups.get(word1.lower())
        g2 = rhyme_groups.get(word2.lower())
        
        if g1 is not None and g2 is not None:
            if g1 == g2:
                issues.append(f"Known non-rhyme pair ('{word1}', '{word2}') has same group: {g1}")
            else:
                correct += 1
        # If one or both missing, can't validate - not counted as error
    
    return correct, total, issues


def validate_siamese_model(model_dir: Path) -> Tuple[bool, List[str]]:
    """
    Check if Siamese model directory exists and is loadable.
    Returns: (is_valid, list_of_issues)
    """
    issues = []
    
    if not model_dir.exists():
        return False, [f"Siamese model directory does not exist: {model_dir}"]
    
    # Check for required files
    required_files = ["config.json", "tokenizer.json"]
    missing_files = []
    for fname in required_files:
        if not (model_dir / fname).exists():
            missing_files.append(fname)
    
    if missing_files:
        issues.append(f"Missing required model files: {missing_files}")
    
    # Try to load the model
    if SiameseRhymeScorer is None:
        issues.append("Cannot import SiameseRhymeScorer - skipping model load test")
        return len(issues) == 0, issues
    
    try:
        scorer = SiameseRhymeScorer(str(model_dir), device="cpu")
        
        # Test with sample words
        test_words = ["cat", "hat", "dog"]
        for word in test_words:
            try:
                emb = scorer.embed(word)
                if emb.numel() == 0:
                    issues.append(f"Model embedding for '{word}' is empty")
            except Exception as e:
                issues.append(f"Failed to embed '{word}': {e}")
        
        # Test similarity scoring
        try:
            sim = scorer.score_pair("cat", "hat")
            if not isinstance(sim, (int, float)):
                issues.append(f"Model similarity score is not numeric: {type(sim)}")
            elif sim < -1 or sim > 1:
                issues.append(f"Model similarity score out of range [-1, 1]: {sim}")
        except Exception as e:
            issues.append(f"Failed to score pair: {e}")
            
    except Exception as e:
        issues.append(f"Failed to load Siamese model: {e}")
    
    return len(issues) == 0, issues


def phonetic_rhyme_check(word1: str, word2: str) -> Optional[bool]:
    """
    Use pronouncing library to check if two words rhyme.
    Returns: True if rhyme, False if not, None if can't determine.
    """
    phones1 = pronouncing.phones_for_word(word1.lower())
    phones2 = pronouncing.phones_for_word(word2.lower())
    
    if not phones1 or not phones2:
        return None
    
    # Get rhyming parts
    rhyme1 = pronouncing.rhyming_part(phones1[0])
    rhyme2 = pronouncing.rhyming_part(phones2[0])
    
    if rhyme1 and rhyme2:
        return rhyme1 == rhyme2
    
    return None


def validate_rhyme_groups_phonetically(rhyme_groups: Dict[str, int], sample_size: int = 50) -> List[str]:
    """
    Sample rhyme groups and check if words in same group actually rhyme phonetically.
    Returns: list of issues
    """
    issues = []
    
    # Group words by group ID
    group_to_words = defaultdict(list)
    for word, group_id in rhyme_groups.items():
        group_to_words[group_id].append(word)
    
    # Sample groups with multiple words
    multi_word_groups = {gid: words for gid, words in group_to_words.items() if len(words) >= 2}
    
    if not multi_word_groups:
        issues.append("No groups with multiple words found - cannot validate phonetically")
        return issues
    
    sampled_groups = dict(list(multi_word_groups.items())[:sample_size])
    
    non_rhyming_pairs = []
    for group_id, words in sampled_groups.items():
        # Check a few pairs within the group
        for i in range(min(3, len(words))):
            for j in range(i + 1, min(i + 4, len(words))):
                word1, word2 = words[i], words[j]
                rhyme_result = phonetic_rhyme_check(word1, word2)
                if rhyme_result is False:
                    non_rhyming_pairs.append((group_id, word1, word2))
    
    if non_rhyming_pairs:
        sample = non_rhyming_pairs[:10]
        issues.append(f"Found {len(non_rhyming_pairs)} word pairs in same group that don't rhyme phonetically. Sample: {sample}")
    
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Validate rhyme data files (CSV and Siamese model)"
    )
    parser.add_argument(
        "--rhyme_csv",
        type=str,
        default=None,
        help="Path to rhymes_grouped.csv (default: from config or data/rhymes_grouped.csv)",
    )
    parser.add_argument(
        "--siamese_dir",
        type=str,
        default=None,
        help="Path to Siamese model directory (default: from config or rhyme_siamese/)",
    )
    parser.add_argument(
        "--phonetic_check",
        action="store_true",
        help="Also check if words in same groups rhyme phonetically",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Number of groups to sample for phonetic validation",
    )
    
    args = parser.parse_args()
    
    # Determine paths
    if args.rhyme_csv:
        rhyme_csv = Path(args.rhyme_csv)
    else:
        try:
            from config.settings import load_settings
            cfg = load_settings()
            rhyme_csv = cfg.rhyme_groups_csv
        except Exception:
            rhyme_csv = ROOT / "data" / "rhymes_grouped.csv"
    
    if args.siamese_dir:
        siamese_dir = Path(args.siamese_dir)
    else:
        try:
            from config.settings import load_settings
            cfg = load_settings()
            siamese_dir = cfg.siamese_model_dir
        except Exception:
            siamese_dir = ROOT / "rhyme_siamese"
    
    print("=" * 70)
    print("RHYME DATA VALIDATION")
    print("=" * 70)
    print(f"Rhyme CSV: {rhyme_csv}")
    print(f"Siamese model: {siamese_dir}")
    print()
    
    all_issues = []
    all_warnings = []
    
    # Validate CSV format
    print("1. Validating CSV format...")
    is_valid, issues = validate_csv_format(rhyme_csv)
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"   [ISSUE] {issue}")
    else:
        print("   [OK] CSV format is valid")
    print()
    
    # Load and test rhyme groups
    if rhyme_csv.exists() and is_valid:
        print("2. Testing rhyme group quality...")
        try:
            if load_rhyme_groups:
                rhyme_groups = load_rhyme_groups(rhyme_csv)
            else:
                df = pd.read_csv(rhyme_csv)
                rhyme_groups = {str(row["word"]).strip().lower(): int(row["group"]) for _, row in df.iterrows()}
            
            print(f"   Loaded {len(rhyme_groups)} rhyme group entries")
            
            # Test known rhyme pairs
            correct, total, issues = test_known_rhyme_pairs(rhyme_groups)
            if issues:
                all_issues.extend(issues)
                for issue in issues[:10]:  # Show first 10
                    print(f"   [ISSUE] {issue}")
                if len(issues) > 10:
                    print(f"   ... and {len(issues) - 10} more issues")
            
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"   Known rhyme pair accuracy: {correct}/{total} ({accuracy:.1f}%)")
            
            # Phonetic validation if requested
            if args.phonetic_check:
                print("3. Phonetic validation of rhyme groups...")
                phonetic_issues = validate_rhyme_groups_phonetically(rhyme_groups, args.sample_size)
                if phonetic_issues:
                    all_issues.extend(phonetic_issues)
                    for issue in phonetic_issues[:5]:
                        print(f"   [ISSUE] {issue}")
                    if len(phonetic_issues) > 5:
                        print(f"   ... and {len(phonetic_issues) - 5} more issues")
                else:
                    print("   [OK] Phonetic validation passed")
            
        except Exception as e:
            all_issues.append(f"Failed to load rhyme groups: {e}")
            print(f"   [ERROR] {e}")
        print()
    
    # Validate Siamese model
    print("4. Validating Siamese model...")
    is_valid_model, issues = validate_siamese_model(siamese_dir)
    if issues:
        all_warnings.extend(issues)  # Model issues are warnings, not critical
        for issue in issues:
            print(f"   [WARNING] {issue}")
    else:
        print("   [OK] Siamese model is valid and loadable")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if all_issues:
        print(f"[FAIL] Found {len(all_issues)} critical issues")
        print("\nRecommendation: Rebuild rhyme groups using rebuild_rhyme_groups.py")
    else:
        print("[PASS] No critical issues found")
    
    if all_warnings:
        print(f"[WARN] Found {len(all_warnings)} warnings (non-critical)")
    
    if not all_issues and not all_warnings:
        print("[OK] All validations passed")
    
    return 0 if not all_issues else 1


if __name__ == "__main__":
    sys.exit(main())

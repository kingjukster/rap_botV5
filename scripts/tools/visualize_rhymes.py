#!/usr/bin/env python
"""
visualize_rhymes.py

Visualization tool to highlight rhyme patterns in text.
Shows rhyme scheme, highlights rhyming words, and displays confidence scores.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from typing import List, Dict

from rapbot.rhyme_detector import RhymeDetector, RhymeType


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Rhyme types
    EXACT = "\033[92m"  # Green
    SLANT = "\033[93m"  # Yellow
    ASSONANCE = "\033[94m"  # Blue
    CONSONANCE = "\033[95m"  # Magenta
    INTERNAL = "\033[96m"  # Cyan
    
    # Backgrounds
    BG_EXACT = "\033[42m"
    BG_SLANT = "\033[43m"
    BG_INTERNAL = "\033[46m"


def colorize_word(word: str, rhyme_type: RhymeType, is_internal: bool = False) -> str:
    """Colorize a word based on rhyme type."""
    if is_internal:
        color = Colors.INTERNAL
    elif rhyme_type == RhymeType.EXACT:
        color = Colors.EXACT
    elif rhyme_type == RhymeType.SLANT:
        color = Colors.SLANT
    elif rhyme_type == RhymeType.ASSONANCE:
        color = Colors.ASSONANCE
    elif rhyme_type == RhymeType.CONSONANCE:
        color = Colors.CONSONANCE
    else:
        return word
    
    return f"{color}{word}{Colors.RESET}"


def highlight_rhymes_in_line(
    line: str,
    detector: RhymeDetector,
    internal_rhymes: List,
) -> str:
    """Highlight rhyming words in a line."""
    words = detector._extract_words(line)
    
    # Create mapping of word -> rhyme info
    word_rhyme_info = {}
    for rhyme in internal_rhymes:
        word_rhyme_info[rhyme.word1.lower()] = (rhyme.rhyme_type, True)
        word_rhyme_info[rhyme.word2.lower()] = (rhyme.rhyme_type, True)
    
    # Replace words in line with colorized versions
    result = line
    for word in words:
        word_lower = word.lower()
        if word_lower in word_rhyme_info:
            rhyme_type, is_internal = word_rhyme_info[word_lower]
            colored = colorize_word(word, rhyme_type, is_internal)
            # Replace word (case-insensitive)
            import re
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            result = pattern.sub(colored, result, count=1)
    
    return result


def visualize_verse(lines: List[str], detector: RhymeDetector, show_internal: bool = True):
    """Visualize rhyme patterns in a verse."""
    print("\n" + "=" * 70)
    print("RHYME VISUALIZATION")
    print("=" * 70)
    
    analysis = detector.analyze_verse(lines)
    
    # Print lines with highlighted rhymes
    print("\nLines with highlighted rhymes:")
    print("-" * 70)
    for i, line in enumerate(lines):
        internal = []
        for internal_data in analysis["internal_rhymes"]:
            if internal_data["line"] == i:
                internal = internal_data["rhymes"]
        
        if show_internal and internal:
            highlighted = highlight_rhymes_in_line(line, detector, internal)
            print(f"{i+1:2d}. {highlighted}")
        else:
            print(f"{i+1:2d}. {line}")
    
    # Print end rhyme analysis
    print("\n" + "-" * 70)
    print("End Rhyme Analysis:")
    print("-" * 70)
    for end_rhyme in analysis["end_rhymes"]:
        line1_idx = end_rhyme["line1"]
        line2_idx = end_rhyme["line2"]
        result = end_rhyme["result"]
        
        line1_word = detector._get_last_word(lines[line1_idx]) or "?"
        line2_word = detector._get_last_word(lines[line2_idx]) or "?"
        
        rhyme_type_str = result.rhyme_type.value
        similarity = result.similarity
        confidence = result.confidence
        
        print(f"  Line {line1_idx+1} ({line1_word}) <-> Line {line2_idx+1} ({line2_word}): "
              f"{colorize_word(rhyme_type_str, result.rhyme_type)} "
              f"(similarity: {similarity:.2f}, confidence: {confidence:.2f})")
    
    # Print internal rhymes
    if show_internal and analysis["internal_rhymes"]:
        print("\n" + "-" * 70)
        print("Internal Rhymes:")
        print("-" * 70)
        for internal_data in analysis["internal_rhymes"]:
            line_idx = internal_data["line"]
            rhymes = internal_data["rhymes"]
            
            print(f"  Line {line_idx+1}:")
            for rhyme in rhymes:
                print(f"    {rhyme.word1} <-> {rhyme.word2}: "
                      f"{colorize_word(rhyme.rhyme_type.value, rhyme.rhyme_type)} "
                      f"(similarity: {rhyme.similarity:.2f})")
    
    # Print rhyme scheme
    print("\n" + "-" * 70)
    print(f"Rhyme Scheme: {Colors.BOLD}{analysis['rhyme_scheme']}{Colors.RESET}")
    print("=" * 70)
    print()


def visualize_from_file(file_path: Path, detector: RhymeDetector):
    """Visualize rhymes from a file (one verse per line or JSONL)."""
    if file_path.suffix == ".jsonl":
        # JSONL format
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    lines = [data.get("line1", ""), data.get("line2", "")]
                    if all(lines):
                        visualize_verse(lines, detector)
    else:
        # Plain text, one line per verse line
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines:
                visualize_verse(lines, detector)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize rhyme patterns in text"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Input file (JSONL or text) or '-' for stdin",
    )
    parser.add_argument(
        "--lines",
        type=str,
        nargs="+",
        default=None,
        help="Lines to analyze (as arguments)",
    )
    parser.add_argument(
        "--rhyme_csv",
        type=str,
        default=None,
        help="Path to rhymes_grouped.csv (optional)",
    )
    parser.add_argument(
        "--no-internal",
        action="store_true",
        help="Don't show internal rhymes",
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    rhyme_csv = Path(args.rhyme_csv) if args.rhyme_csv else None
    detector = RhymeDetector(rhyme_groups_csv=rhyme_csv)
    
    # Get input
    if args.lines:
        lines = args.lines
        visualize_verse(lines, detector, show_internal=not args.no_internal)
    elif args.input:
        if args.input == "-":
            # Read from stdin
            lines = [l.strip() for l in sys.stdin if l.strip()]
            if lines:
                visualize_verse(lines, detector, show_internal=not args.no_internal)
        else:
            file_path = Path(args.input)
            if file_path.exists():
                visualize_from_file(file_path, detector)
            else:
                print(f"[ERROR] File not found: {file_path}")
                return 1
    else:
        # Interactive mode
        print("Enter lines (empty line to finish):")
        lines = []
        while True:
            try:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            except EOFError:
                break
        
        if lines:
            visualize_verse(lines, detector, show_internal=not args.no_internal)
        else:
            print("[ERROR] No lines provided")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

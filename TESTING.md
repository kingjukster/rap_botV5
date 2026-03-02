# Testing Guide for Rhyme Detection System

This guide explains how to test the rhyme detection system.

## Quick Start

### 1. Run Unit Tests

```bash
# Run all unit tests
python -m pytest tests/test_rhyme_detector.py -v

# Run specific test class
python -m pytest tests/test_rhyme_detector.py::TestEndRhymes -v

# Run with coverage
python -m pytest tests/test_rhyme_detector.py --cov=rapbot.rhyme_detector --cov-report=html
```

### 2. Run Integration Tests

```bash
# Run integration tests with test dataset
python -m pytest tests/test_rhyme_integration.py -v
```

### 3. Validate Existing Data

```bash
# Validate rhyme CSV and Siamese model
python scripts/tools/validate_rhyme_data.py

# With custom paths
python scripts/tools/validate_rhyme_data.py --rhyme_csv data/rhymes_grouped.csv --siamese_dir rhyme_siamese

# Include phonetic validation
python scripts/tools/validate_rhyme_data.py --phonetic_check
```

### 4. Evaluate on Test Dataset

```bash
# Run evaluation metrics
python scripts/tools/evaluate_rhyme_detection.py

# With custom test data
python scripts/tools/evaluate_rhyme_detection.py --test_data data/test_rhymes.jsonl

# Save results to file
python scripts/tools/evaluate_rhyme_detection.py --output results/metrics.json
```

### 5. Visualize Rhymes

```bash
# Interactive mode (enter lines manually)
python scripts/tools/visualize_rhymes.py

# From command line
python scripts/tools/visualize_rhymes.py --lines "I'm the best at this test" "I never rest"

# From file
python scripts/tools/visualize_rhymes.py data/test_rhymes.jsonl

# From stdin
echo -e "The cat sat on the mat\nThe hat was flat" | python scripts/tools/visualize_rhymes.py -
```

## Using the API Directly

### Basic Usage

```python
from rapbot.rhyme_detector import RhymeDetector, RhymeType

# Initialize detector (works without rhyme CSV)
detector = RhymeDetector()

# Detect end rhyme
result = detector.detect_end_rhyme("The cat", "The hat")
print(f"Type: {result.rhyme_type}, Similarity: {result.similarity}")

# Detect word rhyme directly
result = detector.detect_word_rhyme("time", "rhyme")
print(f"Type: {result.rhyme_type.value}")

# Detect internal rhymes
line = "I'm the best at this test"
internal = detector.detect_internal_rhymes(line)
for rhyme in internal:
    print(f"{rhyme.word1} <-> {rhyme.word2}: {rhyme.rhyme_type.value}")

# Analyze entire verse
lines = [
    "The cat sat on the mat",
    "The hat was flat",
]
analysis = detector.analyze_verse(lines)
print(f"Rhyme scheme: {analysis['rhyme_scheme']}")
```

### With Rhyme Groups CSV

```python
from pathlib import Path
from rapbot.rhyme_detector import RhymeDetector

# Load with rhyme groups
detector = RhymeDetector(rhyme_groups_csv=Path("data/rhymes_grouped.csv"))

# Detection will use rhyme groups if available, fallback to phonetic
result = detector.detect_word_rhyme("cat", "hat")
```

### With Siamese Model (Hybrid Detection)

```python
from pathlib import Path
from rapbot.rhyme_detector import RhymeDetector

# Enable hybrid detection with Siamese model
detector = RhymeDetector(
    rhyme_groups_csv=Path("data/rhymes_grouped.csv"),
    siamese_model_dir=Path("rhyme_siamese"),
    use_siamese=True,
    hybrid_weight=0.5,  # 50% phonetic, 50% Siamese
)

result = detector.detect_word_rhyme("time", "rhyme")
print(f"Method: {result.method}")  # Will show "hybrid"
```

## Test Examples

### Example 1: Simple End Rhyme

```python
from rapbot.rhyme_detector import RhymeDetector

detector = RhymeDetector()
result = detector.detect_end_rhyme("The cat sat", "The hat was")
assert result.rhyme_type == RhymeType.EXACT
```

### Example 2: Internal Rhymes

```python
detector = RhymeDetector()
line = "I'm the best at this test"
rhymes = detector.detect_internal_rhymes(line)
# Should find: best <-> test
```

### Example 3: Verse Analysis

```python
detector = RhymeDetector()
lines = [
    "I'm the best at this test",
    "I never rest",
    "The flow is slow",
    "I know how to go",
]
analysis = detector.analyze_verse(lines)
print(analysis['rhyme_scheme'])  # e.g., "AABB"
```

## Rebuilding Rhyme Groups

If validation fails, rebuild from scratch:

```bash
# Rebuild using phonetic clustering
python scripts/tools/rebuild_rhyme_groups.py \
    --output_csv data/rhymes_grouped_new.csv \
    --word_list cat hat mat bat rat sat \
    --method phonetic

# Rebuild from corpus
python scripts/tools/rebuild_rhyme_groups.py \
    --output_csv data/rhymes_grouped_new.csv \
    --corpus_path data/elite_kaggle_corpus_clean.txt \
    --min_word_freq 5 \
    --method phonetic
```

## Expected Test Results

- **Unit Tests**: Should pass 11/15 tests (some may fail due to phonetic analysis variations)
- **Integration Tests**: Should achieve >70% accuracy on exact rhymes, >50% on slant rhymes
- **Evaluation**: Should show precision/recall/F1 scores for each rhyme type

## Troubleshooting

### Tests Fail with Import Errors

Make sure you're in the project root:
```bash
cd C:\Users\horne\rap_botV5
python -m pytest tests/
```

### Missing Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

### Phonetic Analysis Not Working

The system uses the `pronouncing` library. If words aren't found:
- Check that `pronouncing` is installed: `pip install pronouncing`
- Some words may not be in CMUdict (falls back to phonetic analysis)

### Siamese Model Not Loading

If using hybrid detection:
- Check that model directory exists
- Model is optional - system works without it using phonetic analysis only

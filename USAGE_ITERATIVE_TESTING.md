# How to Use Iterative ChatGPT Testing

## Quick Start

1. **Set up API key in `.env` file**:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```
   Make sure:
   - No quotes around the key
   - No spaces before/after the key
   - Key starts with `sk-`

2. **Test connection**:
   ```bash
   python scripts/tools/test_iterative_system.py
   ```

3. **Run full iterative testing**:
   ```bash
   python scripts/tools/iterative_tester.py
   ```

## What It Does

The system will:
1. Ask ChatGPT to generate 25 test cases
2. Run your rhyme detector on them
3. Ask ChatGPT to evaluate the results
4. Get code improvement suggestions from ChatGPT
5. Automatically modify `rapbot/rhyme_detector.py` based on suggestions
6. Run unit tests to verify changes
7. Repeat until accuracy reaches 85% or max iterations
8. Request a new test file and repeat

## Example Run

```bash
$ python scripts/tools/iterative_tester.py --max_iterations 5 --accuracy_threshold 0.80

======================================================================
ITERATIVE CHATGPT TESTING
======================================================================
Model: gpt-4
Target accuracy: 80.0%
Max iterations per file: 5

--- Iteration 1/5 ---
[STEP 2] Running tests...
Accuracy: 72.0% (18/25)
[STEP 3] Evaluating results with ChatGPT...
[INFO] Received 3 suggestions
  - [high] Improve slant rhyme threshold
  - [medium] Better multi-syllable detection
[STEP 4] Applying code modifications...
[APPLY] Applied: Improve slant rhyme threshold
[SUCCESS] Applied 1 suggestions. Backup: rapbot/.backups/rhyme_detector_20250101_120000.py

--- Iteration 2/5 ---
[STEP 2] Running tests...
Accuracy: 84.0% (21/25)  # Improved!
...
```

## Safety

- All code changes are backed up automatically
- Syntax is validated before applying
- Unit tests run after each change
- You can restore from backups if needed

## Files Created

- `data/iterative_tests/` - Test files and results
- `rapbot/.backups/` - Code backups (add to .gitignore)

## Tips

- Start with `--max_iterations 3` to test the system
- Use `--apply_priority high` to only apply critical fixes
- Review changes in git before committing
- Check `data/iterative_tests/` for generated test cases

# Iterative ChatGPT Testing System

Automated iterative testing system that uses ChatGPT to generate test cases, evaluate results, and automatically improve the rhyme detector.

## Setup

1. **Install dependencies** (already in requirements.txt):
   ```bash
   pip install openai
   ```

2. **Set up API key**:
   - Create `.env` file in project root:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Or set environment variable:
     ```bash
     export OPENAI_API_KEY=your_api_key_here
     ```

## Quick Test

Test the ChatGPT connection:
```bash
python scripts/tools/test_iterative_system.py
```

## Full Iterative Testing

Run the full iterative testing loop:
```bash
python scripts/tools/iterative_tester.py
```

### Options

```bash
python scripts/tools/iterative_tester.py \
    --model gpt-4 \
    --max_iterations 10 \
    --accuracy_threshold 0.85 \
    --num_test_files 3 \
    --apply_priority high
```

**Parameters**:
- `--model`: ChatGPT model (gpt-4, gpt-3.5-turbo, etc.)
- `--max_iterations`: Max iterations per test file (default: 10)
- `--accuracy_threshold`: Target accuracy (default: 0.85)
- `--num_test_files`: Number of test files to process (default: 3)
- `--apply_priority`: Only apply high/medium/low priority suggestions (default: high)
- `--test_dir`: Directory for test files (default: data/iterative_tests)
- `--target_file`: Code file to modify (default: rapbot/rhyme_detector.py)

## How It Works

1. **Generate Test Cases**: ChatGPT generates diverse test cases (JSONL format)
2. **Run Tests**: Detector runs on test cases, metrics collected
3. **Evaluate**: ChatGPT analyzes results and identifies issues
4. **Suggest Improvements**: ChatGPT provides specific code fixes
5. **Apply Changes**: Code modifier safely applies suggestions
6. **Verify**: Unit tests run to ensure code still works
7. **Iterate**: Repeat until accuracy threshold reached
8. **New Test File**: Request new test file and repeat

## Safety Features

- **Automatic Backups**: Code backed up before each modification
- **Syntax Validation**: All changes validated before applying
- **Rollback**: Can restore from backup if tests fail
- **Iteration Limits**: Stops after max iterations or no improvement
- **Priority Filtering**: Only applies high-priority changes by default

## Output

Test files and results saved to `data/iterative_tests/`:
- `test_file_1_iter_0.jsonl` - Generated test cases
- `test_file_1_iter_1_results.json` - Iteration results
- Backups in `rapbot/.backups/`

## Example Output

```
======================================================================
ITERATIVE CHATGPT TESTING
======================================================================
Model: gpt-4
Target accuracy: 85.0%
Max iterations per file: 10
Number of test files: 3

--- Iteration 1/10 ---
[STEP 2] Running tests...
Accuracy: 72.0% (18/25)
[STEP 3] Evaluating results with ChatGPT...
[INFO] Received 3 suggestions
  - [high] Improve slant rhyme detection threshold
  - [medium] Better handling of multi-syllable rhymes
[STEP 4] Applying code modifications...
[APPLY] Applied: Improve slant rhyme detection threshold
[SUCCESS] Applied 1 suggestions. Backup: rapbot/.backups/rhyme_detector_20250101_120000.py
```

## Manual Review

After iterations, review:
- `data/iterative_tests/` - Test files and results
- `rapbot/.backups/` - Code backups
- Check git diff to see what changed

## Troubleshooting

**API Key Not Found**:
- Check `.env` file exists and has `OPENAI_API_KEY=...`
- Or set environment variable

**Code Modification Fails**:
- Check ChatGPT suggestions are valid Python
- Review `rapbot/.backups/` for previous versions
- Manually apply suggestions if needed

**Tests Fail After Modification**:
- System will warn but continue
- Check unit tests: `python -m pytest tests/`
- Restore from backup if needed

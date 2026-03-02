#!/usr/bin/env python
"""
chatgpt_tester.py

ChatGPT API integration for iterative testing:
- Generate test cases
- Evaluate results
- Suggest improvements
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    # Load .env file
    load_dotenv(ROOT / ".env")
except ImportError:
    # python-dotenv not installed, but that's okay - we'll read .env manually
    pass


@dataclass
class TestCase:
    """A single test case."""
    line1: str
    line2: str
    expected_end_rhyme: str  # EXACT, SLANT, NONE
    expected_internal_rhymes: Optional[List[Dict]] = None
    notes: Optional[str] = None


@dataclass
class TestResults:
    """Results from running tests."""
    total: int
    correct: int
    accuracy: float
    per_type: Dict[str, Dict[str, float]]  # type -> {precision, recall, f1}
    failures: List[Dict]  # List of failed test cases with details


class ChatGPTTester:
    """ChatGPT API client for iterative testing."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize ChatGPT tester.
        
        Args:
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
            max_retries: Maximum retries for API calls
            retry_delay: Delay between retries (seconds)
        """
        # Try loading from .env file first (before env var, to ensure latest)
        api_key = api_key
        env_file = ROOT / ".env"
        if env_file.exists() and not api_key:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        # Remove any comments
                        if "#" in api_key:
                            api_key = api_key.split("#")[0].strip()
                        break
        
        # Fall back to environment variable if not in .env
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found.\n"
                "Set OPENAI_API_KEY environment variable or create .env file with:\n"
                "OPENAI_API_KEY=sk-..."
            )
        
        # Validate key format
        if not api_key.startswith("sk-"):
            print(f"[WARN] API key doesn't start with 'sk-'. Make sure it's a valid OpenAI API key.")
        else:
            # Mask key for display (show first 7 and last 4 chars)
            masked = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
            print(f"[INFO] Using API key: {masked}")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _call_api(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """Make API call with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"[WARN] API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
    
    def generate_test_cases(
        self,
        num_cases: int = 20,
        previous_results: Optional[TestResults] = None,
        iteration: int = 1,
        focus_areas: Optional[List[str]] = None,
    ) -> List[TestCase]:
        """
        Generate test cases using ChatGPT.
        
        Args:
            num_cases: Number of test cases to generate
            previous_results: Results from previous iteration (for focused generation)
            iteration: Current iteration number
            focus_areas: Areas to focus on (e.g., ["slant_rhymes", "internal_rhymes"])
        
        Returns:
            List of TestCase objects
        """
        prompt = f"""You are a test case generator for a rhyme detection system. Generate {num_cases} diverse test cases.

The rhyme detector can identify:
- EXACT rhymes (e.g., "cat"/"hat", "time"/"rhyme")
- SLANT rhymes (e.g., "love"/"move", "orange"/"door hinge")
- Internal rhymes (rhymes within a single line)
- Assonance (vowel-only rhymes)
- Consonance (consonant-only rhymes)
- Multi-syllable rhymes

Generate test cases in JSONL format. Each line should be a JSON object with:
{{
    "line1": "First line of text",
    "line2": "Second line of text",
    "expected": {{
        "end_rhyme": "EXACT|SLANT|NONE",
        "internal_rhymes": [{{"words": ["word1", "word2"], "type": "EXACT|SLANT", "positions": [0, 2]}}]  // optional
    }},
    "notes": "Brief description"
}}

Requirements:
- Include diverse rhyme types
- Mix simple and complex cases
- Include edge cases
- Test internal rhymes within lines
- Test multi-syllable rhymes
- Include some non-rhymes (NONE) as negative examples
"""
        
        if previous_results and previous_results.failures:
            prompt += f"\n\nPrevious iteration had {len(previous_results.failures)} failures. Focus on similar cases to improve detection."
        
        if focus_areas:
            prompt += f"\n\nFocus areas for this iteration: {', '.join(focus_areas)}"
        
        prompt += "\n\nOutput ONLY valid JSONL (one JSON object per line). Do not include any explanation or markdown."
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates test cases in JSONL format."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_api(messages, temperature=0.8)
        
        # Parse JSONL response
        test_cases = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove markdown code blocks if present
            if line.startswith("```"):
                continue
            if line.startswith("```json"):
                continue
            
            try:
                data = json.loads(line)
                test_case = TestCase(
                    line1=data.get("line1", ""),
                    line2=data.get("line2", ""),
                    expected_end_rhyme=data.get("expected", {}).get("end_rhyme", "NONE"),
                    expected_internal_rhymes=data.get("expected", {}).get("internal_rhymes"),
                    notes=data.get("notes"),
                )
                if test_case.line1 and test_case.line2:
                    test_cases.append(test_case)
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse test case line: {line[:100]}... Error: {e}")
                continue
        
        print(f"[INFO] Generated {len(test_cases)} test cases from ChatGPT")
        return test_cases
    
    def evaluate_results(
        self,
        test_results: TestResults,
        detector_outputs: List[Dict],
        current_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ask ChatGPT to evaluate test results and suggest improvements.
        
        Args:
            test_results: TestResults object with metrics
            detector_outputs: Detailed outputs for each test case
            current_code: Current rhyme_detector.py code (optional)
        
        Returns:
            Dictionary with evaluation and suggestions
        """
        prompt = f"""You are evaluating a rhyme detection system. Analyze these test results:

Overall Accuracy: {test_results.accuracy:.1%} ({test_results.correct}/{test_results.total})

Per-Type Performance:
"""
        for rhyme_type, metrics in test_results.per_type.items():
            prompt += f"- {rhyme_type}: Precision={metrics.get('precision', 0):.2f}, Recall={metrics.get('recall', 0):.2f}, F1={metrics.get('f1', 0):.2f}\n"
        
        prompt += f"\nFailures ({len(test_results.failures)}):\n"
        for i, failure in enumerate(test_results.failures[:10], 1):  # Show first 10
            prompt += f"{i}. {failure.get('test_case', {}).get('line1', '?')} / {failure.get('test_case', {}).get('line2', '?')}\n"
            prompt += f"   Expected: {failure.get('expected', '?')}, Got: {failure.get('predicted', '?')}\n"
        
        prompt += """
Analyze the results and provide:
1. What types of rhymes are being detected well?
2. What types are failing?
3. What patterns do you see in the failures?
4. Specific code improvements to fix the issues
5. Priority of fixes (high/medium/low)

Format your response as JSON:
{
    "analysis": "Overall analysis of performance",
    "strengths": ["what works well"],
    "weaknesses": ["what needs improvement"],
    "patterns": ["patterns in failures"],
    "suggestions": [
        {
            "priority": "high|medium|low",
            "issue": "description of issue",
            "fix": "specific code change needed",
            "location": "function/class name or line number range",
            "code": "suggested code snippet"
        }
    ],
    "next_focus": ["areas to focus on in next iteration"]
}
"""
        
        if current_code:
            prompt += f"\n\nCurrent code (for reference):\n```python\n{current_code[:2000]}...\n```"
        
        messages = [
            {"role": "system", "content": "You are an expert code reviewer specializing in rhyme detection algorithms."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_api(messages, temperature=0.5)
        
        # Parse JSON response
        try:
            # Extract JSON from response (might have markdown)
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            evaluation = json.loads(response)
            return evaluation
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse ChatGPT evaluation: {e}")
            print(f"Response: {response[:500]}...")
            return {
                "analysis": response[:500],
                "suggestions": [],
                "next_focus": [],
            }
    
    def generate_new_test_file(
        self,
        previous_performance: Optional[Dict] = None,
    ) -> List[TestCase]:
        """
        Request a new test file from ChatGPT (different focus/complexity).
        
        Args:
            previous_performance: Performance on previous test file
        
        Returns:
            List of TestCase objects
        """
        prompt = """Generate a NEW test file with different characteristics than before.

Focus on:
- Different complexity levels
- Different rhyme type distributions
- Real-world rap/poetry examples
- Challenging edge cases
- Stress testing the detector

Generate 25-30 test cases in JSONL format (same format as before).

Output ONLY valid JSONL. One JSON object per line."""
        
        if previous_performance and isinstance(previous_performance, dict):
            accuracy = previous_performance.get('accuracy', 0)
            if isinstance(accuracy, (int, float)):
                prompt += f"\n\nPrevious test file performance: {accuracy:.1%} accuracy"
                prompt += "\nMake this test file more challenging or focus on different areas."
        
        messages = [
            {"role": "system", "content": "You are a test case generator for rhyme detection."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_api(messages, temperature=0.9)
        
        # Parse JSONL
        test_cases = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            
            try:
                data = json.loads(line)
                test_case = TestCase(
                    line1=data.get("line1", ""),
                    line2=data.get("line2", ""),
                    expected_end_rhyme=data.get("expected", {}).get("end_rhyme", "NONE"),
                    expected_internal_rhymes=data.get("expected", {}).get("internal_rhymes"),
                    notes=data.get("notes"),
                )
                if test_case.line1 and test_case.line2:
                    test_cases.append(test_case)
            except json.JSONDecodeError:
                continue
        
        print(f"[INFO] Generated {len(test_cases)} test cases for new test file")
        return test_cases

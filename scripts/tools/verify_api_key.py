#!/usr/bin/env python
"""Quick script to verify API key is set up correctly without making API calls."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Check environment variable
env_key = os.getenv("OPENAI_API_KEY")

# Check .env file
env_file = ROOT / ".env"
file_key = None
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                file_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if "#" in file_key:
                    file_key = file_key.split("#")[0].strip()
                break

print("=" * 70)
print("API KEY VERIFICATION")
print("=" * 70)

if env_key:
    print(f"[OK] Found in environment variable")
    print(f"     Format: {env_key[:10]}...{env_key[-4:]}")
    print(f"     Length: {len(env_key)}")
    print(f"     Valid format: {env_key.startswith('sk-')}")
elif file_key:
    print(f"[OK] Found in .env file")
    print(f"     Format: {file_key[:10]}...{file_key[-4:]}")
    print(f"     Length: {len(file_key)}")
    print(f"     Valid format: {file_key.startswith('sk-')}")
else:
    print("[ERROR] No API key found")
    print("\nTo fix:")
    print("1. Create or edit .env file in project root")
    print("2. Add line: OPENAI_API_KEY=sk-proj-your-key-here")
    print("3. Run this script again")

print("=" * 70)

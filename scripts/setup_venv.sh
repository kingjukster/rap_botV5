#!/bin/bash
# Run this after: sudo apt install python3.12-venv
# Then: source venv/bin/activate && pip install -e ".[full]"

set -e
cd "$(dirname "$0")/.."

if ! python3 -c "import venv" 2>/dev/null; then
    echo "ERROR: python3-venv not installed."
    echo "Run: sudo apt install python3.12-venv"
    exit 1
fi

echo "Creating venv..."
python3 -m venv venv

echo "Activating and installing..."
source venv/bin/activate
pip install -e ".[full]"

echo ""
echo "✅ Done. Activate with: source venv/bin/activate"
echo "   Then run: python scripts/run_verse_evolution.py --theme \"pressure,mask\" --population 30 --generations 5"

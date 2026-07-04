#!/usr/bin/env bash

# setup.sh - Automated Environment Configuration Script for Linux/macOS
set -e

echo "=========================================================="
echo "          ResearchSphere AI Environment Setup             "
echo "=========================================================="
echo ""

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.11+ and try again."
    exit 1
fi

echo "[1/4] Creating Virtual Environment in '.venv'..."
python3 -m venv .venv

echo "[2/4] Activating Virtual Environment and Upgrading Pip..."
source .venv/bin/activate
pip install --upgrade pip

echo "[3/4] Installing Required Dependencies from 'requirements.txt'..."
pip install -r requirements.txt

echo "[4/4] Creating local environment variables config template if not present..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  -> Created .env from template. Remember to add your API keys!"
else
    echo "  -> .env file already exists. Skipping."
fi

echo ""
echo "=========================================================="
echo "          SETUP COMPLETED SUCCESSFULLY!                   "
echo "=========================================================="
echo ""
echo "To activate your virtual environment, run:"
echo "    source .venv/bin/activate"
echo ""
echo "To run the configuration health check, execute:"
echo "    python main.py"
echo ""
echo "=========================================================="

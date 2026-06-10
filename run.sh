#!/bin/bash
# Boardflow — one-command startup
# Usage: bash run.sh

set -e

echo "🟣 Boardflow starting..."

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Download from https://python.org"
  exit 1
fi

# Check/install Stockfish
if ! command -v stockfish &>/dev/null; then
  echo ""
  echo "⚠️  Stockfish not found. Accuracy scores will be disabled."
  echo "   To install:  brew install stockfish   (Mac)"
  echo "                sudo apt install stockfish (Linux)"
  echo "   Windows: https://stockfishchess.org/download/"
  echo ""
fi

# Install Python deps
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt --quiet

# Launch server
echo ""
echo "✅ Boardflow is running at http://localhost:8000"
echo "   Press Ctrl+C to stop."
echo ""

cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# Boardflow

**Boardflow** turns a screen recording of any chess game into a full analysis report — powered by Roboflow computer vision and Stockfish.

Upload a video → Roboflow reads every piece → Stockfish scores every move → you get Lichess-grade accuracy analysis, instantly.

---

## Setup (5 minutes)

### 1. Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | https://python.org |
| Stockfish | `brew install stockfish` (Mac) · `sudo apt install stockfish` (Linux) · [Windows](https://stockfishchess.org/download/) |
| Roboflow API key | https://app.roboflow.com → top-right → Settings → API |

### 2. Get a chess detection model

You need a Roboflow model trained to detect chess pieces on digital boards (Chess.com / Lichess).

- Browse [Roboflow Universe](https://universe.roboflow.com) and search **"chess pieces"**
- Pick a model, note its **Model ID** and **version number**
- Or use your own trained model from [app.roboflow.com](https://app.roboflow.com)

### 3. Start the app

```bash
# From the boardflow/ folder:
bash run.sh
```

Then open **http://localhost:8000** in your browser.

---

## How to use

1. Screen-record your Chess.com or Lichess game (any screen recorder works — QuickTime, OBS, etc.)
2. Open Boardflow in your browser
3. Drop your video into the upload zone
4. Enter your Roboflow API key and model ID
5. Hit **Analyse Game** — processing takes ~30–60 seconds
6. Review your accuracy, blunders, and best moves

---

## Project structure

```
boardflow/
├── backend/
│   ├── main.py              # FastAPI server — the API layer
│   ├── frame_extractor.py   # Video → key frames (OpenCV)
│   ├── board_detector.py    # Roboflow predictions → board state
│   ├── move_detector.py     # Board state diffs → chess moves
│   └── analyzer.py          # Stockfish eval + Lichess accuracy formula
├── frontend/
│   └── index.html           # Single-page UI
├── requirements.txt
└── run.sh                   # One-command startup
```

---

## Accuracy formula

Boardflow uses the **Lichess open-source accuracy formula**:

```
winPercent(cp) = 50 + 50 × (2 / (1 + e^(−0.00368208 × cp)) − 1)

moveAccuracy = 103.1668 × e^(−0.04354 × max(0, wp_before − wp_after)) − 3.167

gameAccuracy = average(moveAccuracy for all moves by that player)
```

This is identical to the formula Lichess uses in its game analysis.

---

## Tech stack

| Layer | Tool |
|-------|------|
| Computer vision | [Roboflow Inference](https://github.com/roboflow/inference) |
| Chess engine | [Stockfish](https://stockfishchess.org) via [python-chess](https://python-chess.readthedocs.io) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) |
| Frontend board | [chessboard.js](https://chessboardjs.com) + [chess.js](https://github.com/jhlywa/chess.js) |
| Video processing | [OpenCV](https://opencv.org) |

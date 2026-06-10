"""
main.py — Boardflow FastAPI server

Run:  uvicorn main:app --reload --port 8000
Then open:  http://localhost:8000
"""

import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))
from frame_extractor import extract_key_frames
from board_detector  import predictions_to_board, detect_orientation
from move_detector   import boards_to_move, init_chess_board
from analyzer        import analyze_game

import chess

# ── Roboflow config (baked in) ────────────────────────────────────────────────
ROBOFLOW_API_KEY  = os.environ.get("ROBOFLOW_API_KEY",  "sc2UeMDMoHAn22SEJbHv")
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "chess.com-pieces/2")

app = FastAPI(title="Boardflow", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """
    Accepts a chess screen-recording, runs Roboflow + Stockfish,
    returns a full game analysis JSON.
    """
    suffix = Path(video.filename).suffix if video.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        # 1. Extract key frames
        frames = extract_key_frames(tmp_path, sample_fps=1.5, max_frames=120)
        if not frames:
            raise HTTPException(status_code=422, detail="Could not extract frames from video.")

        # 2. Roboflow inference
        try:
            from inference_sdk import InferenceHTTPClient
            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=ROBOFLOW_API_KEY,
            )
        except ImportError:
            raise HTTPException(status_code=500, detail="Run: pip install inference-sdk")

        board_states = []
        for frame in frames:
            try:
                result = client.infer(frame.image_np, model_id=ROBOFLOW_MODEL_ID)
                preds = [
                    {
                        "class":      p["class"],
                        "x":          p["x"],
                        "y":          p["y"],
                        "width":      p["width"],
                        "height":     p["height"],
                        "confidence": p.get("confidence", 1.0),
                    }
                    for p in result.get("predictions", [])
                ]
                board = predictions_to_board(
                    preds,
                    white_at_bottom=True,
                    image_width=frame.image_np.shape[1],
                    image_height=frame.image_np.shape[0],
                )
                board_states.append(board)
            except Exception as e:
                print(f"[inference] frame error: {e}")
                board_states.append(None)

        # 3. Detect moves from board-state diffs
        chess_board = init_chess_board()
        moves_san = []
        prev_state = None

        for state in board_states:
            if state is None:
                continue
            if prev_state is None:
                prev_state = state
                continue
            move = boards_to_move(prev_state, state, chess_board)
            if move and move in chess_board.legal_moves:
                moves_san.append(chess_board.san(move))
                chess_board.push(move)
            prev_state = state

        if not moves_san:
            raise HTTPException(
                status_code=422,
                detail="No moves detected. Make sure the video clearly shows the chess board."
            )

        # 4. Stockfish analysis
        analysis = analyze_game(moves_san, depth=15)

        return JSONResponse(content=_serialize(analysis))

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _serialize(analysis) -> dict:
    return {
        "opening":             analysis.opening_name,
        "white_accuracy":      analysis.white_accuracy,
        "black_accuracy":      analysis.black_accuracy,
        "pgn":                 analysis.pgn,
        "stockfish_available": analysis.stockfish_available,
        "error":               analysis.error,
        "moves": [
            {
                "move_number":    m.move_number,
                "color":          m.color,
                "san":            m.san,
                "uci":            m.uci,
                "fen_before":     m.fen_before,
                "fen_after":      m.fen_after,
                "eval_before":    m.eval_before,
                "eval_after":     m.eval_after,
                "best_move_san":  m.best_move_san,
                "move_accuracy":  m.move_accuracy,
                "classification": m.classification,
                "is_check":       m.is_check,
                "is_capture":     m.is_capture,
            }
            for m in analysis.moves
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

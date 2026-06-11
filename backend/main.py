"""
main.py — Boardflow FastAPI server
"""
import os, sys, time, tempfile
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


def _infer_with_retry(client, image_np, model_id, max_retries=3):
    last_exc = None
    for attempt in range(max_retries):
        try:
            result = client.infer(image_np, model_id=model_id)
            return result.get("predictions", [])
        except Exception as e:
            last_exc = e
            err_str = str(e)
            if "524" in err_str or "520" in err_str or "timeout" in err_str.lower() or "connection" in err_str.lower():
                wait = 2 ** attempt
                print(f"[boardflow] inference attempt {attempt+1}/{max_retries} failed (retrying in {wait}s): {err_str[:120]}")
                time.sleep(wait)
            else:
                raise
    raise last_exc


@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    suffix = Path(video.filename).suffix if video.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name
    try:
        frames = extract_key_frames(
            tmp_path, sample_fps=2.0, max_frames=200, change_threshold=0.005
        )
        if not frames:
            raise HTTPException(status_code=422, detail="Could not extract frames from video.")
        from inference_sdk import InferenceHTTPClient
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY,
        )
        print(f"[boardflow] extracted {len(frames)} frames, model={ROBOFLOW_MODEL_ID}")

        inference_cache = {}
        failed_frames = []
        board_states = []
        for frame in frames:
            if frame.index in inference_cache:
                board_states.append(inference_cache[frame.index])
                continue
            try:
                raw_preds = _infer_with_retry(client, frame.image_np, ROBOFLOW_MODEL_ID)
                if frame.index == 0:
                    classes_seen = [p["class"] for p in raw_preds]
                    print(f"[boardflow] frame 0: {len(raw_preds)} preds, classes={classes_seen[:6]}")
                preds = [
                    {
                        "class": p["class"],
                        "x": p["x"], "y": p["y"],
                        "width": p["width"], "height": p["height"],
                        "confidence": p.get("confidence", 1.0),
                    }
                    for p in raw_preds
                ]
                h_px, w_px = frame.image_np.shape[:2]
                board = predictions_to_board(
                    preds,
                    white_at_bottom=True,
                    image_width=w_px,
                    image_height=h_px,
                )
                inference_cache[frame.index] = board
                board_states.append(board)
            except Exception as e:
                print(f"[boardflow] frame {frame.index} failed: {e}")
                failed_frames.append(frame.index)
                inference_cache[frame.index] = None
                board_states.append(None)

        if failed_frames:
            print(f"[boardflow] {len(failed_frames)} frames failed: {failed_frames}")

        chess_board = init_chess_board()
        moves_san = []
        prev_state = None
        last_move_frame = -999
        MIN_MOVE_GAP = 2  # at 2fps = 1s minimum per move

        for idx, state in enumerate(board_states):
            if state is None:
                continue
            if prev_state is None:
                prev_state = state
                continue

            if idx - last_move_frame < MIN_MOVE_GAP:
                prev_state = state
                continue

            move = boards_to_move(prev_state, state, chess_board)
            if move and move in chess_board.legal_moves:
                san = chess_board.san(move)
                if moves_san and san == moves_san[-1]:
                    prev_state = state
                    continue
                moves_san.append(san)
                chess_board.push(move)
                last_move_frame = idx
                print(f"[boardflow] frame {idx}: {san}")

            prev_state = state

        print(f"[boardflow] detected {len(moves_san)} moves: {moves_san[:8]}")

        if not moves_san:
            raise HTTPException(status_code=422, detail="No moves detected.")

        analysis = analyze_game(moves_san, depth=15)
        return JSONResponse(content=_serialize(analysis))
    finally:
        try:
            os.unlink(tmp_path)
        except:
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

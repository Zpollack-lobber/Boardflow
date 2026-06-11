"""
main.py — Boardflow FastAPI server (Qwen VL edition)
"""
import os, sys, re, time, tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
sys.path.insert(0, str(Path(__file__).parent))
from frame_extractor import extract_key_frames
from move_detector   import init_chess_board
from analyzer        import analyze_game
import chess

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "sc2UeMDMoHAn22SEJbHv")
QWEN_MODEL_ID    = "qwen-vl"

# Prompt asking Qwen VL to output only the FEN piece-placement string
FEN_PROMPT = (
    "This is a screenshot of a chess game. "
    "Output ONLY the FEN piece placement string for the current position "
    "(the part before the first space, e.g. rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR). "
    "Do not explain. Do not add anything else. Just the FEN string."
)

app = FastAPI(title="Boardflow", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


def _extract_fen(raw_response) -> str | None:
    """
    Pull the FEN placement string out of Qwen VL's response.
    Handles dict responses with various keys, or plain strings.
    """
    text = None
    if isinstance(raw_response, dict):
        # Try common response keys
        for k in ("response", "text", "result", "output", "content", "answer"):
            if k in raw_response:
                text = str(raw_response[k])
                break
        if text is None:
            text = str(raw_response)
    elif isinstance(raw_response, str):
        text = raw_response
    else:
        text = str(raw_response)

    if not text:
        return None

    # Look for a FEN-like pattern: rows separated by / with pieces and numbers
    m = re.search(r'([rnbqkpRNBQKP1-8]{1,8}(?:/[rnbqkpRNBQKP1-8]{1,8}){7})', text)
    if m:
        return m.group(1)
    return None


def _fen_to_board_dict(fen_placement: str) -> dict[str, str] | None:
    """Convert FEN placement string to {square: piece_symbol} dict via python-chess."""
    try:
        board = chess.Board(fen_placement + " w - - 0 1")
        result = {}
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                result[chess.square_name(sq)] = piece.symbol()
        return result
    except Exception:
        return None


def _infer_with_retry(client, image_np, prompt, max_retries=3):
    last_exc = None
    for attempt in range(max_retries):
        try:
            result = client.infer(image_np, model_id=QWEN_MODEL_ID, prompt=prompt)
            return result
        except Exception as e:
            last_exc = e
            err_str = str(e)
            if any(code in err_str for code in ["524", "520", "timeout", "connection"]):
                wait = 2 ** attempt
                print(f"[boardflow] attempt {attempt+1}/{max_retries} failed, retrying in {wait}s: {err_str[:120]}")
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
        frames = extract_key_frames(tmp_path, sample_fps=2.0, max_frames=200, change_threshold=0.005)
        if not frames:
            raise HTTPException(status_code=422, detail="Could not extract frames from video.")

        from inference_sdk import InferenceHTTPClient
        client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=ROBOFLOW_API_KEY)
        print(f"[boardflow] {len(frames)} frames → Qwen VL")

        # ── Get FEN for each frame ────────────────────────────────────────────
        fen_states = []
        for frame in frames:
            try:
                raw = _infer_with_retry(client, frame.image_np, FEN_PROMPT)
                if frame.index == 0:
                    print(f"[boardflow] frame 0 raw response: {repr(raw)[:300]}")
                fen = _extract_fen(raw)
                board_dict = _fen_to_board_dict(fen) if fen else None
                if frame.index == 0:
                    print(f"[boardflow] frame 0 FEN: {fen}  board_squares: {len(board_dict) if board_dict else 0}")
                fen_states.append(board_dict)
            except Exception as e:
                print(f"[boardflow] frame {frame.index} failed: {e}")
                fen_states.append(None)

        # ── Detect moves by diffing consecutive board states ──────────────────
        from move_detector import boards_to_move

        chess_board     = init_chess_board()
        moves_san       = []
        prev_state      = None
        last_move_frame = -999
        last_move_obj   = None
        MIN_MOVE_GAP    = 2

        for idx, state in enumerate(fen_states):
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
                if (last_move_obj is not None and
                        move.from_square == last_move_obj.to_square and
                        move.to_square   == last_move_obj.from_square):
                    prev_state = state
                    continue

                san = chess_board.san(move)
                if moves_san and san == moves_san[-1]:
                    prev_state = state
                    continue

                moves_san.append(san)
                chess_board.push(move)
                last_move_frame = idx
                last_move_obj   = move
                print(f"[boardflow] frame {idx}: {san}")

            prev_state = state

        print(f"[boardflow] detected {len(moves_san)} moves: {moves_san}")

        if not moves_san:
            raise HTTPException(status_code=422, detail="No moves detected.")

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

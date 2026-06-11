"""
main.py — Boardflow FastAPI server (WebRTC Workflow Edition)
"""
import os, sys, re, json, time, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
sys.path.insert(0, str(Path(__file__).parent))
from move_detector import init_chess_board
from analyzer import analyze_game
import chess

ROBOFLOW_API_KEY   = os.environ.get("ROBOFLOW_API_KEY",   "sc2UeMDMoHAn22SEJbHv")
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "zachs-workspace-cnn1l")
ROBOFLOW_WORKFLOW  = os.environ.get("ROBOFLOW_WORKFLOW",  "soccer-ball-video-detector-1781110679341")

app = FastAPI(title="Boardflow", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

_executor = ThreadPoolExecutor(max_workers=2)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


def _parse_uci_from_message(msg) -> str | None:
    """
    Try to extract a UCI move string (e.g. 'e2e4') from vision_events_message.
    Handles dicts, JSON strings, and plain text with UCI/coordinate patterns.
    """
    if not msg:
        return None

    # If it's already a dict
    if isinstance(msg, dict):
        for k in ("move", "uci", "from_to", "chess_move"):
            if k in msg:
                raw = str(msg[k]).replace("-", "").replace(" ", "").lower()
                if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', raw):
                    return raw
        from_sq = msg.get("from") or msg.get("from_square") or msg.get("source")
        to_sq   = msg.get("to")   or msg.get("to_square")   or msg.get("target")
        if from_sq and to_sq:
            return str(from_sq).lower().strip() + str(to_sq).lower().strip()

    # If it's a list, check first element
    if isinstance(msg, list):
        for item in msg:
            result = _parse_uci_from_message(item)
            if result:
                return result
        return None

    s = str(msg).strip()

    # Try JSON parse
    try:
        obj = json.loads(s)
        return _parse_uci_from_message(obj)
    except Exception:
        pass

    # Regex: e2e4 or e2-e4 or e2 to e4
    m = re.search(r'\b([a-h][1-8])[\s\-_→to]*([a-h][1-8])\b', s, re.IGNORECASE)
    if m:
        return m.group(1).lower() + m.group(2).lower()

    return None


def _run_webrtc_session(tmp_path: str) -> list[dict]:
    """
    Stream a video file through the Roboflow WebRTC workflow.
    Returns a list of per-frame dicts; frames without events are included for
    debugging but moves are only extracted from frames where message is set.
    """
    from inference_sdk import InferenceHTTPClient
    from inference_sdk.webrtc import VideoFileSource, StreamConfig, VideoMetadata

    client = InferenceHTTPClient.init(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )

    source = VideoFileSource(tmp_path, realtime_processing=False)
    config = StreamConfig(
        stream_output=[],
        data_output=["predictions", "vision_events_error_status", "vision_events_message",
                     "baseball_count"],
        requested_plan="webrtc-gpu-medium",
        requested_region="us",
    )
    session = client.webrtc.stream(
        source=source,
        workflow=ROBOFLOW_WORKFLOW,
        workspace=ROBOFLOW_WORKSPACE,
        image_input="image",
        config=config
    )

    frame_data = []
    frame_count = [0]

    @session.on_data()
    def on_data(data: dict, metadata: VideoMetadata):
        fid = metadata.frame_id
        msg  = data.get("vision_events_message")
        err  = data.get("vision_events_error_status")
        preds = data.get("predictions")

        frame_count[0] += 1

        # Log first 5 frames always, plus every frame that has a message
        if frame_count[0] <= 5 or msg:
            safe_data = {k: v for k, v in data.items() if k not in ("output_image",)}
            print(f"[boardflow] frame {fid} ALL KEYS: {list(safe_data.keys())}")
            print(f"[boardflow] frame {fid} FULL: {json.dumps(safe_data, default=str)[:1200]}")

        frame_data.append({
            "frame_id":    fid,
            "message":     msg,
            "error":       err,
            "predictions": preds,
        })

    session.run()
    event_frames = sum(1 for f in frame_data if f["message"])
    print(f"[boardflow] WebRTC done: {frame_count[0]} frames, {event_frames} with events")
    return frame_data


@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    suffix = Path(video.filename).suffix if video.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name
    try:
        import asyncio
        loop = asyncio.get_event_loop()

        # ── Stream video through Roboflow WebRTC workflow ─────────────────────
        frame_data = await loop.run_in_executor(_executor, _run_webrtc_session, tmp_path)

        if not frame_data:
            raise HTTPException(status_code=422, detail="No frames processed from video.")

        # ── Extract and validate moves from vision_events_message ─────────────
        chess_board = init_chess_board()
        moves_san   = []

        for fd in frame_data:
            msg = fd["message"]
            if not msg:
                continue

            uci = _parse_uci_from_message(msg)
            if not uci:
                print(f"[boardflow] unparseable message: {repr(msg)}")
                continue

            try:
                move = chess.Move.from_uci(uci)
            except Exception as e:
                print(f"[boardflow] bad UCI '{uci}': {e}")
                continue

            if move not in chess_board.legal_moves:
                # Try queen promotion (pawn reaching last rank)
                promo = chess.Move.from_uci(uci + "q") if len(uci) == 4 else None
                if promo and promo in chess_board.legal_moves:
                    move = promo
                else:
                    print(f"[boardflow] '{uci}' illegal at move {len(moves_san)+1} "
                          f"(fen: {chess_board.fen()[:40]})")
                    continue

            san = chess_board.san(move)
            if moves_san and san == moves_san[-1]:
                continue   # deduplicate consecutive identical events

            moves_san.append(san)
            chess_board.push(move)
            print(f"[boardflow] move {len(moves_san)}: {san}  (msg={repr(msg)})")

        print(f"[boardflow] total detected: {len(moves_san)} moves → {moves_san}")

        if not moves_san:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No moves detected. Processed {len(frame_data)} frames; "
                    f"{sum(1 for f in frame_data if f['message'])} had vision events. "
                    "Check Railway logs for raw message format."
                ),
            )

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

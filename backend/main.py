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
        frames = extract_key_frames(tmp_path, sample_fps=2.0, max_frames=200, change_threshold=0.005)
        if not frames:
            raise HTTPException(status_code=422, detail="Could not extract frames from video.")
        from inference_sdk import InferenceHTTPClient
        client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=ROBOFLOW_API_KEY)
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
                    print(f"[boardflow] frame 0 predictions: {len(raw_preds)} pieces, classes={classes_seen[:6]}")
                preds = [{"class": p["class"], "x": p["x"], "y": p["y"], "width": p["width"], "height": p["height"], "confidence": p.get("confidence", 1.0)} for p in raw_preds]
                board = predictions_to_board(preds, white_at_bottom=True, image_width=frame.image_np.shape[1], image

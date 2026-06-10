"""
frame_extractor.py — pull key frames from a chess screen-recording

Strategy:
  1. Sample the video at ~2 fps.
  2. Detect the chess board region using colour-based square detection.
  3. Crop every sampled frame to just the board region.
  4. Keep only frames where the board has changed significantly
     (skips duplicate frames, saves Roboflow API calls).
"""

import cv2
import numpy as np
import base64
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractedFrame:
    index: int          # sequential frame number among kept frames
    timestamp: float    # seconds into the video
    image_b64: str      # base64-encoded JPEG of the board crop
    image_np: np.ndarray  # numpy array for further processing


def detect_board_region(frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    Locate the chess board in a Chess.com screen recording.

    Strategy: mask the frame for Chess.com's characteristic square colours
    (multiple board themes supported), then find the largest square-ish blob.
    Falls back to contour-based detection if colour matching yields nothing.
    Returns (x, y, w, h) or None.
    """
    h, w = frame.shape[:2]

    # ── Chess.com board colour palettes (BGR order) ─────────────────────────
    # Each tuple: (light_lo, light_hi, dark_lo, dark_hi)
    # Brown classic  (light ≈ #f0d9b5, dark ≈ #b58863)
    # Green          (light ≈ #eeeed2, dark ≈ #769656)
    # Blue           (light ≈ #dee3e6, dark ≈ #8ca2ad)
    COLOR_SCHEMES = [
        (np.array([165, 195, 220]), np.array([210, 230, 255]),   # brown light
         np.array([ 75, 110, 155]), np.array([130, 165, 210])),  # brown dark
        (np.array([185, 215, 215]), np.array([230, 250, 250]),   # green light
         np.array([ 65, 120,  90]), np.array([120, 175, 140])),  # green dark
        (np.array([190, 195, 195]), np.array([240, 240, 240]),   # blue light
         np.array([100, 130, 100]), np.array([155, 180, 155])),  # blue dark
        # Broad fallback
        (np.array([150, 150, 150]), np.array([255, 255, 255]),
         np.array([ 50,  50,  50]), np.array([160, 160, 200])),
    ]

    best_box   = None
    best_score = 0
    morph_k    = np.ones((12, 12), np.uint8)

    for ll, lh, dl, dh in COLOR_SCHEMES:
        mask = cv2.bitwise_or(
            cv2.inRange(frame, ll, lh),
            cv2.inRange(frame, dl, dh),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  morph_k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (min(h, w) * 0.15) ** 2:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh else 0
            if not (0.75 < aspect < 1.33):
                continue
            score = area * (1 - abs(aspect - 1))
            if score > best_score:
                best_score = score
                best_box   = (bx, by, bw, bh)

    if best_box:
        bx, by, bw, bh = best_box
        # Force a square crop (take the smaller dimension, centred)
        side = min(bw, bh)
        bx   = bx + (bw - side) // 2
        by   = by + (bh - side) // 2
        return (bx, by, side, side)

    # ── Fallback: largest near-square contour from adaptive threshold ────────
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    fb_best  = None
    fb_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (min(h, w) * 0.2) ** 2:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh else 0
        if not (0.7 < aspect < 1.4):
            continue
        score = area * (1 - abs(aspect - 1))
        if score > fb_score:
            fb_score = score
            fb_best  = (bx, by, bw, bh)

    return fb_best


def frames_are_different(a: np.ndarray, b: np.ndarray, threshold: float = 0.02) -> bool:
    """Return True if frames differ enough to represent a new board state."""
    if a.shape != b.shape:
        a = cv2.resize(a, (b.shape[1], b.shape[0]))
    diff = cv2.absdiff(a, b)
    changed_fraction = np.count_nonzero(diff > 30) / diff.size
    return changed_fraction > threshold


def extract_key_frames(
    video_path: str,
    sample_fps: float = 1.5,
    change_threshold: float = 0.02,
    max_frames: int = 120,
) -> list[ExtractedFrame]:
    """
    Extract frames where the board state has changed.

    Automatically adjusts the sampling rate for long videos so that a
    3-minute game is handled the same as a 30-second clip — no manual
    speed-up required from the user.

    Returns a list of ExtractedFrame objects.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps

    # Auto-scale: target at most max_frames samples across the whole video.
    # For a 3-minute game this means ~1 sample every ~1.5 s by default,
    # but for a 10-minute game it automatically stretches to 1 per 5 s, etc.
    target_samples   = max_frames
    auto_sample_fps  = min(sample_fps, target_samples / max(duration_sec, 1))
    step = max(1, int(video_fps / auto_sample_fps))

    board_region: Optional[tuple] = None
    prev_board: Optional[np.ndarray] = None
    result: list[ExtractedFrame] = []
    frame_idx = 0
    kept = 0

    while cap.isOpened() and kept < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / video_fps

        # Auto-detect board region on first frame (and cache it)
        if board_region is None:
            board_region = detect_board_region(frame)

        # Crop to board
        if board_region:
            x, y, bw, bh = board_region
            board_crop = frame[y:y+bh, x:x+bw]
        else:
            board_crop = frame

        # Resize to a standard size for consistent Roboflow calls
        board_std = cv2.resize(board_crop, (640, 640))

        # Only keep if the board changed since the last kept frame
        if prev_board is None or frames_are_different(prev_board, board_std, change_threshold):
            # Encode to JPEG base64
            _, buf = cv2.imencode(".jpg", board_std, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            result.append(ExtractedFrame(
                index=kept,
                timestamp=timestamp,
                image_b64=b64,
                image_np=board_std.copy(),
            ))
            prev_board = board_std
            kept += 1

        frame_idx += 1

    cap.release()
    return result

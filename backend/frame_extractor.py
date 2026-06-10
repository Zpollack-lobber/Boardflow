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
    Attempt to locate the chess board in a frame.
    Returns (x, y, w, h) or None.

    Approach: detect the largest near-square region that consists of
    alternating light/dark squares — characteristic of a chess board.
    Falls back to returning the full frame if detection fails.
    """
    h, w = frame.shape[:2]

    # Convert to HSV and look for chessboard-like colour clusters
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to find the alternating-square pattern
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (min(h, w) * 0.2) ** 2:
            continue  # too small

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch else 0
        if not (0.7 < aspect < 1.4):
            continue  # not square-ish

        # Prefer larger, more square regions
        squareness = 1 - abs(aspect - 1)
        score = area * squareness
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    return best


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

    Returns a list of ExtractedFrame objects.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(video_fps / sample_fps))

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

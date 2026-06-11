"""
board_detector.py — Roboflow inference + board-state parsing

The model returns bounding boxes like:
  { "x": 320, "y": 240, "width": 40, "height": 40,
    "class": "white-queen", "confidence": 0.92 }

We estimate board boundaries from the piece positions themselves, then
map each piece centre to one of the 64 squares (a1-h8).
"""

import math
import numpy as np
from typing import Optional

# ── Roboflow class labels → python-chess piece symbols ──────────────────────
PIECE_SYMBOL: dict[str, str] = {
    "b-king": "k",  "b-queen": "q",  "b-rook": "r",
    "b-bishop": "b",  "b-knight": "n",  "b-pawn": "p",
    "w-king": "K",  "w-queen": "Q",  "w-rook": "R",
    "w-bishop": "B",  "w-knight": "N",  "w-pawn": "P",
    "white-king":   "K",  "white-queen":  "Q",  "white-rook":   "R",
    "white-bishop": "B",  "white-knight": "N",  "white-pawn":   "P",
    "black-king":   "k",  "black-queen":  "q",  "black-rook":   "r",
    "black-bishop": "b",  "black-knight": "n",  "black-pawn":   "p",
    "bp": "p",  "bр": "p",  "wp": "P",
    "bk": "k",  "wk": "K",
    "bq": "q",  "wq": "Q",
    "br": "r",  "wr": "R",
    "bb": "b",  "wb": "B",
    "bn": "n",  "wn": "N",
    "BP": "p",  "WP": "P",
    "BK": "k",  "WK": "K",
    "BQ": "q",  "WQ": "Q",
    "BR": "r",  "WR": "R",
    "BB": "b",  "WB": "B",
    "BN": "n",  "WN": "N",
    "king":   "K",  "queen":  "Q",  "rook":   "R",
    "bishop": "B",  "knight": "N",  "pawn":   "P",
    "White King":   "K",  "White Queen":  "Q",  "White Rook":   "R",
    "White Bishop": "B",  "White Knight": "N",  "White Pawn":   "P",
    "Black King":   "k",  "Black Queen":  "q",  "Black Rook":   "r",
    "Black Bishop": "b",  "Black Knight": "n",  "Black Pawn":   "p",
    "King": "K", "Queen": "Q", "Rook": "R", "Bishop": "B", "Knight": "N", "Pawn": "P",
}


def _normalize(label: str) -> str:
    return PIECE_SYMBOL.get(label) or PIECE_SYMBOL.get(label.replace("-", " ").title())


def _get_board_bbox(predictions: list[dict]) -> Optional[tuple[float, float, float, float]]:
    for p in predictions:
        if p.get("class", "").lower() == "board":
            cx, cy = p["x"], p["y"]
            hw, hh = p["width"] / 2, p["height"] / 2
            return (cx - hw, cy - hh, cx + hw, cy + hh)
    return None


def predictions_to_board(predictions: list[dict],
                          white_at_bottom: bool = True,
                          image_width: int = 0,
                          image_height: int = 0,
                          calibration: Optional[dict] = None) -> Optional[dict[str, str]]:
    pieces = []
    for p in predictions:
        if not isinstance(p, dict):
            continue
        sym = _normalize(p.get("class", ""))
        if not sym:
            continue
        pieces.append({
            "sym":  sym,
            "cx":   p["x"],
            "cy":   p["y"],
            "conf": p.get("confidence", 1.0),
        })

    if len(pieces) < 4:
        return None

    xs_arr = np.array([p["cx"] for p in pieces])
    ys_arr = np.array([p["cy"] for p in pieces])

    def _iqr_bounds(arr: np.ndarray, k: float = 1.5):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr

    x_lo, x_hi = _iqr_bounds(xs_arr)
    y_lo, y_hi = _iqr_bounds(ys_arr)
    pieces = [p for p in pieces
              if x_lo <= p["cx"] <= x_hi and y_lo <= p["cy"] <= y_hi]

    if len(pieces) > 32:
        pieces.sort(key=lambda p: p["conf"], reverse=True)
        pieces = pieces[:32]

    if len(pieces) < 4:
        return None

    if calibration:
        board_x0 = calibration["x0"]
        board_y0 = calibration["y0"]
        board_w  = calibration["w"]
        board_h  = calibration["h"]
    else:
        xs = [p["cx"] for p in pieces]
        ys = [p["cy"] for p in pieces]

        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)
        cell_w = (x_spread / 7) if x_spread > 0 else 40
        cell_h = (y_spread / 7) if y_spread > 0 else 40

        board_x0 = min(xs) - cell_w * 0.5
        board_y0 = min(ys) - cell_h * 0.5
        board_x1 = max(xs) + cell_w * 0.5
        board_y1 = max(ys) + cell_h * 0.5

        board_w = board_x1 - board_x0
        board_h = board_y1 - board_y0

        if board_w <= 0 or board_h <= 0:
            return None

    board: dict[str, str] = {}
    for p in pieces:
        col = round((p["cx"] - board_x0) / board_w * 7)
        row = round((p["cy"] - board_y0) / board_h * 7)

        col = max(0, min(7, col))
        row = max(0, min(7, row))

        if white_at_bottom:
            file = chr(ord("a") + col)
            rank = str(8 - row)
        else:
            file = chr(ord("h") - col)
            rank = str(row + 1)

        square = file + rank
        board[square] = p["sym"]

    return board


def board_to_fen_placement(board: dict[str, str]) -> str:
    rows = []
    for rank in range(8, 0, -1):
        empty = 0
        row_str = ""
        for file_idx in range(8):
            sq = chr(ord("a") + file_idx) + str(rank)
            piece = board.get(sq)
            if piece:
                if empty:
                    row_str += str(empty)
                    empty = 0
                row_str += piece
            else:
                empty += 1
        if empty:
            row_str += str(empty)
        rows.append(row_str)
    return "/".join(rows)


def extract_calibration(predictions: list[dict]) -> Optional[dict]:
    pieces = []
    for p in predictions:
        if not isinstance(p, dict):
            continue
        sym = _normalize(p.get("class", ""))
        if not sym:
            continue
        pieces.append({"cx": p["x"], "cy": p["y"]})

    if len(pieces) < 4:
        return None

    xs_arr = np.array([p["cx"] for p in pieces])
    ys_arr = np.array([p["cy"] for p in pieces])

    def _iqr_bounds(arr: np.ndarray, k: float = 1.5):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr

    x_lo, x_hi = _iqr_bounds(xs_arr)
    y_lo, y_hi = _iqr_bounds(ys_arr)
    pieces = [p for p in pieces
              if x_lo <= p["cx"] <= x_hi and y_lo <= p["cy"] <= y_hi]

    if len(pieces) < 4:
        return None

    xs = [p["cx"] for p in pieces]
    ys = [p["cy"] for p in pieces]

    x_spread = max(xs) - min(xs)
    y_spread = max(ys) - min(ys)
    cell_w = (x_spread / 7) if x_spread > 0 else 40
    cell_h = (y_spread / 7) if y_spread > 0 else 40

    x0 = min(xs) - cell_w * 0.5
    y0 = min(ys) - cell_h * 0.5
    x1 = max(xs) + cell_w * 0.5
    y1 = max(ys) + cell_h * 0.5
    w = x1 - x0
    h = y1 - y0

    if w <= 0 or h <= 0:
        return None

    return {"x0": x0, "y0": y0, "w": w, "h": h}


def detect_orientation(board: dict[str, str]) -> bool:
    wk = None
    bk = None
    for sq, sym in board.items():
        if sym == "K":
            wk = sq
        elif sym == "k":
            bk = sq

    if wk and bk:
        wk_rank = int(wk[1])
        bk_rank = int(bk[1])
        return wk_rank < bk_rank
    return True

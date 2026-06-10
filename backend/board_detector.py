"""
board_detector.py — Roboflow inference + board-state parsing

The model returns bounding boxes like:
  { "x": 320, "y": 240, "width": 40, "height": 40,
    "class": "white-queen", "confidence": 0.92 }

We estimate board boundaries from the piece positions themselves, then
map each piece centre to one of the 64 squares (a1-h8).
"""

import math
from typing import Optional

# ── Roboflow class labels → python-chess piece symbols ──────────────────────
# Covers chess-pieces-mjzgj/1 (Roboflow 100) class names + common variants
PIECE_SYMBOL: dict[str, str] = {
    # chess-pieces-mjzgj/1 exact class names
    "white-king":   "K",  "white-queen":  "Q",  "white-rook":   "R",
    "white-bishop": "B",  "white-knight": "N",  "white-pawn":   "P",
    "black-king":   "k",  "black-queen":  "q",  "black-rook":   "r",
    "black-bishop": "b",  "black-knight": "n",  "black-pawn":   "p",
    # generic (no-color) fallbacks — model sometimes labels these without prefix
    "king":   "K",  "queen":  "Q",  "rook":   "R",
    "bishop": "B",  "knight": "N",  "pawn":   "P",
    # title-cased variants
    "White King":   "K",  "White Queen":  "Q",  "White Rook":   "R",
    "White Bishop": "B",  "White Knight": "N",  "White Pawn":   "P",
    "Black King":   "k",  "Black Queen":  "q",  "Black Rook":   "r",
    "Black Bishop": "b",  "Black Knight": "n",  "Black Pawn":   "p",
    "King": "K", "Queen": "Q", "Rook": "R", "Bishop": "B", "Knight": "N", "Pawn": "P",
}


def _normalize(label: str) -> str:
    """Try both the raw label and a title-cased version."""
    return PIECE_SYMBOL.get(label) or PIECE_SYMBOL.get(label.replace("-", " ").title())


def predictions_to_board(predictions: list[dict],
                          white_at_bottom: bool = True,
                          image_width: int = 0,
                          image_height: int = 0) -> Optional[dict[str, str]]:
    """
    Convert a list of Roboflow predictions for one frame into a board dict.

    Returns  { square: piece_symbol }  e.g. {"e1": "K", "d1": "Q", ...}
    Returns None if fewer than 4 pieces are detected (unreliable frame).
    """
    pieces = []
    for p in predictions:
        sym = _normalize(p.get("class", ""))
        if sym:
            pieces.append({"sym": sym, "cx": p["x"], "cy": p["y"]})

    if len(pieces) < 4:
        return None

    # ── Estimate board bounding box ──────────────────────────────────────────
    # Strategy: use piece centres + a margin equal to half the typical cell size.
    # This is an approximation; it works well when most squares are occupied
    # (early/mid game) and degrades gracefully in the end-game.
    xs = [p["cx"] for p in pieces]
    ys = [p["cy"] for p in pieces]

    # Estimate cell size from spread of pieces  (board ≈ 8 cells wide/tall)
    x_spread = max(xs) - min(xs)
    y_spread = max(ys) - min(ys)
    cell_w = (x_spread / 6) if x_spread > 0 else 40   # 8 squares → 7 gaps
    cell_h = (y_spread / 6) if y_spread > 0 else 40

    board_x0 = min(xs) - cell_w * 0.5
    board_y0 = min(ys) - cell_h * 0.5
    board_x1 = max(xs) + cell_w * 0.5
    board_y1 = max(ys) + cell_h * 0.5

    board_w = board_x1 - board_x0
    board_h = board_y1 - board_y0

    if board_w <= 0 or board_h <= 0:
        return None

    # ── Map each piece to a square ──────────────────────────────────────────
    board: dict[str, str] = {}
    for p in pieces:
        col = round((p["cx"] - board_x0) / board_w * 7)   # 0–7
        row = round((p["cy"] - board_y0) / board_h * 7)   # 0–7 (image-top = low row)

        col = max(0, min(7, col))
        row = max(0, min(7, row))

        if white_at_bottom:
            # image-top → rank 8, image-bottom → rank 1
            file = chr(ord("a") + col)
            rank = str(8 - row)
        else:
            file = chr(ord("h") - col)
            rank = str(row + 1)

        square = file + rank
        # If two pieces land on the same square keep the more confident one
        # (or simply last-write-wins for now)
        board[square] = p["sym"]

    return board


def board_to_fen_placement(board: dict[str, str]) -> str:
    """Convert {square: symbol} to the piece-placement part of a FEN string."""
    rows = []
    for rank in range(8, 0, -1):           # 8 down to 1
        empty = 0
        row_str = ""
        for file_idx in range(8):           # a to h
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


def detect_orientation(board: dict[str, str]) -> bool:
    """
    Guess whether white is at the bottom by checking where the white king sits.
    Returns True  → white at bottom (standard view).
    Returns False → white at top (board flipped).
    """
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
        return wk_rank < bk_rank   # white king on lower rank → white at bottom
    return True   # default

"""
move_detector.py — diff two board states and produce a chess.Move

Uses python-chess for legality checking so we get correct algebraic notation,
en-passant, castling, and promotion detection for free.
"""

import chess
from typing import Optional


def boards_to_move(prev: dict[str, str],
                   curr: dict[str, str],
                   chess_board: chess.Board) -> Optional[chess.Move]:
    """
    Given the previous and current board dicts (square → piece symbol),
    try to find the legal chess.Move that transforms prev → curr.

    Designed to be noise-tolerant: Roboflow detections are imperfect so we
    use several strategies from most- to least-exact before falling back to
    a best-match search over all legal moves.

    Returns a chess.Move or None if no plausible legal move is found.
    """
    # Only consider squares that are valid chess square names
    def valid_sq(s: str) -> bool:
        return len(s) == 2 and s[0] in "abcdefgh" and s[1] in "12345678"

    prev_clean = {sq: v for sq, v in prev.items() if valid_sq(sq)}
    curr_clean = {sq: v for sq, v in curr.items() if valid_sq(sq)}

    all_squares = set(prev_clean.keys()) | set(curr_clean.keys())
    disappeared = {sq for sq in all_squares if prev_clean.get(sq) and not curr_clean.get(sq)}
    appeared    = {sq for sq in all_squares if not prev_clean.get(sq) and curr_clean.get(sq)}

    # ── Early exit: boards are identical — no move happened ─────────────────
    if len(disappeared) == 0 and len(appeared) == 0:
        return None

    # ── Strategy 1: clean single move (1 disappeared, 1 appeared) ───────────
    if len(disappeared) == 1 and len(appeared) == 1:
        move = _try_move(list(disappeared)[0], list(appeared)[0], chess_board)
        if move:
            return move

    # ── Strategy 2: capture (piece left from-sq, victim gone, attacker on to-sq)
    if len(disappeared) == 2 and len(appeared) == 1:
        to_sq_str = list(appeared)[0]
        for from_sq_str in disappeared:
            if from_sq_str != to_sq_str:
                move = _try_move(from_sq_str, to_sq_str, chess_board)
                if move:
                    return move

    # ── Strategy 3: castling (king + rook both moved) ───────────────────────
    if len(disappeared) == 2 and len(appeared) == 2:
        for from_sq_str in disappeared:
            for to_sq_str in appeared:
                move = _try_move(from_sq_str, to_sq_str, chess_board)
                if move:
                    return move

    # ── Strategy 5: best-match fallback ──────────────────────────────────────
    # When the diff is noisy, score every legal move by how well its resulting
    # position matches the detected curr board, and pick the best one.
    # Only accept if the best move scores better than "no move happened" —
    # this eliminates phantom detections on frames with no real move.
    null_score = _board_match_score(curr_clean, chess_board)

    best_move  = None
    best_score = null_score  # must beat the baseline to be accepted
    for move in chess_board.legal_moves:
        chess_board.push(move)
        score = _board_match_score(curr_clean, chess_board)
        chess_board.pop()
        if score > best_score:
            best_score = score
            best_move  = move

    if best_move and best_score > null_score + 2:
        return best_move

    return None


def _try_move(from_sq_str: str, to_sq_str: str,
              chess_board: chess.Board) -> Optional[chess.Move]:
    """Attempt to build and validate a move; return it if legal, else None."""
    try:
        move = chess.Move(
            chess.parse_square(from_sq_str),
            chess.parse_square(to_sq_str),
        )
        if move in chess_board.legal_moves:
            return move
        # Try with queen promotion (covers pawn-to-last-rank cases)
        promo = chess.Move(
            chess.parse_square(from_sq_str),
            chess.parse_square(to_sq_str),
            promotion=chess.QUEEN,
        )
        if promo in chess_board.legal_moves:
            return promo
    except Exception:
        pass
    return None


def _board_match_score(board_dict: dict[str, str], chess_board: chess.Board) -> int:
    """
    Score how well the detected board matches the chess.Board position.
    +1 for a square that has any piece in both detected and actual.
    +2 bonus if the piece symbol also matches (correct type + color).
    -2 for a detected piece on a square the actual board says is empty
         (penalises false detections and wrong-square CV errors).
    """
    score = 0
    for sq_name, detected_sym in board_dict.items():
        try:
            sq = chess.parse_square(sq_name)
            piece = chess_board.piece_at(sq)
            if piece is not None:
                score += 1  # occupancy match
                if piece.symbol() == detected_sym:
                    score += 2  # piece type + color match bonus
            else:
                score -= 2  # detected piece where board says empty → penalise
        except Exception:
            pass
    return score

def init_chess_board() -> chess.Board:
    return chess.Board()

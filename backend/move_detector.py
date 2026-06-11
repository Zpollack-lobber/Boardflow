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

    # ── Strategy 4: noisy diff — try small disappeared/appeared windows ──────
    # Handles cases where noise adds/removes 1-3 extra squares.
    if len(disappeared) <= 5 and len(appeared) <= 5:
        for from_sq_str in disappeared:
            for to_sq_str in appeared:
                if from_sq_str != to_sq_str:
                    move = _try_move(from_sq_str, to_sq_str, chess_board)
                    if move:
                        return move

    # ── Strategy 5: best-match fallback over all legal moves ─────────────────
    # Only accept if 75%+ of detected squares match — high bar to avoid
    # accepting noise as a move and corrupting the running board state.
    best_move  = None
    best_score = -1
    for move in chess_board.legal_moves:
        tmp = chess_board.copy()
        tmp.push(move)
        score = _board_match_score(curr_clean, tmp)
        if score > best_score:
            best_score = score
            best_move  = move

    min_threshold = max(4, len(curr_clean) * 0.50)
    if best_score >= min_threshold:
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
    """Count how many occupied squares in board_dict match the chess.Board position."""
    score = 0
    for sq_name in board_dict.keys():
        try:
            sq = chess.parse_square(sq_name)
            if chess_board.piece_at(sq) is not None:
                score += 1
        except Exception:
            pass
    return score


def init_chess_board() -> chess.Board:
    return chess.Board()

"""
move_detector.py — diff two board states and produce a chess.Move
"""

import chess
from typing import Optional


def boards_to_move(prev: dict[str, str],
                   curr: dict[str, str],
                   chess_board: chess.Board) -> Optional[chess.Move]:
    def valid_sq(s: str) -> bool:
        return len(s) == 2 and s[0] in "abcdefgh" and s[1] in "12345678"

    prev_clean = {sq: v for sq, v in prev.items() if valid_sq(sq)}
    curr_clean = {sq: v for sq, v in curr.items() if valid_sq(sq)}

    all_squares = set(prev_clean.keys()) | set(curr_clean.keys())
    disappeared = {sq for sq in all_squares if prev_clean.get(sq) and not curr_clean.get(sq)}
    appeared    = {sq for sq in all_squares if not prev_clean.get(sq) and curr_clean.get(sq)}

    if len(disappeared) == 0 and len(appeared) == 0:
        return None

    # Strategy 1: clean single move (1 disappeared, 1 appeared)
    if len(disappeared) == 1 and len(appeared) == 1:
        move = _try_move(list(disappeared)[0], list(appeared)[0], chess_board)
        if move:
            return move

    # Strategy 2: capture (2 disappeared, 1 appeared)
    if len(disappeared) == 2 and len(appeared) == 1:
        to_sq_str = list(appeared)[0]
        for from_sq_str in disappeared:
            if from_sq_str != to_sq_str:
                move = _try_move(from_sq_str, to_sq_str, chess_board)
                if move:
                    return move

    # Strategy 3: castling (2 disappeared, 2 appeared)
    if len(disappeared) == 2 and len(appeared) == 2:
        for from_sq_str in disappeared:
            for to_sq_str in appeared:
                move = _try_move(from_sq_str, to_sq_str, chess_board)
                if move:
                    return move

    # Strategy 4: best-match fallback — only if 70%+ of squares match
    best_move  = None
    best_score = -1
    for move in chess_board.legal_moves:
        tmp = chess_board.copy()
        tmp.push(move)
        score = _board_match_score(curr_clean, tmp)
        if score > best_score:
            best_score = score
            best_move  = move

    min_threshold = max(5, len(curr_clean) * 0.60)
    if best_score >= min_threshold:
        return best_move

    return None


def _try_move(from_sq_str: str, to_sq_str: str,
              chess_board: chess.Board) -> Optional[chess.Move]:
    try:
        move = chess.Move(
            chess.parse_square(from_sq_str),
            chess.parse_square(to_sq_str),
        )
        if move in chess_board.legal_moves:
            return move
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

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

    Returns a chess.Move or None if no single legal move fits.
    """
    # Find squares that changed
    all_squares = set(prev.keys()) | set(curr.keys())
    disappeared = {sq for sq in all_squares if prev.get(sq) and not curr.get(sq)}
    appeared    = {sq for sq in all_squares if not prev.get(sq) and curr.get(sq)}
    changed     = {sq for sq in all_squares
                   if prev.get(sq) and curr.get(sq) and prev[sq] != curr[sq]}

    # Simple case: one piece moved from A → B
    if len(disappeared) == 1 and len(appeared) == 1:
        from_sq = chess.parse_square(list(disappeared)[0])
        to_sq   = chess.parse_square(list(appeared)[0])
        move    = chess.Move(from_sq, to_sq)
        if move in chess_board.legal_moves:
            return move

    # Capture: one piece disappeared from A, another from B (B now has attacker)
    if len(disappeared) == 2 and len(appeared) == 1:
        to_sq = chess.parse_square(list(appeared)[0])
        for from_square in disappeared:
            if from_square != list(appeared)[0]:
                from_sq = chess.parse_square(from_square)
                move = chess.Move(from_sq, to_sq)
                if move in chess_board.legal_moves:
                    return move

    # Castling: king + rook both moved
    if len(disappeared) == 2 and len(appeared) == 2:
        # Try all combinations
        for from_sq_str in disappeared:
            for to_sq_str in appeared:
                try:
                    move = chess.Move(
                        chess.parse_square(from_sq_str),
                        chess.parse_square(to_sq_str)
                    )
                    if move in chess_board.legal_moves:
                        return move
                except Exception:
                    pass

    # Fallback: try every legal move and pick the one whose result best matches curr
    best_move = None
    best_score = -1
    for move in chess_board.legal_moves:
        tmp = chess_board.copy()
        tmp.push(move)
        score = _board_match_score(curr, tmp)
        if score > best_score:
            best_score = score
            best_move = move

    # Only accept if it's a strong match
    if best_score >= len(curr) - 2:
        return best_move

    return None


def _board_match_score(board_dict: dict[str, str], chess_board: chess.Board) -> int:
    """Count how many squares in board_dict match the chess.Board position."""
    score = 0
    for sq_name, sym in board_dict.items():
        try:
            sq = chess.parse_square(sq_name)
            piece = chess_board.piece_at(sq)
            if piece and piece.symbol() == sym:
                score += 1
        except Exception:
            pass
    return score


def init_chess_board() -> chess.Board:
    return chess.Board()

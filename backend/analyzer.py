"""
analyzer.py — Stockfish evaluation + Lichess accuracy metric

Lichess accuracy formula (open source, from lila/modules/analyse):
  winPercent(cp) = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)
  moveAccuracy  = 103.1668 * exp(-0.04354 * max(0, wp_before - wp_after)) - 3.167
  gameAccuracy  = average of move accuracies

Stockfish is called via python-chess engine API.
Install Stockfish:  brew install stockfish  (Mac)
                    apt install stockfish   (Linux)
                    https://stockfishchess.org/download/ (Windows)
"""

import math
import chess
import chess.engine
import chess.pgn
import io
import shutil
from typing import Optional
from dataclasses import dataclass, field


# ── Stockfish search paths ───────────────────────────────────────────────────
STOCKFISH_PATHS = [
    "stockfish",
    "/usr/games/stockfish",        # Debian/Ubuntu apt install location
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish", # Homebrew Apple Silicon
    "/usr/local/opt/stockfish/bin/stockfish",
    "C:/Users/Public/stockfish/stockfish.exe",
]


def _find_stockfish() -> Optional[str]:
    for path in STOCKFISH_PATHS:
        if shutil.which(path):
            return path
        try:
            import os
            if os.path.isfile(path):
                return path
        except Exception:
            pass
    return None


# ── Lichess accuracy formulas ────────────────────────────────────────────────

def win_percent(cp: float) -> float:
    """Convert centipawns (from White's perspective) to win-percentage 0-100."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy_score(cp_before: float, cp_after: float, is_white: bool) -> float:
    """
    Lichess single-move accuracy (0-100).
    cp_before / cp_after are from WHITE's perspective.
    """
    if is_white:
        wp_before = win_percent(cp_before)
        wp_after  = win_percent(cp_after)
    else:
        wp_before = win_percent(-cp_before)
        wp_after  = win_percent(-cp_after)

    raw = 103.1668100711649 * math.exp(
        -0.04354415386753951 * max(0.0, wp_before - wp_after)
    ) - 3.166924740191411

    return max(0.0, min(100.0, raw))


def classify_move(accuracy: float) -> str:
    """Turn a move-accuracy score into a Lichess-style label."""
    if accuracy >= 99.5:
        return "best"
    elif accuracy >= 90:
        return "excellent"
    elif accuracy >= 75:
        return "good"
    elif accuracy >= 60:
        return "inaccuracy"
    elif accuracy >= 40:
        return "mistake"
    else:
        return "blunder"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class MoveAnalysis:
    move_number: int
    color: str                       # "white" | "black"
    san: str                         # e.g. "Nf3"
    uci: str                         # e.g. "g1f3"
    fen_before: str
    fen_after: str
    eval_before: Optional[float]     # centipawns, White's perspective
    eval_after: Optional[float]
    best_move_san: Optional[str]
    move_accuracy: Optional[float]   # Lichess 0-100
    classification: str              # "best" / "excellent" / "good" / "inaccuracy" / "mistake" / "blunder"
    is_check: bool
    is_capture: bool


@dataclass
class GameAnalysis:
    moves: list[MoveAnalysis] = field(default_factory=list)
    white_accuracy: Optional[float] = None
    black_accuracy: Optional[float] = None
    opening_name: str = "Unknown Opening"
    pgn: str = ""
    stockfish_available: bool = True
    error: Optional[str] = None


# ── Main analysis function ────────────────────────────────────────────────────

def analyze_game(moves_san: list[str], depth: int = 15) -> GameAnalysis:
    """
    Analyse a game given as a list of SAN moves.
    Returns a GameAnalysis with per-move accuracy and aggregated stats.
    """
    stockfish_path = _find_stockfish()
    result = GameAnalysis(stockfish_available=stockfish_path is not None)

    # Build python-chess board and validate moves
    board = chess.Board()
    game  = chess.pgn.Game()
    node  = game
    valid_moves: list[chess.Move] = []

    for san in moves_san:
        try:
            move = board.parse_san(san)
            node = node.add_variation(move)
            valid_moves.append(move)
            board.push(move)
        except Exception:
            break

    result.pgn = str(game)
    result.opening_name = _identify_opening(valid_moves[:12])

    if not valid_moves:
        result.error = "No valid moves found."
        return result

    if not stockfish_path:
        # Return moves without evaluation
        board = chess.Board()
        for i, move in enumerate(valid_moves):
            is_white = board.turn == chess.WHITE
            san = board.san(move)
            fen_before = board.fen()
            is_cap = board.is_capture(move)
            board.push(move)
            result.moves.append(MoveAnalysis(
                move_number=(i // 2) + 1,
                color="white" if is_white else "black",
                san=san,
                uci=move.uci(),
                fen_before=fen_before,
                fen_after=board.fen(),
                eval_before=None,
                eval_after=None,
                best_move_san=None,
                move_accuracy=None,
                classification="?",
                is_check=board.is_check(),
                is_capture=is_cap,
            ))
        result.error = "Stockfish not found — install it for accuracy scores."
        return result

    # ── Stockfish evaluation ─────────────────────────────────────────────────
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board()
            white_accs: list[float] = []
            black_accs: list[float] = []

            # Evaluate starting position
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            cp_prev = _score_cp(info["score"])

            for i, move in enumerate(valid_moves):
                is_white  = board.turn == chess.WHITE
                san       = board.san(move)
                fen_before = board.fen()
                is_cap    = board.is_capture(move)

                best_move_obj = info.get("pv", [None])[0]
                best_san = board.san(best_move_obj) if best_move_obj else None

                board.push(move)

                # Evaluate after the move
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                cp_after = _score_cp(info["score"])

                acc = move_accuracy_score(cp_prev, cp_after, is_white)
                cls = classify_move(acc)

                if is_white:
                    white_accs.append(acc)
                else:
                    black_accs.append(acc)

                result.moves.append(MoveAnalysis(
                    move_number=(i // 2) + 1,
                    color="white" if is_white else "black",
                    san=san,
                    uci=move.uci(),
                    fen_before=fen_before,
                    fen_after=board.fen(),
                    eval_before=cp_prev,
                    eval_after=cp_after,
                    best_move_san=best_san,
                    move_accuracy=round(acc, 1),
                    classification=cls,
                    is_check=board.is_check(),
                    is_capture=is_cap,
                ))

                cp_prev = cp_after

    except Exception as e:
        result.stockfish_available = False
        result.error = f"Stockfish error: {e}"
        return result

    # Lichess game accuracy = average of move accuracies
    result.white_accuracy = round(sum(white_accs) / len(white_accs), 1) if white_accs else None
    result.black_accuracy = round(sum(black_accs) / len(black_accs), 1) if black_accs else None

    return result


def _score_cp(score: chess.engine.PovScore) -> float:
    """Centipawns from White's perspective; clamp mate scores."""
    try:
        if score.is_mate():
            mate = score.white().mate()
            return 10000.0 if (mate and mate > 0) else -10000.0
        val = score.white().score()
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


# ── Opening book (ECO subset) ─────────────────────────────────────────────────

OPENINGS = [
    (["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],              "Ruy López"),
    (["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],              "Italian Game"),
    (["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],              "Scotch Game"),
    (["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3"], "Giuoco Piano"),
    (["e2e4", "c7c5"],                                       "Sicilian Defense"),
    (["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"], "Sicilian Najdorf"),
    (["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4"], "Sicilian Open"),
    (["e2e4", "e7e6"],                                       "French Defense"),
    (["e2e4", "c7c6"],                                       "Caro-Kann Defense"),
    (["e2e4", "d7d5"],                                       "Scandinavian Defense"),
    (["e2e4", "g8f6"],                                       "Alekhine's Defense"),
    (["d2d4", "d7d5", "c2c4"],                               "Queen's Gambit"),
    (["d2d4", "d7d5", "c2c4", "e7e6"],                      "Queen's Gambit Declined"),
    (["d2d4", "d7d5", "c2c4", "c7c6"],                      "Slav Defense"),
    (["d2d4", "d7d5", "c2c4", "d5c4"],                      "Queen's Gambit Accepted"),
    (["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4"], "King's Indian Defense"),
    (["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5"],     "Queen's Gambit Declined"),
    (["d2d4", "g8f6", "c2c4", "c7c5"],                      "Benoni Defense"),
    (["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"],              "Catalan Opening"),
    (["e2e4", "e7e5"],                                       "Open Game"),
    (["d2d4", "d7d5"],                                       "Closed Game"),
    (["g1f3", "d7d5", "g2g3"],                               "King's Indian Attack"),
    (["c2c4"],                                               "English Opening"),
    (["g1f3"],                                               "Réti Opening"),
    (["e2e4"],                                               "King's Pawn Opening"),
    (["d2d4"],                                               "Queen's Pawn Opening"),
]


def _identify_opening(moves: list[chess.Move]) -> str:
    uci = [m.uci() for m in moves]
    best, best_len = "Unknown Opening", 0
    for pattern, name in OPENINGS:
        n = len(pattern)
        if uci[:n] == pattern and n > best_len:
            best, best_len = name, n
    return best

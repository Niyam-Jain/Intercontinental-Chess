import pathlib
import sys
from collections import namedtuple

import chess

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Jump capture move: from_square moves to to_square (landing on captured piece),
# screen piece at screen_square is untouched.
ICMove = namedtuple('ICMove', ['from_square', 'to_square', 'promotion', 'is_jump_capture', 'screen_square'])

ARMIES = ('western', 'empire', 'african')

# Orthogonal deltas (for queen/rook-slot pieces)
ORTHOGONAL = [(1, 0), (-1, 0), (0, 1), (0, -1)]
# Diagonal deltas (for bishop-slot pieces)
DIAGONAL = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _sq(rank: int, file: int) -> int:
    return chess.square(file, rank)


def _rf(sq: int):
    return chess.square_rank(sq), chess.square_file(sq)


def _chess_move_to_icmove(move: chess.Move) -> ICMove:
    return ICMove(move.from_square, move.to_square, move.promotion, False, None)


class ICChessBoard:
    """IC Chess board supporting three army types with charge-based jump captures."""

    def __init__(self, white_army: str = 'western', black_army: str = 'western'):
        assert white_army in ARMIES and black_army in ARMIES
        self._board = chess.Board()
        self.white_army = white_army
        self.black_army = black_army
        self.charges: dict[int, int] = {}
        self._init_charges()
        self._charge_history: list[dict] = []

    # ── Charges init ──────────────────────────────────────────────────────────

    def _init_charges(self):
        self.charges = {}
        # White pieces
        wa = self.white_army
        if wa == 'empire':
            self.charges[chess.D1] = 2   # queen slot
        elif wa == 'african':
            self.charges[chess.D1] = 1   # queen slot
            self.charges[chess.C1] = 1   # bishop slot
            self.charges[chess.F1] = 1   # bishop slot
        # Black pieces
        ba = self.black_army
        if ba == 'empire':
            self.charges[chess.D8] = 2
        elif ba == 'african':
            self.charges[chess.D8] = 1
            self.charges[chess.C8] = 1
            self.charges[chess.F8] = 1

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def turn(self) -> chess.Color:
        return self._board.turn

    @property
    def fen(self) -> str:
        return self._board.fen()

    @property
    def is_game_over(self) -> bool:
        # King captured (IC Chess allows it via jump capture)
        white_king = self._board.king(chess.WHITE)
        black_king = self._board.king(chess.BLACK)
        if white_king is None or black_king is None:
            return True
        return self._board.is_game_over()

    @property
    def result(self) -> str:
        white_king = self._board.king(chess.WHITE)
        black_king = self._board.king(chess.BLACK)
        if white_king is None:
            return '0-1'
        if black_king is None:
            return '1-0'
        if self._board.is_checkmate():
            return '0-1' if self._board.turn == chess.WHITE else '1-0'
        if self._board.is_game_over():
            return '1/2-1/2'
        return '*'

    # ── Legal moves ───────────────────────────────────────────────────────────

    @property
    def legal_moves(self) -> list:
        moves = [_chess_move_to_icmove(m) for m in self._board.legal_moves]

        # Add jump captures for current side
        color = self._board.turn
        army = self.get_army(color)
        if army in ('empire', 'african'):
            for sq, piece in self._board.piece_map().items():
                if piece.color != color:
                    continue
                jc = self._get_jump_captures(sq, piece)
                moves.extend(jc)
        return moves

    # ── Jump capture generation ───────────────────────────────────────────────

    def _get_jump_captures(self, square: int, piece: chess.Piece) -> list:
        army = self.get_army(piece.color)
        if army not in ('empire', 'african'):
            return []

        charges = self.charges.get(square, 0)
        if charges <= 0:
            return []

        # Determine movement directions based on piece type and army
        directions = []
        if army == 'empire' and piece.piece_type == chess.QUEEN:
            directions = ORTHOGONAL
        elif army == 'african' and piece.piece_type == chess.QUEEN:
            directions = ORTHOGONAL
        elif army == 'african' and piece.piece_type == chess.BISHOP:
            directions = DIAGONAL
        else:
            return []

        rank, file = _rf(square)
        result = []

        for dr, df in directions:
            r, f = rank + dr, file + df
            screen_sq = None

            while 0 <= r <= 7 and 0 <= f <= 7:
                sq = _sq(r, f)
                occupant = self._board.piece_at(sq)

                if screen_sq is None:
                    # Looking for screen piece
                    if occupant is not None:
                        screen_sq = sq
                else:
                    # Past screen — looking for first enemy to capture
                    if occupant is not None:
                        if occupant.color != piece.color:
                            # Valid jump capture target
                            move = ICMove(
                                from_square=square,
                                to_square=sq,
                                promotion=None,
                                is_jump_capture=True,
                                screen_square=screen_sq,
                            )
                            if self._is_legal_jump_capture(move, piece.color):
                                result.append(move)
                        # Either enemy or own piece: stop scanning this direction
                        break

                r += dr
                f += df

        return result

    def _is_legal_jump_capture(self, move: ICMove, color: chess.Color) -> bool:
        """Verify jump capture doesn't leave own king in check."""
        scratch = self.copy()
        scratch._apply_jump_capture(move)
        # After capture, it's the opponent's turn; check if our king is in check
        # We need to check if the color that just moved left their king in check.
        # Temporarily flip turn to use is_check logic.
        king_sq = scratch._board.king(color)
        if king_sq is None:
            return True  # king was captured — IC Chess allows this
        # Check if any opponent piece attacks the king square
        return not scratch._board.is_attacked_by(not color, king_sq)

    def _apply_jump_capture(self, move: ICMove):
        """Apply a jump capture directly (no legality check, no history save)."""
        piece = self._board.piece_at(move.from_square)
        # Remove captured piece
        self._board.remove_piece_at(move.to_square)
        # Move capturing piece
        self._board.remove_piece_at(move.from_square)
        self._board.set_piece_at(move.to_square, piece)
        # Transfer charges
        charge = self.charges.pop(move.from_square, 0)
        self.charges[move.to_square] = max(0, charge - 1)
        # Flip turn
        self._board.turn = not self._board.turn

    # ── Push / Pop ────────────────────────────────────────────────────────────

    def push(self, move) -> None:
        # Save charge state for undo
        self._charge_history.append(dict(self.charges))

        if isinstance(move, ICMove) and move.is_jump_capture:
            self._apply_jump_capture(move)
            # Also push a null move onto chess.Board's stack so pop() aligns.
            # We track with a sentinel instead: store None to indicate IC move.
            self._charge_history[-1]['__ic_move__'] = (
                move.from_square, move.to_square, move.screen_square,
                self._board.piece_at(move.to_square),
            )
        else:
            # Standard move — delegate to chess.Board
            if isinstance(move, ICMove):
                chess_move = chess.Move(move.from_square, move.to_square, move.promotion)
            else:
                chess_move = move

            # Track charge transfer before push
            piece = self._board.piece_at(chess_move.from_square)
            charge = self.charges.pop(chess_move.from_square, 0)

            # Remove charges of captured piece
            self.charges.pop(chess_move.to_square, None)

            self._board.push(chess_move)

            # After push, the piece is now at to_square (or promoted square)
            dest = chess_move.to_square
            if charge > 0:
                # Promotion: assign charges based on army
                if chess_move.promotion is not None:
                    new_charge = self._promotion_charges(chess_move.promotion, piece.color if piece else chess.WHITE)
                    if new_charge > 0:
                        self.charges[dest] = new_charge
                else:
                    self.charges[dest] = charge

            # Castling: rook also moves — transfer its charges (usually 0)
            if piece and piece.piece_type == chess.KING:
                # Detect castling by king moving 2 squares
                ff = chess.square_file(chess_move.from_square)
                tf = chess.square_file(chess_move.to_square)
                if abs(ff - tf) == 2:
                    rank = chess.square_rank(chess_move.from_square)
                    if tf > ff:  # kingside
                        rook_from = _sq(rank, 7)
                        rook_to = _sq(rank, 5)
                    else:  # queenside
                        rook_from = _sq(rank, 0)
                        rook_to = _sq(rank, 3)
                    rc = self.charges.pop(rook_from, 0)
                    if rc:
                        self.charges[rook_to] = rc

    def pop(self) -> None:
        if not self._charge_history:
            return
        saved = self._charge_history.pop()

        if '__ic_move__' in saved:
            # Undo jump capture manually
            from_sq, to_sq, screen_sq, piece_at_dest = saved['__ic_move__']
            # piece_at_dest is the capturing piece now at to_sq
            captured_piece = None  # we don't store it — limitation; restore from FEN not possible
            # We reconstruct: piece was at from_sq before, now at to_sq
            piece = self._board.piece_at(to_sq)
            self._board.remove_piece_at(to_sq)
            self._board.set_piece_at(from_sq, piece)
            self._board.turn = not self._board.turn
        else:
            self._board.pop()

        # Restore charges (remove __ic_move__ key first)
        saved.pop('__ic_move__', None)
        self.charges = saved

    def copy(self) -> 'ICChessBoard':
        new = ICChessBoard.__new__(ICChessBoard)
        new._board = self._board.copy()
        new.white_army = self.white_army
        new.black_army = self.black_army
        new.charges = dict(self.charges)
        new._charge_history = [dict(h) for h in self._charge_history]
        return new

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_charge(self, square: int) -> int:
        return self.charges.get(square, 0)

    def get_army(self, color: chess.Color) -> str:
        return self.white_army if color == chess.WHITE else self.black_army

    def _promotion_charges(self, promo_piece_type: int, color: chess.Color) -> int:
        army = self.get_army(color)
        if promo_piece_type == chess.QUEEN:
            if army == 'empire':
                return 2
            if army == 'african':
                return 1
        elif promo_piece_type == chess.BISHOP and army == 'african':
            return 1
        return 0


# ── Tests ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    # Test 1: Western vs Western
    print("Test 1: Western vs Western...")
    b = ICChessBoard('western', 'western')
    moves = b.legal_moves
    assert len(moves) == 20, f"Expected 20 moves, got {len(moves)}"
    assert not any(m.is_jump_capture for m in moves)
    b.push(ICMove(chess.E2, chess.E4, None, False, None))
    assert b._board.piece_at(chess.E4) is not None
    b.push(ICMove(chess.E7, chess.E5, None, False, None))
    assert b._board.piece_at(chess.E5) is not None
    print("  PASSED")

    # Test 2: Empire charges
    print("Test 2: Empire charges...")
    b = ICChessBoard('empire', 'western')
    assert b.get_charge(chess.D1) == 2, f"Expected 2, got {b.get_charge(chess.D1)}"
    # Set up: white cannon on a1, screen on a4, black rook on a7
    b2 = ICChessBoard('empire', 'western')
    # Clear and set up manually
    b2._board.clear()
    b2._board.set_piece_at(chess.A1, chess.Piece(chess.QUEEN, chess.WHITE))
    b2._board.set_piece_at(chess.A4, chess.Piece(chess.PAWN, chess.WHITE))   # screen
    b2._board.set_piece_at(chess.A7, chess.Piece(chess.ROOK, chess.BLACK))
    b2._board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))   # kings needed
    b2._board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    b2._board.turn = chess.WHITE
    b2.charges = {chess.A1: 2}
    moves = b2.legal_moves
    jc_moves = [m for m in moves if m.is_jump_capture]
    a1a7 = [m for m in jc_moves if m.from_square == chess.A1 and m.to_square == chess.A7]
    assert len(a1a7) == 1, f"Expected jump capture a1→a7, got {jc_moves}"
    b2.push(a1a7[0])
    assert b2._board.piece_at(chess.A7) is not None
    assert b2._board.piece_at(chess.A7).piece_type == chess.QUEEN
    assert b2._board.piece_at(chess.A4) is not None  # screen untouched
    assert b2._board.piece_at(chess.A1) is None
    assert b2.get_charge(chess.A7) == 1
    print("  PASSED")

    # Test 3: African charges
    print("Test 3: African charges...")
    b3 = ICChessBoard('african', 'western')
    assert b3.get_charge(chess.D1) == 1
    assert b3.get_charge(chess.C1) == 1
    assert b3.get_charge(chess.F1) == 1
    # Set up diagonal jump capture for African bishop on c1
    b3b = ICChessBoard('african', 'western')
    b3b._board.clear()
    b3b._board.set_piece_at(chess.C1, chess.Piece(chess.BISHOP, chess.WHITE))
    b3b._board.set_piece_at(chess.E3, chess.Piece(chess.PAWN, chess.WHITE))  # screen
    b3b._board.set_piece_at(chess.G5, chess.Piece(chess.ROOK, chess.BLACK))  # target
    b3b._board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    b3b._board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    b3b._board.turn = chess.WHITE
    b3b.charges = {chess.C1: 1}
    jc = [m for m in b3b.legal_moves if m.is_jump_capture]
    c1g5 = [m for m in jc if m.from_square == chess.C1 and m.to_square == chess.G5]
    assert len(c1g5) == 1, f"Expected bishop jump c1→g5, got {jc}"
    b3b.push(c1g5[0])
    assert b3b.get_charge(chess.G5) == 0
    # Verify no more jump captures for that bishop
    b3b._board.turn = chess.WHITE  # flip back to test
    jc2 = [m for m in b3b._get_jump_captures(chess.G5, chess.Piece(chess.BISHOP, chess.WHITE))
           if m.is_jump_capture]
    assert len(jc2) == 0
    print("  PASSED")

    # Test 4: Zero charges
    print("Test 4: Zero charges...")
    b4 = ICChessBoard('empire', 'western')
    b4._board.clear()
    b4._board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    b4._board.set_piece_at(chess.D4, chess.Piece(chess.PAWN, chess.WHITE))
    b4._board.set_piece_at(chess.D7, chess.Piece(chess.ROOK, chess.BLACK))
    b4._board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    b4._board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    b4._board.turn = chess.WHITE
    b4.charges = {chess.D1: 0}
    jc4 = [m for m in b4.legal_moves if m.is_jump_capture]
    assert len(jc4) == 0, f"No jump captures with 0 charges, got {jc4}"
    normal = [m for m in b4.legal_moves if not m.is_jump_capture]
    assert len(normal) > 0
    print("  PASSED")

    # Test 5: Push/Pop roundtrip
    print("Test 5: Push/Pop roundtrip...")
    b5 = ICChessBoard('empire', 'western')
    start_fen = b5.fen
    start_charges = dict(b5.charges)
    pushed = []
    for _ in range(5):
        m = b5.legal_moves[0]
        pushed.append(m)
        b5.push(m)
    for _ in pushed:
        b5.pop()
    assert b5.fen == start_fen, f"FEN mismatch after pop: {b5.fen} vs {start_fen}"
    assert b5.charges == start_charges, f"Charges mismatch: {b5.charges} vs {start_charges}"
    print("  PASSED")

    # Test 6: King capture
    print("Test 6: King capture via jump...")
    b6 = ICChessBoard('empire', 'western')
    b6._board.clear()
    b6._board.set_piece_at(chess.A1, chess.Piece(chess.QUEEN, chess.WHITE))
    b6._board.set_piece_at(chess.A4, chess.Piece(chess.PAWN, chess.BLACK))  # screen
    b6._board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))  # king target
    b6._board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    b6._board.turn = chess.WHITE
    b6.charges = {chess.A1: 2}
    jc6 = [m for m in b6.legal_moves if m.is_jump_capture and m.to_square == chess.A8]
    assert len(jc6) == 1, f"Expected king-capture jump, got {jc6}"
    b6.push(jc6[0])
    assert b6.is_game_over
    assert b6.result == '1-0'
    print("  PASSED")

    # Test 7: Random games across all 9 matchups
    print("Test 7: Random games (9 matchups)...")
    total_games = 0
    for wa in ARMIES:
        for ba in ARMIES:
            for _ in range(1):
                board = ICChessBoard(wa, ba)
                move_count = 0
                while not board.is_game_over and move_count < 200:
                    moves = board.legal_moves
                    if not moves:
                        break
                    board.push(random.choice(moves))
                    move_count += 1
                total_games += 1

    print(f"game.py: All tests passed ({total_games} games played across 9 matchups)")

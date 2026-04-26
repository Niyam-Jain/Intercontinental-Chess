import pathlib
import sys

import chess
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.encoder import encode_board, square_to_tensor_coords

IC_NUM_PLANES = 20

# Charge normalization: 0→0.0, 1→0.5, 2→1.0
def _normalize_charge(c: int) -> float:
    return min(c, 2) / 2.0

# Army encoding: western→0.0, empire→0.5, african→1.0
_ARMY_VALUE = {'western': 0.0, 'empire': 0.5, 'african': 1.0}


def encode_ic_board(ic_board) -> np.ndarray:
    """
    Encode an ICChessBoard into a (20, 8, 8) float32 tensor.

    Planes 0-17: Standard encoder planes (delegated to encode_board).
    Plane 18:    Normalized charge count per square (0→0.0, 1→0.5, 2→1.0).
    Plane 19:    Army type of piece owner per square (western→0.0, empire→0.5, african→1.0).
                 Empty squares → 0.0.
    """
    features = np.zeros((IC_NUM_PLANES, 8, 8), dtype=np.float32)

    # Planes 0-17
    features[:18] = encode_board(ic_board._board)

    # Planes 18 & 19
    for sq, piece in ic_board._board.piece_map().items():
        row, col = square_to_tensor_coords(sq)
        charge = ic_board.charges.get(sq, 0)
        features[18, row, col] = _normalize_charge(charge)
        army = ic_board.get_army(piece.color)
        features[19, row, col] = _ARMY_VALUE[army]

    return features


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT_DIR))
    from mcts.game import ICChessBoard

    # Test 1: shape
    b = ICChessBoard('empire', 'african')
    enc = encode_ic_board(b)
    assert enc.shape == (20, 8, 8), f"Shape: {enc.shape}"

    # Test 2: charge plane (plane 18)
    # White queen d1 (empire) → 2 charges → 1.0; row=7, col=3
    r, c = square_to_tensor_coords(chess.D1)
    assert enc[18, r, c] == 1.0, f"D1 charge plane: {enc[18, r, c]}"
    # Black queen d8 (african) → 1 charge → 0.5; row=0, col=3
    r8, c8 = square_to_tensor_coords(chess.D8)
    assert enc[18, r8, c8] == 0.5, f"D8 charge plane: {enc[18, r8, c8]}"
    # Black bishops c8, f8 (african) → 1 charge → 0.5
    rc8r, rc8c = square_to_tensor_coords(chess.C8)
    rf8r, rf8c = square_to_tensor_coords(chess.F8)
    assert enc[18, rc8r, rc8c] == 0.5, f"C8 charge plane: {enc[18, rc8r, rc8c]}"
    assert enc[18, rf8r, rf8c] == 0.5, f"F8 charge plane: {enc[18, rf8r, rf8c]}"

    # Test 3: army plane (plane 19)
    # All white pieces → empire → 0.5
    for sq, piece in b._board.piece_map().items():
        row, col = square_to_tensor_coords(sq)
        if piece.color == chess.WHITE:
            assert enc[19, row, col] == 0.5, f"White piece at {sq} should be 0.5 (empire)"
        else:
            assert enc[19, row, col] == 1.0, f"Black piece at {sq} should be 1.0 (african)"
    # Empty squares → 0.0
    for sq in chess.SQUARES:
        if b._board.piece_at(sq) is None:
            row, col = square_to_tensor_coords(sq)
            assert enc[19, row, col] == 0.0

    # Test 4: after a move, charges update
    b2 = ICChessBoard('empire', 'western')
    b2.push(chess.Move.from_uci("e2e4"))
    enc2 = encode_ic_board(b2)
    # White queen still on d1 → 2 charges
    r, c = square_to_tensor_coords(chess.D1)
    assert enc2[18, r, c] == 1.0, "Queen charge should still be 1.0 after pawn move"

    print("encoder_ic.py: All tests passed")

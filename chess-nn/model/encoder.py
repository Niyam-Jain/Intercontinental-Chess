import chess
import numpy as np


PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def square_to_tensor_coords(square: chess.Square) -> tuple[int, int]:
    row = 7 - chess.square_rank(square)
    col = chess.square_file(square)
    return row, col


def encode_board(board: chess.Board) -> np.ndarray:
    features = np.zeros((18, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        row, col = square_to_tensor_coords(square)
        features[plane, row, col] = 1.0

    if board.turn == chess.WHITE:
        features[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        features[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        features[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        features[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        features[16, :, :] = 1.0

    if board.ep_square is not None:
        ep_row, ep_col = square_to_tensor_coords(board.ep_square)
        features[17, ep_row, ep_col] = 1.0

    return features


if __name__ == "__main__":
    b = chess.Board()
    x = encode_board(b)
    assert x.shape == (18, 8, 8)
    assert x.dtype == np.float32
    assert int(x[0].sum()) == 8
    assert int(x[0, 6, :].sum()) == 8
    assert int(x[5].sum()) == 1
    assert x[5, 7, 4] == 1.0
    assert int(x[11].sum()) == 1
    assert x[11, 0, 4] == 1.0
    assert np.all(x[12] == 1.0)
    assert np.all(x[13] == 1.0)
    assert np.all(x[14] == 1.0)
    assert np.all(x[15] == 1.0)
    assert np.all(x[16] == 1.0)
    assert np.all(x[17] == 0.0)

    b.push(chess.Move.from_uci("e2e4"))
    x2 = encode_board(b)
    assert np.all(x2[12] == 0.0)
    assert int(x2[17].sum()) == 1
    assert x2[17, 5, 4] == 1.0

    print("encoder.py: All tests passed")

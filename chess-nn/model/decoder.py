import chess


POLICY_SIZE = 4288
UNDERPROMOTION_BASE = 4096
UNDERPROMOTION_TO_OFFSET = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
}
OFFSET_TO_PROMOTION = {
    0: chess.KNIGHT,
    1: chess.BISHOP,
    2: chess.ROOK,
}


def move_to_index(move: chess.Move) -> int:
    if move.promotion in UNDERPROMOTION_TO_OFFSET:
        return UNDERPROMOTION_BASE + (UNDERPROMOTION_TO_OFFSET[move.promotion] * 64) + move.to_square
    return (move.from_square * 64) + move.to_square


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    if index < 0 or index >= POLICY_SIZE:
        return None

    if index < UNDERPROMOTION_BASE:
        from_square = index // 64
        to_square = index % 64
        candidate = chess.Move(from_square=from_square, to_square=to_square)
        if candidate in board.legal_moves:
            return candidate

        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_square)
            if (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0):
                queen_candidate = chess.Move(from_square=from_square, to_square=to_square, promotion=chess.QUEEN)
                if queen_candidate in board.legal_moves:
                    return queen_candidate
        return None

    rel = index - UNDERPROMOTION_BASE
    promo_offset = rel // 64
    to_square = rel % 64
    if promo_offset not in OFFSET_TO_PROMOTION:
        return None
    promo_piece = OFFSET_TO_PROMOTION[promo_offset]

    for move in board.legal_moves:
        if move.to_square == to_square and move.promotion == promo_piece:
            return move
    return None


def get_legal_move_mask(board: chess.Board) -> list[int]:
    return [move_to_index(move) for move in board.legal_moves]


if __name__ == "__main__":
    start = chess.Board()
    mask = get_legal_move_mask(start)
    assert len(mask) == 20

    e2e4 = chess.Move.from_uci("e2e4")
    idx_e2e4 = move_to_index(e2e4)
    dec_e2e4 = index_to_move(idx_e2e4, start)
    assert dec_e2e4 == e2e4

    a7a8q = chess.Move.from_uci("a7a8q")
    idx_q = move_to_index(a7a8q)
    assert 0 <= idx_q <= 4095

    a7a8n = chess.Move.from_uci("a7a8n")
    idx_n = move_to_index(a7a8n)
    assert 4096 <= idx_n <= 4159

    for mv in chess.Board().legal_moves:
        idx = move_to_index(mv)
        dec = index_to_move(idx, chess.Board())
        assert dec == mv

    print("decoder.py: All tests passed")

/**
 * encoder.js — IC Chess board encoder for ONNX Runtime Web inference.
 *
 * Mirrors model/encoder_ic.py exactly. Produces a Float32Array of shape
 * [20 * 8 * 8 = 1280] representing a (20, 8, 8) tensor in row-major order.
 *
 * Plane layout (matches Python encoder.py + encoder_ic.py):
 *   0  White pawns        6  Black pawns
 *   1  White knights      7  Black knights
 *   2  White bishops      8  Black bishops
 *   3  White rooks        9  Black rooks
 *   4  White queens      10  Black queens
 *   5  White kings       11  Black kings
 *  12  Turn (all 1.0 if white to move)
 *  13  White kingside castling rights (all 1.0 if available)
 *  14  White queenside castling rights
 *  15  Black kingside castling rights
 *  16  Black queenside castling rights
 *  17  En passant square (single 1.0 at the ep square)
 *  18  Charge count per square: 0→0.0, 1→0.5, 2→1.0
 *  19  Army per square: western→0.0, empire→0.5, african→1.0
 *
 * Board orientation: gameState.board[r][c] where r=0 is rank 8 (Black's back
 * rank) and r=7 is rank 1 (White's back rank). This matches python-chess's
 * square_to_tensor_coords: row = 7 - rank, col = file.
 * Therefore tensor row == game board row — no orientation flip needed.
 *
 * @param {Object} gameState
 *   .board      8×8 array; each cell is null or { type, color, charges? }
 *               type: 'p'|'n'|'b'|'r'|'q'|'k'
 *               color: 'w'|'b'
 *               charges: number (optional, defaults to 0)
 *   .turn       'w' | 'b'
 *   .castling   { w: { k: bool, q: bool }, b: { k: bool, q: bool } }
 *   .enPassant  { r, c } | null
 *   .armies     { w: 'western'|'empire'|'african', b: same }
 *
 * @returns {Float32Array} length 1280 (20 planes × 64 squares)
 */
function encodeBoard(gameState) {
    const PLANES = 20;
    const planes = new Float32Array(PLANES * 64);

    const PIECE_PLANE = {
        'pw': 0, 'nw': 1, 'bw': 2, 'rw': 3, 'qw': 4, 'kw': 5,
        'pb': 6, 'nb': 7, 'bb': 8, 'rb': 9, 'qb': 10, 'kb': 11,
    };
    const ARMY_VAL = { western: 0.0, empire: 0.5, african: 1.0 };

    const board = gameState.board;

    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const piece = board[r][c];
            if (!piece) continue;

            const sq = r * 8 + c;

            // Piece occupancy plane (0-11)
            const planeIdx = PIECE_PLANE[piece.type + piece.color];
            if (planeIdx !== undefined) {
                planes[planeIdx * 64 + sq] = 1.0;
            }

            // Plane 18: charge count (0→0.0, 1→0.5, ≥2→1.0)
            const charges = piece.charges || 0;
            if (charges > 0) {
                planes[18 * 64 + sq] = Math.min(charges, 2) / 2.0;
            }

            // Plane 19: army type (piece.army is set directly on each piece by placePiece())
            planes[19 * 64 + sq] = ARMY_VAL[piece.army] || 0.0;
        }
    }

    // Plane 12: turn
    if (gameState.turn === 'w') {
        planes.fill(1.0, 12 * 64, 13 * 64);
    }

    // Planes 13-16: castling rights
    if (gameState.castling.w.k) planes.fill(1.0, 13 * 64, 14 * 64);
    if (gameState.castling.w.q) planes.fill(1.0, 14 * 64, 15 * 64);
    if (gameState.castling.b.k) planes.fill(1.0, 15 * 64, 16 * 64);
    if (gameState.castling.b.q) planes.fill(1.0, 16 * 64, 17 * 64);

    // Plane 17: en passant
    if (gameState.enPassant) {
        const epSq = gameState.enPassant.r * 8 + gameState.enPassant.c;
        planes[17 * 64 + epSq] = 1.0;
    }

    return planes;
}

/**
 * Convert a game move { from: {r,c}, to: {r,c} } to a policy index.
 *
 * Policy index encoding (matches mcts.py / decoder.py):
 *   Normal moves:       from_sq * 64 + to_sq   (0–4095)
 *   Underpromotions:    4096 + offset * 64 + to_sq
 *     offset 0 = knight, 1 = bishop, 2 = rook
 *
 * Square index: sq = (7 - r) * 8 + c  (python-chess format: a1=0, h8=63)
 */
function moveToPolicyIndex(move, promotionPiece) {
    const fromSq = (7 - move.from.r) * 8 + move.from.c;
    const toSq   = (7 - move.to.r)   * 8 + move.to.c;

    const UNDERPROMO_OFFSET = { n: 0, b: 1, r: 2 };
    if (promotionPiece && promotionPiece !== 'q') {
        const offset = UNDERPROMO_OFFSET[promotionPiece];
        if (offset !== undefined) {
            return 4096 + offset * 64 + toSq;
        }
    }
    return fromSq * 64 + toSq;
}

/**
 * Convert a policy index back to { from: {r,c}, to: {r,c} }.
 * For underpromotions the from-square is not encoded in the index; callers
 * must match against legal moves to resolve it.
 */
function policyIndexToMove(index) {
    if (index < 4096) {
        const fromSq = Math.floor(index / 64);
        const toSq   = index % 64;
        return {
            from: { r: 7 - Math.floor(fromSq / 8), c: fromSq % 8 },
            to:   { r: 7 - Math.floor(toSq  / 8), c: toSq  % 8 },
        };
    }
    // Underpromotion: from-square unknown without legal move context
    const rel    = index - 4096;
    const toSq   = rel % 64;
    return {
        from: null,
        to:   { r: 7 - Math.floor(toSq / 8), c: toSq % 8 },
        underpromotionOffset: Math.floor(rel / 64),
    };
}

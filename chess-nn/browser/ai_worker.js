/**
 * ai_worker.js — Web Worker for IC Chess neural network AI.
 *
 * Loads the ONNX model once, then handles search requests asynchronously
 * so the UI thread is never blocked.
 *
 * Messages IN  (postMessage from main thread):
 *   { type: 'init',   modelUrl: string }
 *   { type: 'search', gameState: Object, difficulty: string }
 *   { type: 'cancel' }
 *
 * Messages OUT (postMessage to main thread):
 *   { type: 'ready' }
 *   { type: 'thinking', simsDone: number, simsTotal: number }
 *   { type: 'move',  move: { from: {r,c}, to: {r,c} }, eval: number }
 *   { type: 'error', message: string }
 */

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
importScripts('encoder.js');
importScripts('mcts.js');

// ── Difficulty presets ───────────────────────────────────────────────────────
const DIFFICULTY = {
    beginner: { numSimulations:    1, temperature: 1.5 },
    easy:     { numSimulations:   50, temperature: 1.0 },
    medium:   { numSimulations:  200, temperature: 0.5 },
    hard:     { numSimulations:  600, temperature: 0.1 },
    master:   { numSimulations: 1200, temperature: 0.0 },
};

// ── State ────────────────────────────────────────────────────────────────────
let session   = null;
let cancelled = false;

// ── Message handler ──────────────────────────────────────────────────────────
self.onmessage = async function (e) {
    const msg = e.data;

    if (msg.type === 'init') {
        try {
            ort.env.wasm.numThreads = 1;   // single-threaded WASM inside Worker
            session = await ort.InferenceSession.create(msg.modelUrl, {
                executionProviders: ['wasm'],
            });
            self.postMessage({ type: 'ready' });
        } catch (err) {
            self.postMessage({ type: 'error', message: `Model load failed: ${err.message}` });
        }
        return;
    }

    if (msg.type === 'cancel') {
        cancelled = true;
        return;
    }

    if (msg.type === 'search') {
        if (!session) {
            self.postMessage({ type: 'error', message: 'Model not loaded yet' });
            return;
        }
        cancelled = false;

        const config = DIFFICULTY[msg.difficulty] || DIFFICULTY.medium;
        const mcts   = new MCTS(session, config);

        try {
            const result = await mcts.search(
                msg.gameState,
                getAllMovesForState,
                applyMoveToState,
                isTerminalState,
                getResultFromState,
                () => cancelled,
            );

            if (!cancelled && result.move) {
                self.postMessage({ type: 'move', move: result.move, eval: result.value });
            }
        } catch (err) {
            self.postMessage({ type: 'error', message: `Search failed: ${err.message}` });
        }
    }
};

// ── Game state helpers ────────────────────────────────────────────────────────
// These mirror the index.html game logic for use inside the worker.
// The worker receives a plain gameState snapshot; it must re-implement move
// generation and application without touching the DOM.

/**
 * Deep-clone a gameState snapshot.
 * Values: board (8×8 of piece objects or null), turn, castling, enPassant, armies.
 */
function cloneState(gs) {
    const board = gs.board.map(row => row.map(cell =>
        cell ? { ...cell } : null
    ));
    return {
        board,
        turn:      gs.turn,
        castling:  { w: { ...gs.castling.w }, b: { ...gs.castling.b } },
        enPassant: gs.enPassant ? { ...gs.enPassant } : null,
        armies:    { ...gs.armies },
    };
}

/**
 * Return all legal moves for the current player as an array of move objects.
 * Move objects: { from: {r,c}, to: {r,c}, isJumpCapture?: bool, promotion?: string }
 *
 * This is a lightweight re-implementation of the parts of getLegalMoves() from
 * index.html that are needed for search. It delegates full legality filtering
 * (king-not-in-check) by making the move and checking for check.
 */
function getAllMovesForState(gs) {
    const moves = [];
    const color = gs.turn;
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const piece = gs.board[r][c];
            if (!piece || piece.color !== color) continue;
            const pieceMoves = getPseudoLegalMoves(r, c, gs);
            for (const m of pieceMoves) {
                if (isLegalMove(r, c, m, gs)) {
                    moves.push({ from: { r, c }, to: m });
                }
            }
        }
    }
    return moves;
}

function applyMoveToState(gs, move) {
    const next = cloneState(gs);
    executeOnState(next, move.from, move.to);
    return next;
}

function isTerminalState(gs) {
    // Game over if current player has no legal moves
    return getAllMovesForState(gs).length === 0;
}

function getResultFromState(gs) {
    // Called when isTerminalState is true. Check if current player's king is in check.
    // If in check → they lost (return -1 from their POV = good for opponent who just moved,
    // but MCTS backprop already flips sign, so return the losing side's value: -1).
    // If not in check → stalemate → 0.
    const color = gs.turn;
    const kingPos = findKing(color, gs.board);
    if (!kingPos) return -1;   // king captured — current player lost
    const opp = color === 'w' ? 'b' : 'w';
    return isAttacked(kingPos.r, kingPos.c, opp, gs.board) ? -1 : 0;
}

// ── Piece movement helpers (mirrors index.html logic) ────────────────────────

const DIRS_STRAIGHT = [[0,1],[0,-1],[1,0],[-1,0]];
const DIRS_DIAG     = [[1,1],[1,-1],[-1,1],[-1,-1]];
const DIRS_ALL      = [...DIRS_STRAIGHT, ...DIRS_DIAG];

function inBounds(r, c) { return r >= 0 && r < 8 && c >= 0 && c < 8; }

function getPseudoLegalMoves(r, c, gs) {
    const piece = gs.board[r][c];
    if (!piece) return [];
    const { type, color } = piece;
    const opp = color === 'w' ? 'b' : 'w';
    const moves = [];

    const addMove = (tr, tc, extra = {}) => {
        moves.push({ r: tr, c: tc, ...extra });
    };

    const slide = (dirs) => {
        for (const [dr, dc] of dirs) {
            for (let d = 1; d < 8; d++) {
                const tr = r + dr * d, tc = c + dc * d;
                if (!inBounds(tr, tc)) break;
                const target = gs.board[tr][tc];
                if (target) {
                    if (target.color === opp) addMove(tr, tc);
                    break;
                }
                addMove(tr, tc);
            }
        }
    };

    if (type === 'p') {
        const dir  = color === 'w' ? -1 : 1;
        const start = color === 'w' ? 6 : 1;
        const promo = color === 'w' ? 0 : 7;
        const addPawn = (tr, tc) => {
            if (tr === promo) {
                for (const p of ['q','r','b','n']) addMove(tr, tc, { promotion: p });
            } else {
                addMove(tr, tc);
            }
        };
        if (inBounds(r + dir, c) && !gs.board[r + dir][c]) {
            addPawn(r + dir, c);
            if (r === start && !gs.board[r + 2 * dir][c]) addPawn(r + 2 * dir, c);
        }
        for (const dc of [-1, 1]) {
            const tr = r + dir, tc = c + dc;
            if (!inBounds(tr, tc)) continue;
            const target = gs.board[tr][tc];
            if ((target && target.color === opp) ||
                (gs.enPassant && gs.enPassant.r === tr && gs.enPassant.c === tc)) {
                addPawn(tr, tc);
            }
        }
    } else if (type === 'n') {
        for (const [dr, dc] of [[2,1],[2,-1],[-2,1],[-2,-1],[1,2],[1,-2],[-1,2],[-1,-2]]) {
            const tr = r + dr, tc = c + dc;
            if (inBounds(tr, tc) && (!gs.board[tr][tc] || gs.board[tr][tc].color === opp))
                addMove(tr, tc);
        }
    } else if (type === 'b') {
        slide(DIRS_DIAG);
        // African bishop jump captures
        if (gs.armies[color] === 'african' && piece.charges > 0) {
            for (const [dr, dc] of DIRS_DIAG) {
                for (let d = 1; d < 8; d++) {
                    const sr = r + dr * d, sc = c + dc * d;
                    if (!inBounds(sr, sc)) break;
                    const screen = gs.board[sr][sc];
                    if (screen) {
                        const tr = sr + dr, tc = sc + dc;
                        if (inBounds(tr, tc)) {
                            const target = gs.board[tr][tc];
                            if (target && target.color === opp) {
                                addMove(tr, tc, { isJumpCapture: true });
                            }
                        }
                        break;
                    }
                }
            }
        }
    } else if (type === 'r') {
        slide(DIRS_STRAIGHT);
    } else if (type === 'q') {
        slide(DIRS_ALL);
        // Empire / African queen jump captures (piece.army is set directly on the piece)
        const army = piece.army;
        if ((army === 'empire' || army === 'african') && piece.charges > 0) {
            for (const [dr, dc] of DIRS_STRAIGHT) {
                for (let d = 1; d < 8; d++) {
                    const sr = r + dr * d, sc = c + dc * d;
                    if (!inBounds(sr, sc)) break;
                    const screen = gs.board[sr][sc];
                    if (screen) {
                        const tr = sr + dr, tc = sc + dc;
                        if (inBounds(tr, tc)) {
                            const target = gs.board[tr][tc];
                            if (target && target.color === opp) {
                                addMove(tr, tc, { isJumpCapture: true });
                            }
                        }
                        break;
                    }
                }
            }
        }
    } else if (type === 'k') {
        for (const [dr, dc] of DIRS_ALL) {
            const tr = r + dr, tc = c + dc;
            if (inBounds(tr, tc) && (!gs.board[tr][tc] || gs.board[tr][tc].color === opp))
                addMove(tr, tc);
        }
        // Castling
        if (color === 'w' && r === 7 && c === 4) {
            if (gs.castling.w.k && !gs.board[7][5] && !gs.board[7][6]) addMove(7, 6);
            if (gs.castling.w.q && !gs.board[7][3] && !gs.board[7][2] && !gs.board[7][1]) addMove(7, 2);
        } else if (color === 'b' && r === 0 && c === 4) {
            if (gs.castling.b.k && !gs.board[0][5] && !gs.board[0][6]) addMove(0, 6);
            if (gs.castling.b.q && !gs.board[0][3] && !gs.board[0][2] && !gs.board[0][1]) addMove(0, 2);
        }
    }

    return moves;
}

function isLegalMove(fromR, fromC, toMove, gs) {
    const next = cloneState(gs);
    executeOnState(next, { r: fromR, c: fromC }, toMove);
    const color = gs.board[fromR][fromC].color;
    const kingPos = findKing(color, next.board);
    if (!kingPos) return false;
    const opp = color === 'w' ? 'b' : 'w';
    return !isAttacked(kingPos.r, kingPos.c, opp, next.board);
}

function executeOnState(gs, from, to) {
    const piece = gs.board[from.r][from.c];
    if (!piece) return;

    const color = piece.color;
    const opp   = color === 'w' ? 'b' : 'w';

    // Jump capture: consume a charge
    if (to.isJumpCapture && piece.charges > 0) {
        piece.charges--;
    }

    // En passant capture
    if (piece.type === 'p' && to.c !== from.c && !gs.board[to.r][to.c]) {
        gs.board[from.r][to.c] = null;
    }

    // Castling: move rook
    if (piece.type === 'k' && Math.abs(to.c - from.c) === 2) {
        const rookSrcCol = to.c === 6 ? 7 : 0;
        const rookDstCol = to.c === 6 ? 5 : 3;
        gs.board[to.r][rookDstCol] = gs.board[to.r][rookSrcCol];
        gs.board[to.r][rookSrcCol] = null;
    }

    // Move piece
    gs.board[to.r][to.c] = piece;
    gs.board[from.r][from.c] = null;

    // Promotion
    if (piece.type === 'p' && to.promotion) {
        piece.type = to.promotion;
        piece.charges = 0;
    }

    // Update castling rights
    if (piece.type === 'k') {
        gs.castling[color].k = false;
        gs.castling[color].q = false;
    }
    if (piece.type === 'r') {
        if (from.c === 0) gs.castling[color].q = false;
        if (from.c === 7) gs.castling[color].k = false;
    }

    // En passant square
    gs.enPassant = null;
    if (piece.type === 'p' && Math.abs(to.r - from.r) === 2) {
        gs.enPassant = { r: (from.r + to.r) / 2, c: from.c };
    }

    // Switch turn
    gs.turn = opp;
}

function findKing(color, board) {
    for (let r = 0; r < 8; r++)
        for (let c = 0; c < 8; c++)
            if (board[r][c] && board[r][c].type === 'k' && board[r][c].color === color)
                return { r, c };
    return null;
}

function isAttacked(r, c, byColor, board) {
    // Check if (r,c) is attacked by any piece of byColor
    const opp = byColor === 'w' ? 'b' : 'w';

    // Pawns
    const pawnDir = byColor === 'w' ? 1 : -1;  // white pawns attack upward (row decreases)
    // White pawns at row r+1 attack diagonally to row r
    for (const dc of [-1, 1]) {
        const pr = r + pawnDir, pc = c + dc;
        if (inBounds(pr, pc)) {
            const p = board[pr][pc];
            if (p && p.color === byColor && p.type === 'p') return true;
        }
    }

    // Knights
    for (const [dr, dc] of [[2,1],[2,-1],[-2,1],[-2,-1],[1,2],[1,-2],[-1,2],[-1,-2]]) {
        const nr = r + dr, nc = c + dc;
        if (inBounds(nr, nc)) {
            const p = board[nr][nc];
            if (p && p.color === byColor && p.type === 'n') return true;
        }
    }

    // Sliders (rook/queen straight, bishop/queen diagonal)
    const check = (dirs, types) => {
        for (const [dr, dc] of dirs) {
            for (let d = 1; d < 8; d++) {
                const sr = r + dr * d, sc = c + dc * d;
                if (!inBounds(sr, sc)) break;
                const p = board[sr][sc];
                if (p) {
                    if (p.color === byColor && types.includes(p.type)) return true;
                    break;
                }
            }
        }
        return false;
    };
    if (check(DIRS_STRAIGHT, ['r','q'])) return true;
    if (check(DIRS_DIAG,     ['b','q'])) return true;

    // King
    for (const [dr, dc] of DIRS_ALL) {
        const kr = r + dr, kc = c + dc;
        if (inBounds(kr, kc)) {
            const p = board[kr][kc];
            if (p && p.color === byColor && p.type === 'k') return true;
        }
    }

    return false;
}

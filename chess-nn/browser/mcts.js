/**
 * mcts.js — JavaScript MCTS for IC Chess browser inference.
 *
 * Mirrors mcts/mcts.py. Runs inside a Web Worker (loaded by ai_worker.js)
 * so it has no DOM access. Calls ort.InferenceSession for neural network
 * evaluation via ONNX Runtime Web.
 *
 * Depends on encoder.js being importScripts'd before this file.
 */

class MCTSNode {
    constructor(prior) {
        this.prior = prior;
        this.visitCount = 0;
        this.totalValue = 0.0;
        this.children = new Map();   // moveKey → MCTSNode
        this.moves = new Map();      // moveKey → move object
        this.isExpanded = false;
    }

    get qValue() {
        return this.visitCount > 0 ? this.totalValue / this.visitCount : 0.0;
    }
}

class MCTS {
    /**
     * @param {ort.InferenceSession} session   ONNX Runtime session
     * @param {Object} config
     *   .numSimulations  number of MCTS simulations (default 200)
     *   .cPuct           exploration constant (default 1.5)
     *   .temperature     move selection temperature (0=greedy, default 0.0)
     */
    constructor(session, config = {}) {
        this.session = session;
        this.numSimulations = config.numSimulations || 200;
        this.cPuct = config.cPuct || 1.5;
        this.temperature = config.temperature !== undefined ? config.temperature : 0.0;
    }

    /** Run the neural network and return { priors: Map<moveKey,float>, value: float }. */
    async _evaluate(gameState, legalMoves) {
        const input = encodeBoard(gameState);
        const tensor = new ort.Tensor('float32', input, [1, 20, 8, 8]);
        const results = await this.session.run({ board: tensor });

        const policyData = results.policy.data;   // Float32Array(4288)
        const value      = results.value.data[0]; // float

        // Mask to legal moves and softmax
        const NEG_INF = -1e9;
        const masked = new Float32Array(4288).fill(NEG_INF);
        for (const move of legalMoves) {
            const idx = moveToPolicyIndex(move, move.promotion || null);
            if (idx >= 0 && idx < 4288) {
                masked[idx] = policyData[idx];
            }
        }

        // Stable softmax
        let maxVal = NEG_INF;
        for (let i = 0; i < 4288; i++) if (masked[i] > maxVal) maxVal = masked[i];
        let sum = 0;
        const exp = new Float32Array(4288);
        for (let i = 0; i < 4288; i++) {
            if (masked[i] > NEG_INF / 2) {
                exp[i] = Math.exp(masked[i] - maxVal);
                sum += exp[i];
            }
        }
        if (sum === 0) sum = 1;

        const priors = new Map();
        for (const move of legalMoves) {
            const idx = moveToPolicyIndex(move, move.promotion || null);
            const key = _moveKey(move);
            priors.set(key, idx >= 0 && idx < 4288 ? exp[idx] / sum : 0);
        }

        return { priors, value };
    }

    _selectChild(node) {
        const sqrtN = Math.sqrt(node.visitCount);
        let bestScore = -Infinity, bestKey = null;
        for (const [key, child] of node.children) {
            const score = child.qValue + this.cPuct * child.prior * sqrtN / (1 + child.visitCount);
            if (score > bestScore) { bestScore = score; bestKey = key; }
        }
        return bestKey;
    }

    /**
     * Run MCTS search on the given game state.
     *
     * @param {Object}   gameState     Board state (see encoder.js for schema)
     * @param {Function} getLegalMoves (r, c) => [{r,c,...}] — game's legal move function
     * @param {Function} getAllMoves   () => [{from,to,...}] — all legal moves for current turn
     * @param {Function} applyMove    (gameState, move) => newGameState (pure, no mutation)
     * @param {Function} isTerminal   (gameState) => bool
     * @param {Function} getResult    (gameState) => +1|-1|0 from current player's perspective
     * @param {Function} cancelCheck  () => bool — return true to stop early
     * @returns {{ move, value }}
     */
    async search(gameState, getAllMoves, applyMove, isTerminal, getResult, cancelCheck = () => false) {
        const root = new MCTSNode(1.0);

        const rootMoves = getAllMoves(gameState);
        if (!rootMoves.length) return { move: null, value: 0 };

        // Expand root
        const { priors, value: rootValue } = await this._evaluate(gameState, rootMoves);
        for (const move of rootMoves) {
            const key = _moveKey(move);
            root.children.set(key, new MCTSNode(priors.get(key) || 1 / rootMoves.length));
            root.moves.set(key, move);
        }
        root.isExpanded = true;
        root.visitCount = 1;

        for (let sim = 0; sim < this.numSimulations - 1; sim++) {
            if (cancelCheck()) break;

            // Post progress every 10 sims
            if (sim % 10 === 0) {
                self.postMessage({ type: 'thinking', simsDone: sim, simsTotal: this.numSimulations });
            }

            // Selection — walk down expanding leaf
            let node = root;
            let state = gameState;
            const path = [node];
            const keyPath = [];

            while (node.isExpanded && node.children.size > 0 && !isTerminal(state)) {
                const key = this._selectChild(node);
                const move = node.moves.get(key);
                state = applyMove(state, move);
                keyPath.push({ parent: node, key });
                node = node.children.get(key);
                path.push(node);
            }

            // Expansion / terminal
            let leafValue;
            if (isTerminal(state)) {
                leafValue = getResult(state);
            } else if (!node.isExpanded) {
                const moves = getAllMoves(state);
                if (!moves.length) {
                    leafValue = getResult(state);
                } else {
                    const { priors: p, value: v } = await this._evaluate(state, moves);
                    for (const m of moves) {
                        const k = _moveKey(m);
                        node.children.set(k, new MCTSNode(p.get(k) || 1 / moves.length));
                        node.moves.set(k, m);
                    }
                    node.isExpanded = true;
                    leafValue = v;
                }
            } else {
                leafValue = 0.0;
            }

            // Backpropagation — alternate sign at each ply
            let sign = 1.0;
            for (let i = path.length - 1; i >= 0; i--) {
                path[i].visitCount += 1;
                path[i].totalValue += sign * leafValue;
                sign *= -1.0;
            }
        }

        // Select move from root
        const move = this._selectMove(root);
        const rootChild = root.children.get(_moveKey(move));
        const evalValue = rootChild ? rootChild.qValue : rootValue;
        return { move, value: evalValue };
    }

    _selectMove(root) {
        if (this.temperature === 0.0) {
            let bestKey = null, bestN = -1;
            for (const [key, child] of root.children) {
                if (child.visitCount > bestN) { bestN = child.visitCount; bestKey = key; }
            }
            return root.moves.get(bestKey);
        }
        const keys   = [...root.children.keys()];
        const visits = keys.map(k => root.children.get(k).visitCount);
        const pow    = visits.map(v => Math.pow(v, 1.0 / this.temperature));
        const total  = pow.reduce((a, b) => a + b, 0);
        const probs  = pow.map(v => v / total);
        let r = Math.random(), idx = 0;
        for (; idx < probs.length - 1 && r > probs[idx]; idx++) r -= probs[idx];
        return root.moves.get(keys[idx]);
    }
}

/** Stable hashable key for a move object from index.html. */
function _moveKey(move) {
    const isJump = move.isJumpCapture ? 1 : 0;
    const promo  = move.promotion || '';
    return `${move.from.r},${move.from.c}-${move.to.r},${move.to.c}:${isJump}:${promo}`;
}

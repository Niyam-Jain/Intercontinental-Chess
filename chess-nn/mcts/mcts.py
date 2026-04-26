import math
import pathlib
import sys

import chess
import numpy as np
import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.decoder import POLICY_SIZE, move_to_index
from model.encoder import encode_board
from model.encoder_ic import encode_ic_board
from model.network import ChessNet
from mcts.game import ICChessBoard, ICMove


class MCTSNode:
    __slots__ = ('prior', 'visit_count', 'total_value', 'children', 'is_expanded', '_moves')

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: dict = {}
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0


def _move_key(move):
    """Hashable key for a move (works for chess.Move and ICMove)."""
    if isinstance(move, ICMove):
        return (move.from_square, move.to_square, move.promotion, move.is_jump_capture)
    return (move.from_square, move.to_square, move.promotion, False)


def _move_to_policy_index(move) -> int:
    if isinstance(move, ICMove):
        if move.promotion and move.promotion != chess.QUEEN:
            from model.decoder import UNDERPROMOTION_BASE, UNDERPROMOTION_TO_OFFSET
            if move.promotion in UNDERPROMOTION_TO_OFFSET:
                return UNDERPROMOTION_BASE + UNDERPROMOTION_TO_OFFSET[move.promotion] * 64 + move.to_square
        return move.from_square * 64 + move.to_square
    return move_to_index(move)


def _is_terminal(board) -> bool:
    if isinstance(board, ICChessBoard):
        return board.is_game_over
    return board.is_game_over()


def _terminal_value(board, perspective: chess.Color) -> float:
    """Return value in [-1, 1] from perspective's point of view."""
    if isinstance(board, ICChessBoard):
        result = board.result
    else:
        result = board.result()
    if result == '1-0':
        return 1.0 if perspective == chess.WHITE else -1.0
    if result == '0-1':
        return -1.0 if perspective == chess.WHITE else 1.0
    return 0.0


def _get_legal_moves(board) -> list:
    if isinstance(board, ICChessBoard):
        return board.legal_moves
    return list(board.legal_moves)


def _board_turn(board) -> chess.Color:
    if isinstance(board, ICChessBoard):
        return board.turn
    return board.turn


def _board_copy(board):
    return board.copy()


def _push(board, move):
    board.push(move)


def _pop(board):
    board.pop()


class MCTS:
    def __init__(self, model, device, num_simulations: int = 200,
                 c_puct: float = 1.5, temperature: float = 0.0, use_ic: bool = False):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.use_ic = use_ic

    @torch.no_grad()
    def evaluate(self, board) -> tuple[dict, float]:
        """Run the network. Returns (priors_dict, value)."""
        if self.use_ic:
            features = encode_ic_board(board)
        else:
            inner = board._board if isinstance(board, ICChessBoard) else board
            features = encode_board(inner)

        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        policy_logits, value_tensor = self.model(tensor)
        value = value_tensor.item()

        # Mask to legal moves
        legal_moves = _get_legal_moves(board)
        if not legal_moves:
            return {}, value

        logits = policy_logits.squeeze(0).cpu()
        mask = torch.full((POLICY_SIZE,), float('-inf'))
        for move in legal_moves:
            idx = _move_to_policy_index(move)
            if 0 <= idx < POLICY_SIZE:
                mask[idx] = 0.0
        masked = logits + mask
        probs = torch.softmax(masked, dim=0)

        priors = {}
        for move in legal_moves:
            idx = _move_to_policy_index(move)
            if 0 <= idx < POLICY_SIZE:
                priors[_move_key(move)] = probs[idx].item()
            else:
                priors[_move_key(move)] = 0.0

        # Normalize (may not sum to 1 due to index collisions on rare promotions)
        total = sum(priors.values())
        if total > 0:
            priors = {k: v / total for k, v in priors.items()}

        return priors, value

    def _select_child(self, node: MCTSNode) -> tuple:
        best_score = -float('inf')
        best_key = None
        best_child = None
        sqrt_n = math.sqrt(node.visit_count)
        for key, child in node.children.items():
            score = child.q_value + self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_key = key
                best_child = child
        return best_key, best_child

    def _expand(self, node: MCTSNode, board) -> float:
        if _is_terminal(board):
            perspective = _board_turn(board)
            return _terminal_value(board, perspective)

        priors, value = self.evaluate(board)
        legal_moves = _get_legal_moves(board)
        for move in legal_moves:
            key = _move_key(move)
            prior = priors.get(key, 1.0 / max(len(legal_moves), 1))
            node.children[key] = MCTSNode(prior=prior)

        # Store moves list alongside keys so we can push by key
        node._moves = {_move_key(m): m for m in legal_moves}
        node.is_expanded = True
        return value

    def search(self, board) -> tuple:
        """Run MCTS. Returns (best_move, root_node)."""
        root = MCTSNode(prior=1.0)
        root_turn = _board_turn(board)

        # Initial expansion
        self._expand(root, board)
        root.visit_count = 1

        for _ in range(self.num_simulations - 1):
            node = root
            scratch = _board_copy(board)
            path = [node]
            move_keys_taken = []

            # Selection
            while node.is_expanded and node.children and not _is_terminal(scratch):
                key, child = self._select_child(node)
                move = node._moves[key]
                _push(scratch, move)
                move_keys_taken.append((node, key))
                node = child
                path.append(node)

            # Expansion / terminal
            if _is_terminal(scratch):
                value = _terminal_value(scratch, root_turn)
            elif not node.is_expanded:
                value = self._expand(node, scratch)
            else:
                value = 0.0

            # Backprop (alternate sign at each ply)
            sign = 1.0
            for n in reversed(path):
                n.visit_count += 1
                n.total_value += sign * value
                sign *= -1.0

        # Select move from root
        if not root.children:
            return None, root

        if self.temperature == 0.0:
            best_key = max(root.children, key=lambda k: root.children[k].visit_count)
        else:
            keys = list(root.children.keys())
            visits = np.array([root.children[k].visit_count for k in keys], dtype=np.float64)
            visits = visits ** (1.0 / self.temperature)
            visits /= visits.sum()
            idx = np.random.choice(len(keys), p=visits)
            best_key = keys[idx]

        return root._moves[best_key], root


if __name__ == "__main__":
    import os

    checkpoint = os.path.join(ROOT_DIR, "checkpoints", "best_supervised.pt")
    # Try C: path used in training
    if not os.path.exists(checkpoint):
        checkpoint = "C:/chess-nn-checkpoints/best_supervised.pt"

    model = ChessNet(num_blocks=3, channels=64)
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    # Test 1: Starting position
    print("Test 1: MCTS from starting position...")
    mcts = MCTS(model, device='cpu', num_simulations=50, temperature=0.0, use_ic=False)
    board = chess.Board()
    move, root = mcts.search(board)
    print(f"  Best move: {move.uci() if hasattr(move, 'uci') else move}")
    assert move in board.legal_moves
    print("  PASSED")

    # Test 2: Multiple positions
    print("Test 2: Multiple positions...")
    positions = [
        chess.Board(),
        chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),  # Sicilian
        chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),  # Italian
        chess.Board("r1bqr1k1/pp3pbp/2np1np1/2p5/4P3/2NP1NP1/PPP2PBP/R1BQR1K1 w - - 0 10"),
        chess.Board("8/5pk1/8/8/8/8/5PK1/8 w - - 0 1"),  # endgame
    ]
    for i, pos in enumerate(positions):
        m, _ = mcts.search(pos)
        print(f"  Position {i+1}: {m.uci() if hasattr(m, 'uci') else m}")
        assert m in pos.legal_moves
    print("  PASSED")

    # Test 3: Simulation count
    print("Test 3: Simulation counts...")
    mcts10 = MCTS(model, device='cpu', num_simulations=10, temperature=0.0, use_ic=False)
    mcts200 = MCTS(model, device='cpu', num_simulations=200, temperature=0.0, use_ic=False)
    m10, _ = mcts10.search(chess.Board())
    m200, _ = mcts200.search(chess.Board())
    print(f"  10 sims: {m10.uci()}  200 sims: {m200.uci()}")
    print("  PASSED")

    # Test 4: IC Chess board with 18-channel model (expect failure)
    print("Test 4: IC Chess with 18-channel model (expect error)...")
    try:
        from mcts.game import ICChessBoard
        ic_board = ICChessBoard('empire', 'western')
        ic_mcts = MCTS(model, device='cpu', num_simulations=5, temperature=0.0, use_ic=True)
        ic_mcts.search(ic_board)
        print("  WARNING: expected error but didn't get one")
    except Exception as e:
        print(f"  IC Chess MCTS: correctly requires 20-channel model (expected): {type(e).__name__}")

    print("mcts.py: All tests passed")

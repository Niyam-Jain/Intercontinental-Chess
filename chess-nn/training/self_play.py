import argparse
import math
import pathlib
import sys
import time

import chess
import numpy as np
import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.decoder import POLICY_SIZE
from model.encoder_ic import IC_NUM_PLANES, encode_ic_board
from model.network import ChessNet
from mcts.game import ARMIES, ICChessBoard
from mcts.mcts import MCTS, _move_key, _move_to_policy_index


def load_ic_model(checkpoint_path: str, num_blocks: int, channels: int,
                  device: torch.device) -> ChessNet:
    """Load a pre-trained 18-plane model into a 20-plane ChessNet."""
    ic_model = ChessNet(num_blocks=num_blocks, channels=channels, input_planes=IC_NUM_PLANES)
    pretrained = torch.load(checkpoint_path, map_location=device)

    ic_state = ic_model.state_dict()
    first_conv_key = 'input_block.0.weight'

    for key in pretrained:
        if key == first_conv_key:
            ic_state[key][:, :18, :, :] = pretrained[key]
            # Planes 18-19 keep Kaiming random init (already set by default)
        else:
            if key in ic_state:
                ic_state[key] = pretrained[key]

    ic_model.load_state_dict(ic_state)
    ic_model.eval()
    return ic_model


def select_move_from_root(root, temperature: float):
    """Select a move from root node's children visit counts."""
    if not root.children:
        return None
    if temperature == 0.0:
        best_key = max(root.children, key=lambda k: root.children[k].visit_count)
    else:
        keys = list(root.children.keys())
        visits = np.array([root.children[k].visit_count for k in keys], dtype=np.float64)
        visits = visits ** (1.0 / temperature)
        visits /= visits.sum()
        idx = np.random.choice(len(keys), p=visits)
        best_key = keys[idx]
    return root._moves[best_key]


def play_game(ic_model, device, args, white_army: str, black_army: str) -> tuple[list, str]:
    """Play one self-play game. Returns (examples, result_str)."""
    board = ICChessBoard(white_army, black_army)
    mcts = MCTS(ic_model, device=device, num_simulations=args.simulations,
                temperature=1.0, use_ic=True)
    examples = []
    move_count = 0

    while not board.is_game_over and move_count < 300:
        move, root = mcts.search(board)
        if move is None:
            break

        # Build policy distribution from visit counts
        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits > 0:
            for key, child in root.children.items():
                move_obj = root._moves[key]
                idx = _move_to_policy_index(move_obj)
                if 0 <= idx < POLICY_SIZE:
                    policy_target[idx] += child.visit_count / total_visits

        examples.append({
            'features': encode_ic_board(board),
            'policy': policy_target,
            'turn': board.turn,
        })

        board.push(move)
        move_count += 1

    result = board.result
    if result == '*':
        result = '1/2-1/2'  # treat unfinished as draw

    for ex in examples:
        if result == '1-0':
            ex['value'] = 1.0 if ex['turn'] == chess.WHITE else -1.0
        elif result == '0-1':
            ex['value'] = -1.0 if ex['turn'] == chess.WHITE else 1.0
        else:
            ex['value'] = 0.0

    return examples, result


def parse_args():
    p = argparse.ArgumentParser(description="IC Chess self-play data generation")
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--num-blocks', type=int, default=3)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--num-games', type=int, default=10)
    p.add_argument('--simulations', type=int, default=50)
    p.add_argument('--output', default='data/self_play/')
    p.add_argument('--white-army', default='empire')
    p.add_argument('--black-army', default='western')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ic_model = load_ic_model(args.checkpoint, args.num_blocks, args.channels, device)
    ic_model.to(device)
    print(f"Loaded IC model ({IC_NUM_PLANES} input planes)")

    # Determine matchups
    all_matchups = []
    wa_list = ARMIES if args.white_army == 'all' else [args.white_army]
    ba_list = ARMIES if args.black_army == 'all' else [args.black_army]
    for wa in wa_list:
        for ba in ba_list:
            all_matchups.append((wa, ba))

    games_per_matchup = math.ceil(args.num_games / len(all_matchups))
    total_games = games_per_matchup * len(all_matchups)

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_features, all_policies, all_values = [], [], []
    wins_white = wins_black = draws = 0
    total_moves = 0
    game_num = 0

    for wa, ba in all_matchups:
        for _ in range(games_per_matchup):
            game_num += 1
            t0 = time.time()
            examples, result = play_game(ic_model, device, args, wa, ba)
            elapsed = time.time() - t0

            if result == '1-0':
                wins_white += 1
            elif result == '0-1':
                wins_black += 1
            else:
                draws += 1

            total_moves += len(examples)
            for ex in examples:
                all_features.append(ex['features'])
                all_policies.append(ex['policy'])
                all_values.append(ex['value'])

            print(f"Game {game_num}/{total_games} [{wa} vs {ba}]: "
                  f"{len(examples)} moves, result={result}, {elapsed:.1f}s")

    # Save
    feat_arr = np.stack(all_features, axis=0).astype(np.float32)
    pol_arr = np.stack(all_policies, axis=0).astype(np.float32)
    val_arr = np.array(all_values, dtype=np.float32)

    np.save(out_dir / 'sp_batch0_features.npy', feat_arr)
    np.save(out_dir / 'sp_batch0_policies.npy', pol_arr)
    np.save(out_dir / 'sp_batch0_values.npy', val_arr)

    print(f"\nSelf-play complete:")
    print(f"  Games: {total_games}")
    print(f"  Total positions: {len(all_values)}")
    print(f"  White wins: {wins_white}, Black wins: {wins_black}, Draws: {draws}")
    avg = total_moves / total_games if total_games else 0
    print(f"  Avg game length: {avg:.1f} moves")
    print(f"  Data saved to: {out_dir}")
    print(f"  Shapes — features: {feat_arr.shape}, policies: {pol_arr.shape}, values: {val_arr.shape}")


if __name__ == '__main__':
    main()

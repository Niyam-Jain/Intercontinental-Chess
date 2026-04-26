import argparse
import pathlib
import sys

import chess
import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.encoder_ic import IC_NUM_PLANES
from model.network import ChessNet
from mcts.game import ICChessBoard
from mcts.mcts import MCTS


def load_model_for_eval(checkpoint_path, num_blocks, channels, input_planes, device):
    """Load a checkpoint for IC Chess evaluation.

    input_planes=18: expand 18-plane supervised checkpoint to 20-plane IC model
                     (planes 18-19 get Kaiming random init).
    input_planes=20: load native 20-plane IC checkpoint directly.
    """
    model = ChessNet(num_blocks=num_blocks, channels=channels, input_planes=IC_NUM_PLANES)
    pretrained = torch.load(checkpoint_path, map_location=device)

    if input_planes == 18:
        state = model.state_dict()
        first_conv_key = 'input_block.0.weight'
        for key, val in pretrained.items():
            if key == first_conv_key:
                state[key][:, :18, :, :] = val
            elif key in state and val.shape == state[key].shape:
                state[key] = val
        model.load_state_dict(state)
    else:
        model.load_state_dict(pretrained)

    model.eval()
    return model.to(device)


def play_eval_game(mcts_white: MCTS, mcts_black: MCTS,
                   white_army: str, black_army: str) -> str:
    board = ICChessBoard(white_army, black_army)
    move_count = 0

    while not board.is_game_over and move_count < 300:
        if board.turn == chess.WHITE:
            move, _ = mcts_white.search(board)
        else:
            move, _ = mcts_black.search(board)

        if move is None:
            break
        board.push(move)
        move_count += 1

    result = board.result
    return result if result != '*' else '1/2-1/2'


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate two IC Chess models against each other")
    p.add_argument('--model-a', required=True)
    p.add_argument('--model-b', required=True)
    p.add_argument('--num-blocks', type=int, default=3)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--input-planes-a', type=int, default=20,
                   help='Input planes for model A: 20=native IC, 18=supervised (auto-expanded)')
    p.add_argument('--input-planes-b', type=int, default=18,
                   help='Input planes for model B: 20=native IC, 18=supervised (auto-expanded)')
    p.add_argument('--num-games', type=int, default=10)
    p.add_argument('--simulations', type=int, default=50)
    p.add_argument('--white-army', default='western')
    p.add_argument('--black-army', default='western')
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--use-ic', action='store_true', default=True,
                   help='Use IC Chess boards (always true; kept for CLI compatibility)')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')

    model_a = load_model_for_eval(args.model_a, args.num_blocks, args.channels,
                                  args.input_planes_a, device)
    model_b = load_model_for_eval(args.model_b, args.num_blocks, args.channels,
                                  args.input_planes_b, device)
    print(f"Loaded Model A ({args.input_planes_a}-plane source): {args.model_a}")
    print(f"Loaded Model B ({args.input_planes_b}-plane source): {args.model_b}")

    mcts_a = MCTS(model_a, device=device, num_simulations=args.simulations,
                  temperature=args.temperature, use_ic=True)
    mcts_b = MCTS(model_b, device=device, num_simulations=args.simulations,
                  temperature=args.temperature, use_ic=True)

    a_wins = b_wins = draws = 0

    for i in range(args.num_games):
        if i % 2 == 0:
            result = play_eval_game(mcts_a, mcts_b, args.white_army, args.black_army)
            if result == '1-0':
                a_wins += 1
            elif result == '0-1':
                b_wins += 1
            else:
                draws += 1
        else:
            result = play_eval_game(mcts_b, mcts_a, args.white_army, args.black_army)
            if result == '1-0':
                b_wins += 1
            elif result == '0-1':
                a_wins += 1
            else:
                draws += 1

        print(f"  Game {i+1}/{args.num_games}: {result}  "
              f"(A {a_wins}W/{b_wins}L/{draws}D so far)")

    total = a_wins + b_wins + draws
    a_pct = 100.0 * (a_wins + 0.5 * draws) / max(total, 1)
    b_pct = 100.0 * (b_wins + 0.5 * draws) / max(total, 1)

    print(f"\nEvaluation: Model A vs Model B ({args.white_army} vs {args.black_army})")
    print(f"  Games: {total}")
    print(f"  Model A (IC fine-tuned): {a_wins}W / {b_wins}L / {draws}D  ({a_pct:.1f}%)")
    print(f"  Model B (pre-trained):   {b_wins}W / {a_wins}L / {draws}D  ({b_pct:.1f}%)")


if __name__ == '__main__':
    main()

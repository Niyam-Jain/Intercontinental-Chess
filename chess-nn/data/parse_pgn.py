import argparse
import pathlib
import random
import sys
from typing import Optional

import chess
import chess.pgn
import numpy as np


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.decoder import POLICY_SIZE, index_to_move, move_to_index
from model.encoder import encode_board


def local_storage_root() -> pathlib.Path:
    return pathlib.Path.home() / "chess-nn-data"


PIECE_SYMBOLS_BY_PLANE = {
    0: "P",
    1: "N",
    2: "B",
    3: "R",
    4: "Q",
    5: "K",
    6: "p",
    7: "n",
    8: "b",
    9: "r",
    10: "q",
    11: "k",
}


def should_skip_game(game: chess.pgn.Game) -> bool:
    result = game.headers.get("Result", "*")
    if result == "*":
        return True
    white_title = (game.headers.get("WhiteTitle") or "").strip().upper()
    black_title = (game.headers.get("BlackTitle") or "").strip().upper()
    if white_title == "BOT" or black_title == "BOT":
        return True
    move_count = sum(1 for _ in game.mainline_moves())
    if move_count < 20:
        return True
    return False


def game_outcome_from_result(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    if result == "1/2-1/2":
        return 0.0
    raise ValueError(f"Unsupported PGN result: {result}")


def save_shard(base_name: str, shard_idx: int, output_dir: pathlib.Path, features: list[np.ndarray], policies: list[int], values: list[float]) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    f_arr = np.asarray(features, dtype=np.float32)
    p_arr = np.asarray(policies, dtype=np.int64)
    v_arr = np.asarray(values, dtype=np.float32)
    m = len(f_arr)

    assert f_arr.shape == (m, 18, 8, 8)
    assert np.all((p_arr >= 0) & (p_arr < POLICY_SIZE))
    assert np.all(np.isin(v_arr, np.array([-1.0, 0.0, 1.0], dtype=np.float32)))

    feat_path = output_dir / f"{base_name}_shard{shard_idx}_features.npy"
    pol_path = output_dir / f"{base_name}_shard{shard_idx}_policies.npy"
    val_path = output_dir / f"{base_name}_shard{shard_idx}_values.npy"
    np.save(feat_path, f_arr)
    np.save(pol_path, p_arr)
    np.save(val_path, v_arr)
    print(f"Saved shard {shard_idx}: {m} positions")
    print(f"  {feat_path.name}")
    print(f"  {pol_path.name}")
    print(f"  {val_path.name}")
    return feat_path, pol_path, val_path


def features_to_board(features: np.ndarray) -> Optional[chess.Board]:
    board = chess.Board.empty()
    try:
        for plane, symbol in PIECE_SYMBOLS_BY_PLANE.items():
            squares = np.argwhere(features[plane] == 1.0)
            for row, col in squares:
                rank = 7 - int(row)
                file_idx = int(col)
                sq = chess.square(file_idx, rank)
                piece = chess.Piece.from_symbol(symbol)
                board.set_piece_at(sq, piece)

        board.turn = bool(features[12, 0, 0] == 1.0)

        rights = ""
        if features[13, 0, 0] == 1.0:
            rights += "K"
        if features[14, 0, 0] == 1.0:
            rights += "Q"
        if features[15, 0, 0] == 1.0:
            rights += "k"
        if features[16, 0, 0] == 1.0:
            rights += "q"
        board.set_castling_fen(rights if rights else "-")

        ep_squares = np.argwhere(features[17] == 1.0)
        if len(ep_squares) == 1:
            row, col = ep_squares[0]
            rank = 7 - int(row)
            file_idx = int(col)
            board.ep_square = chess.square(file_idx, rank)
        else:
            board.ep_square = None

        board.halfmove_clock = 0
        board.fullmove_number = 1
        if not board.is_valid():
            return None
        return board
    except Exception:
        return None


def verify_roundtrip(processed_dir: pathlib.Path, num_samples: int = 10) -> None:
    feat_files = sorted(processed_dir.glob("*_features.npy"))
    if not feat_files:
        print("No feature shards found for round-trip verification.")
        return

    shard_data = []
    total = 0
    for ff in feat_files:
        pf = pathlib.Path(str(ff).replace("_features.npy", "_policies.npy"))
        vf = pathlib.Path(str(ff).replace("_features.npy", "_values.npy"))
        features = np.load(ff, mmap_mode="r")
        policies = np.load(pf, mmap_mode="r")
        values = np.load(vf, mmap_mode="r")
        shard_data.append((features, policies, values))
        total += len(features)

    actual_samples = min(num_samples, total)
    samples = []
    for _ in range(actual_samples):
        global_idx = random.randrange(total)
        running = 0
        chosen = None
        for features, policies, values in shard_data:
            if global_idx < running + len(features):
                local_idx = global_idx - running
                chosen = (features[local_idx], int(policies[local_idx]), float(values[local_idx]))
                break
            running += len(features)
        if chosen is not None:
            samples.append(chosen)

    print("\nRound-trip verification samples:")
    for i, (features, policy, value) in enumerate(samples, start=1):
        board = features_to_board(np.asarray(features))
        if board is None:
            print(f"[{i}] Invalid board reconstruction | policy={policy} value={value}")
            continue
        move = index_to_move(policy, board)
        move_uci = move.uci() if move is not None else "None(illegal)"
        print(f"[{i}] FEN: {board.fen()}")
        print(f"    Move: {move_uci}")
        print(f"    Value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PGN into sharded numpy arrays.")
    parser.add_argument("--input", required=True, help="Path to input .pgn file")
    parser.add_argument("--output", required=True, help="Output directory for processed shards")
    parser.add_argument("--max-games", type=int, default=None, help="Max number of valid games to parse")
    parser.add_argument("--shard-size", type=int, default=2_000_000, help="Max positions per shard")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    if input_path.exists() and input_path.stat().st_size < 2048:
        pointer = input_path.read_text(encoding="utf-8").strip()
        if pointer.lower().endswith(".pgn") and pathlib.Path(pointer).exists():
            print(f"Resolved PGN pointer {input_path} -> {pointer}")
            input_path = pathlib.Path(pointer)

    output_dir = pathlib.Path(args.output)
    repo_processed_default = ROOT_DIR / "data" / "processed"
    if output_dir.resolve() == repo_processed_default.resolve():
        output_dir = local_storage_root() / "processed"
        print(f"Redirecting output to local storage: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_path.stem
    max_positions_by_bytes = int(1_500_000_000 // (18 * 8 * 8 * 4))
    effective_shard_size = min(args.shard_size, max_positions_by_bytes)
    if effective_shard_size != args.shard_size:
        print(
            f"Adjusted shard size from {args.shard_size} to {effective_shard_size} "
            f"to keep feature shards under ~1.5GB."
        )

    features_buffer: list[np.ndarray] = []
    policies_buffer: list[int] = []
    values_buffer: list[float] = []
    shard_idx = 0
    total_games = 0
    total_positions = 0
    saved_shards = 0

    with input_path.open("r", encoding="utf-8") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            if should_skip_game(game):
                continue

            result = game.headers.get("Result", "*")
            outcome = game_outcome_from_result(result)
            board = game.board()

            for move in game.mainline_moves():
                value = outcome if board.turn == chess.WHITE else -outcome
                features_buffer.append(encode_board(board))
                policies_buffer.append(move_to_index(move))
                values_buffer.append(float(value))
                board.push(move)
                total_positions += 1

                if len(features_buffer) >= effective_shard_size:
                    save_shard(base_name, shard_idx, output_dir, features_buffer, policies_buffer, values_buffer)
                    features_buffer.clear()
                    policies_buffer.clear()
                    values_buffer.clear()
                    shard_idx += 1
                    saved_shards += 1

            total_games += 1
            if total_games % 10_000 == 0:
                avg = total_positions / total_games if total_games else 0.0
                print(f"Parsed {total_games} games | {total_positions} positions | {avg:.1f} positions/game avg")

            if args.max_games is not None and total_games >= args.max_games:
                break

    if features_buffer:
        save_shard(base_name, shard_idx, output_dir, features_buffer, policies_buffer, values_buffer)
        saved_shards += 1

    print("\nParse complete.")
    print(f"Games parsed: {total_games}")
    print(f"Positions extracted: {total_positions}")
    print(f"Shards created: {saved_shards}")

    verify_roundtrip(output_dir, num_samples=10)


if __name__ == "__main__":
    main()

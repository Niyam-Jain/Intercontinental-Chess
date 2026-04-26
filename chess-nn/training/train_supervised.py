import argparse
import pathlib
import random
import sys
import time

import chess
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.dataset import ChessDataset
from model.decoder import POLICY_SIZE, get_legal_move_mask, index_to_move
from model.encoder import encode_board
from model.network import ChessNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised training for ChessNet.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using device: cuda ({name})")
        return torch.device("cuda")
    if req == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
    print("Using device: cpu")
    return torch.device("cpu")


def run_inference_sanity(model: ChessNet, device: torch.device, save_path: pathlib.Path) -> None:
    state_dict = torch.load(save_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    board = chess.Board()
    features = encode_board(board)
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = model(tensor)

    legal_mask = get_legal_move_mask(board)
    masked_logits = policy_logits.squeeze(0).detach().cpu()
    full_mask = torch.full((POLICY_SIZE,), float("-inf"))
    for idx in legal_mask:
        full_mask[idx] = 0.0
    masked_logits = masked_logits + full_mask

    probs = torch.softmax(masked_logits, dim=0)
    top5 = probs.topk(5)
    top5_indices = top5.indices.tolist()
    top5_probs = top5.values.tolist()

    print("\nStarting position — Top 5 predicted moves:")
    for idx, prob in zip(top5_indices, top5_probs):
        move = index_to_move(idx, board)
        move_str = move.uci() if move is not None else "None"
        print(f"  {move_str} ({prob * 100:.1f}%)")
    print(f"\nPosition evaluation: {value.item():.3f} (positive = White advantage)")


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "best_supervised.pt"

    dataset = ChessDataset(args.data_dir)
    print(f"Dataset: {len(dataset):,} positions")

    val_size = len(dataset) // 20
    train_size = len(dataset) - val_size
    split_gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=split_gen)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = ChessNet(num_blocks=args.num_blocks, channels=args.channels).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters")

    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Resumed from checkpoint: {args.resume}")

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    policy_loss_fn = CrossEntropyLoss()
    value_loss_fn = MSELoss()

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_p_loss = 0.0
        train_v_loss = 0.0
        train_batches = 0

        for features, policy_target, value_target in train_loader:
            features = features.to(device, non_blocking=True)
            policy_target = policy_target.to(device, non_blocking=True)
            value_target = value_target.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            policy_logits, value_pred = model(features)
            p_loss = policy_loss_fn(policy_logits, policy_target)
            v_loss = value_loss_fn(value_pred, value_target)
            loss = p_loss + v_loss
            loss.backward()
            optimizer.step()

            train_p_loss += p_loss.item()
            train_v_loss += v_loss.item()
            train_batches += 1

        avg_train_p = train_p_loss / max(1, train_batches)
        avg_train_v = train_v_loss / max(1, train_batches)

        model.eval()
        val_p_loss = 0.0
        val_v_loss = 0.0
        val_batches = 0
        val_total = 0
        val_top1 = 0
        val_top5 = 0

        with torch.no_grad():
            for features, policy_target, value_target in val_loader:
                features = features.to(device, non_blocking=True)
                policy_target = policy_target.to(device, non_blocking=True)
                value_target = value_target.to(device, non_blocking=True).unsqueeze(1)

                policy_logits, value_pred = model(features)
                p_loss = policy_loss_fn(policy_logits, policy_target)
                v_loss = value_loss_fn(value_pred, value_target)

                val_p_loss += p_loss.item()
                val_v_loss += v_loss.item()
                val_batches += 1

                preds = policy_logits.argmax(dim=1)
                val_top1 += (preds == policy_target).sum().item()

                top5 = policy_logits.topk(5, dim=1).indices
                val_top5 += (top5 == policy_target.unsqueeze(1)).any(dim=1).sum().item()
                val_total += policy_target.size(0)

        avg_val_p = val_p_loss / max(1, val_batches)
        avg_val_v = val_v_loss / max(1, val_batches)
        avg_val_total = avg_val_p + avg_val_v
        top1_pct = (100.0 * val_top1 / max(1, val_total))
        top5_pct = (100.0 * val_top5 / max(1, val_total))

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - t0

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train P-Loss: {avg_train_p:.4f}  V-Loss: {avg_train_v:.4f} | "
            f"Val P-Loss: {avg_val_p:.4f}  V-Loss: {avg_val_v:.4f} | "
            f"Top-1 Acc: {top1_pct:.1f}%  Top-5 Acc: {top5_pct:.1f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_seconds:.0f}s"
        )

        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (val loss: {best_val_loss:.4f})")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    run_inference_sanity(model, device, save_path)


if __name__ == "__main__":
    train(parse_args())

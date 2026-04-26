import argparse
import os
import pathlib
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.encoder_ic import IC_NUM_PLANES, encode_ic_board
from model.network import ChessNet
from mcts.game import ICChessBoard


def policy_loss_soft(logits, target_distribution):
    """Cross-entropy loss with soft (MCTS visit-count) targets."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_distribution * log_probs).sum(dim=1).mean()


class SelfPlayDataset(Dataset):
    def __init__(self, data_dir):
        features_files = sorted(glob(os.path.join(data_dir, '*_features.npy')))
        policies_files = sorted(glob(os.path.join(data_dir, '*_policies.npy')))
        values_files = sorted(glob(os.path.join(data_dir, '*_values.npy')))

        if not features_files:
            raise FileNotFoundError(f"No *_features.npy files found in {data_dir}")

        self.features = np.concatenate([np.load(f) for f in features_files])
        self.policies = np.concatenate([np.load(f) for f in policies_files])
        self.values = np.concatenate([np.load(f) for f in values_files])
        print(f"  Loaded {len(self.features):,} positions from {len(features_files)} batch(es)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),               # (20, 8, 8)
            torch.from_numpy(self.policies[idx]),               # (4288,) distribution
            torch.tensor(self.values[idx], dtype=torch.float32),  # scalar
        )


def load_pretrained_to_ic(checkpoint_path, num_blocks, channels, device):
    """Load an 18-plane supervised checkpoint into a 20-plane IC Chess model.

    Planes 0-17 are copied from the checkpoint; planes 18-19 keep Kaiming init.
    """
    model = ChessNet(num_blocks=num_blocks, channels=channels, input_planes=IC_NUM_PLANES)
    pretrained = torch.load(checkpoint_path, map_location=device)

    new_state = model.state_dict()
    first_conv_key = 'input_block.0.weight'

    for key, val in pretrained.items():
        if key == first_conv_key:
            new_state[key][:, :18, :, :] = val
        elif key in new_state and val.shape == new_state[key].shape:
            new_state[key] = val
        else:
            print(f"  Skipping incompatible key: {key}")

    model.load_state_dict(new_state)
    return model.to(device)


def train(args):
    device = torch.device(args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading self-play data from: {args.self_play_dir}")
    dataset = SelfPlayDataset(args.self_play_dir)
    print(f"Self-play dataset: {len(dataset):,} positions")

    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = load_pretrained_to_ic(args.pretrained, args.num_blocks, args.channels, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters (20-channel IC Chess)")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    value_loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'best_ic_finetuned.pt')

    for epoch in range(args.epochs):
        model.train()
        train_p_loss = train_v_loss = 0.0
        num_batches = 0

        for features, policy_target, value_target in train_loader:
            features = features.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device).unsqueeze(1)

            policy_logits, value_pred = model(features)
            p_loss = policy_loss_soft(policy_logits, policy_target)
            v_loss = value_loss_fn(value_pred, value_target)

            optimizer.zero_grad()
            (p_loss + v_loss).backward()
            optimizer.step()

            train_p_loss += p_loss.item()
            train_v_loss += v_loss.item()
            num_batches += 1

        model.eval()
        val_p_loss = val_v_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for features, policy_target, value_target in val_loader:
                features = features.to(device)
                policy_target = policy_target.to(device)
                value_target = value_target.to(device).unsqueeze(1)

                policy_logits, value_pred = model(features)
                val_p_loss += policy_loss_soft(policy_logits, policy_target).item()
                val_v_loss += value_loss_fn(value_pred, value_target).item()
                val_batches += 1

        avg_val = (val_p_loss + val_v_loss) / max(val_batches, 1)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train P: {train_p_loss/num_batches:.4f}  V: {train_v_loss/num_batches:.4f} | "
            f"Val P: {val_p_loss/val_batches:.4f}  V: {val_v_loss/val_batches:.4f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best IC model (val loss: {best_val_loss:.4f})")

    print(f"\nFine-tuning complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")

    # Inference sanity check on IC Chess starting position
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    ic_board = ICChessBoard('empire', 'western')
    features_np = encode_ic_board(ic_board)
    tensor = torch.from_numpy(features_np).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = model(tensor)

    probs = F.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
    top5_idx = np.argsort(probs)[-5:][::-1]

    print(f"\nIC Chess (Empire vs Western) starting position:")
    print(f"  Value: {value.item():.3f}")
    print(f"  Top 5 policy indices: {top5_idx.tolist()}")
    print(f"  Top 5 probabilities:  {[round(float(p), 4) for p in probs[top5_idx]]}")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ChessNet on IC Chess self-play data")
    p.add_argument('--pretrained', required=True, help='18-plane supervised checkpoint')
    p.add_argument('--self-play-dir', required=True, help='Directory with sp_batch*_*.npy files')
    p.add_argument('--save-dir', default='checkpoints')
    p.add_argument('--num-blocks', type=int, default=3)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)

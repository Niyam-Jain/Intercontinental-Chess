import pathlib
import sys

import torch
import torch.nn as nn

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.decoder import POLICY_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ChessNet(nn.Module):
    def __init__(self, num_blocks: int = 3, channels: int = 64, input_planes: int = 18,
                 policy_channels: int = 2):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=input_planes, out_channels=channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.res_tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(policy_channels * 8 * 8, POLICY_SIZE),
        )

        self.value_head_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head_fc = nn.Sequential(
            nn.Linear(8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.input_block(x)
        out = self.res_tower(out)
        policy_logits = self.policy_head(out)
        value = self.value_head_conv(out)
        value = self.value_head_fc(value)
        return policy_logits, value


if __name__ == "__main__":
    torch.manual_seed(0)

    # Smoke model — policy_channels=2 (fixed)
    smoke_model = ChessNet(num_blocks=3, channels=64, policy_channels=2)
    smoke_params = sum(p.numel() for p in smoke_model.parameters())
    print(f"Smoke model params (3b/64c/pc=2):  {smoke_params:,}")
    x = torch.randn(4, 18, 8, 8)
    p_logits, v = smoke_model(x)
    assert tuple(p_logits.shape) == (4, POLICY_SIZE)
    assert tuple(v.shape) == (4, 1)
    assert torch.all(v <= 1.0).item() and torch.all(v >= -1.0).item()

    # Full production model — policy_channels=2
    full_model = ChessNet(num_blocks=10, channels=128, policy_channels=2)
    full_params = sum(p.numel() for p in full_model.parameters())
    print(f"Full model params  (10b/128c/pc=2): {full_params:,}")

    # IC Chess smoke model — 20 input planes, policy_channels=2
    ic_model = ChessNet(num_blocks=3, channels=64, input_planes=20, policy_channels=2)
    ic_params = sum(p.numel() for p in ic_model.parameters())
    print(f"IC smoke model     (3b/64c/20p/pc=2): {ic_params:,}")
    x_ic = torch.randn(4, 20, 8, 8)
    p_ic, v_ic = ic_model(x_ic)
    assert tuple(p_ic.shape) == (4, POLICY_SIZE)
    assert tuple(v_ic.shape) == (4, 1)

    print("network.py: All tests passed")

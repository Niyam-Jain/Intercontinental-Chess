# Intercontinental Chess

A browser-based chess variant that pits three distinct armies against each other — Western, Empire (Dragon Cannon), and African — backed by a custom-trained neural network AI using an AlphaZero-style architecture.

---

## What Is Intercontinental Chess?

Standard chess uses identical armies. Intercontinental Chess gives each side a choice of three armies with unique pieces and mechanics:

| Army | Style | Special Ability |
|---|---|---|
| **Western** | Standard chess | Classic piece movement |
| **Empire (Dragon Cannon)** | Xiangqi-inspired | Cannon pieces that jump over a screen to capture |
| **African** | Animal-themed | Pieces with limited "charge" jump captures |

You can mix and match — play Western vs Empire, African vs African, or any combination.

---

## Features

- **Three armies** with distinct pieces and movement rules
- **Five difficulty levels**: Beginner, Easy, Medium, Hard, Master
- **Two AI engines**: Neural network (MCTS) and classical minimax with alpha-beta pruning
- **Full chess rules**: Castling, en passant, pawn promotion, check/stalemate detection
- **Game review**: Replay any game move-by-move with board highlighting
- **Save / Share**: Export games as JSON or share via URL
- **Sound effects** for moves, captures, check, and game end
- **No installation required** to play — pure HTML/JS with CDN dependencies

---

## How to Run

The game must be served over HTTP (not opened as a local file) because the neural network AI runs in a [Web Worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API), which browsers block on `file://` URLs.

**Clone the repo:**
```bash
git clone https://github.com/Niyam-Jain/Intercontinental-Chess.git
cd Intercontinental-Chess
```

**Start a local server** (pick any one):

```bash
# Python (pre-installed on most systems)
python -m http.server 8000

# Node.js
npx serve .
```

Then open **http://localhost:8000** in your browser.

> If you skip the HTTP server and open `index.html` directly, the game still works but falls back to the minimax engine — the neural network AI will not load.

---

## Project Structure

```
Intercontinental-Chess/
├── index.html                  # Entire game UI, rules engine, and minimax AI
└── chess-nn/
    ├── browser/
    │   ├── ai_worker.js        # Web Worker: loads ONNX model, runs MCTS search
    │   ├── encoder.js          # Encodes board state into 20-plane tensor
    │   └── mcts.js             # Monte Carlo Tree Search (JavaScript)
    ├── model/
    │   ├── network.py          # ResNet tower + dual policy/value heads
    │   ├── encoder.py          # Standard chess board encoder (18 planes)
    │   ├── encoder_ic.py       # IC Chess encoder (18 + 2 IC-specific planes)
    │   └── decoder.py          # Maps policy index ↔ move (4288 possible moves)
    ├── training/
    │   ├── train_supervised.py # Supervised learning from Lichess PGN data
    │   ├── train_rl.py         # Fine-tuning on IC Chess self-play data
    │   ├── self_play.py        # Generates IC Chess training games via MCTS
    │   └── evaluate.py         # Pits two model checkpoints against each other
    ├── data/
    │   ├── download.py         # Downloads Lichess elite PGN datasets
    │   ├── parse_pgn.py        # Converts PGN games → feature tensors
    │   └── dataset.py          # PyTorch Dataset with mmap sharding
    ├── export/
    │   ├── to_onnx.py          # Exports PyTorch checkpoint → ONNX model
    │   └── chess_model.onnx    # Exported model used by the browser
    ├── checkpoints/
    │   └── best_supervised.pt  # Best supervised-learning checkpoint
    └── requirements.txt
```

---

## Neural Network Architecture

The AI uses an **AlphaZero-style dual-headed residual network**:

```
Input: 20 × 8 × 8 tensor
  └─ Planes 0–5:   White piece positions (P, N, B, R, Q, K)
  └─ Planes 6–11:  Black piece positions (P, N, B, R, Q, K)
  └─ Planes 12–17: Castling rights, en passant, turn
  └─ Plane 18:     Charge counts (IC-specific jump captures)
  └─ Plane 19:     Army type flags (empire/african)

Residual Tower: 3–10 blocks × [Conv 3×3 → BN → ReLU → Conv 3×3 → BN → Skip]

Policy Head → 4288 logits (64×64 moves + underpromotions)
Value Head  → scalar in [-1, 1]
```

The policy output covers all 64×64 source/destination square pairs (4096) plus 192 underpromotion moves, giving 4288 total.

---

## Training Pipeline

Training happens in two stages:

### Stage 1 — Supervised Learning (standard chess)
1. Download elite Lichess games (`data/download.py`)
2. Parse PGN → board tensors + policy/value targets (`data/parse_pgn.py`)
3. Train the residual network to predict master moves and outcomes (`training/train_supervised.py`)

### Stage 2 — Fine-tuning on IC Chess (reinforcement learning)
1. Load the supervised checkpoint, extend input to 20 planes for IC features
2. Generate IC Chess games via MCTS self-play (`training/self_play.py`)
3. Fine-tune on self-play data with visit-count policy targets (`training/train_rl.py`)
4. Evaluate checkpoints by having them play each other (`training/evaluate.py`)

### Exporting to the Browser
```bash
pip install -r chess-nn/requirements.txt
python chess-nn/export/to_onnx.py \
  --checkpoint chess-nn/checkpoints/best_supervised.pt \
  --output chess-nn/export/chess_model.onnx
```

---

## AI Difficulty Levels

| Level | Engine | Depth / Simulations | Notes |
|---|---|---|---|
| Beginner | Minimax | Depth 1, high randomness | Plays almost randomly |
| Easy | Minimax | Depth 1, some randomness | Light search |
| Medium | Minimax | Depth 3, low randomness | Moderate play |
| Hard | Minimax | Depth 4, no randomness | Strong classical AI |
| Master | Neural Network (MCTS) | 1200 simulations | Strongest — uses NN + MCTS, capped at 15s |

Master difficulty requires the ONNX model to be present. If it fails to load, the game shows a toast notification and falls back to the Hard minimax engine.

---

## Python Requirements (training only)

Playing the game requires no Python. Training and exporting the model requires:

```
torch>=2.0.0
numpy>=1.24.0
python-chess>=1.9.0
requests>=2.28.0
zstandard>=0.21.0
onnx>=1.14.0
onnxruntime>=1.15.0
```

Install with:
```bash
pip install -r chess-nn/requirements.txt
```

---

## Browser Compatibility

Tested on Chrome, Edge, and Firefox. Requires a browser with:
- Web Workers support
- WebAssembly support (for ONNX Runtime)
- An active internet connection (for Tailwind CSS and ONNX Runtime CDN assets)

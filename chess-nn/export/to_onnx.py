import argparse
import os
import pathlib
import sys

import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.network import ChessNet


def export(checkpoint_path, num_blocks, channels, input_planes, output_path):
    model = ChessNet(num_blocks=num_blocks, channels=channels, input_planes=input_planes)
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, input_planes, 8, 8)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['board'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'board':  {0: 'batch'},
            'policy': {0: 'batch'},
            'value':  {0: 'batch'},
        },
        opset_version=17,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size = os.path.getsize(output_path)
    print(f"Exported:          {output_path}")
    print(f"Size:              {file_size / 1024 / 1024:.2f} MB")
    print(f"ONNX validation:   passed")

    import onnxruntime as ort
    import numpy as np
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    result = session.run(None, {'board': dummy.numpy()})
    print(f"Policy shape:      {result[0].shape}")
    print(f"Value shape:       {result[1].shape}")
    print(f"ONNX Runtime test: passed")


def parse_args():
    p = argparse.ArgumentParser(description="Export ChessNet to ONNX")
    p.add_argument('--checkpoint', required=True, help='PyTorch checkpoint (.pt)')
    p.add_argument('--num-blocks', type=int, default=3)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--input-planes', type=int, default=20)
    p.add_argument('--output', default='export/chess_model.onnx')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export(args.checkpoint, args.num_blocks, args.channels, args.input_planes, args.output)

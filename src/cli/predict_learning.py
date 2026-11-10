"""仅凭 J,h 加载冻结神经模型权重预测初始化并用能量驱动 L-BFGS-B 求解。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from learning.inference import load_predictor
from learning.optimization import refine
from records import emit


def main() -> None:
    """部署不加载目标态；输出优化前后角度、能量和实际耗时。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="冻结神经模型权重路径")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--grid-points", type=int, default=128)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    with np.load(args.instance, allow_pickle=False) as archive:
        J, h = archive["J"].astype(np.float64), archive["h"].astype(np.float64)

    predict, checkpoint = load_predictor(args.checkpoint, args.grid_points)
    architecture = checkpoint["architecture"]
    initial, prediction = predict(J, h)
    suffix = {"Flat-Top": "Flat", "Endpoint-Matched": "End",
              "Alternating-Phase": "Alt"}.get(checkpoint["method"], checkpoint["method"])
    if checkpoint["architecture"] == "Echo-Prior":
        method_name = f"Echo-Prior-{suffix}"
    else:
        method_name = checkpoint["method"]
    depth = checkpoint["depth"]

    result = refine(J, h, initial)
    stage1_seconds = prediction.pop("seconds")
    emit(dict(
        instance=str(args.instance), architecture=architecture,
        method=method_name, depth=depth,
        initial_angles=initial.tolist(), angles=result["angles"],
        initial_energy=result["initial_energy"], energy=result["energy"],
        stage1_calls=0, stage2_calls=result["calls"],
        stage1_seconds=stage1_seconds, stage2_seconds=result["seconds"],
        total_seconds=stage1_seconds + result["seconds"],
        status=result["status"], message=result["message"], **prediction,
    ))


if __name__ == "__main__":
    main()

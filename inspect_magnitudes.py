"""Compare parameter and activation magnitudes on fixed expert observations."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch

from train import ExpertData, Policy


class Moments:
    def __init__(self):
        self.n = 0
        self.total = self.absolute = self.squares = 0.0

    def update(self, x):
        self.n += x.numel()
        self.total += x.sum(dtype=torch.float64).item()
        self.absolute += x.abs().sum(dtype=torch.float64).item()
        self.squares += x.square().sum(dtype=torch.float64).item()

    def stats(self):
        return dict(elements=self.n, mean=self.total / self.n,
                    mean_abs=self.absolute / self.n, rms=math.sqrt(self.squares / self.n))


@torch.inference_mode()
def measure(model, observations):
    activations = defaultdict(Moments)
    products = defaultdict(float)
    handles = []
    index = -1

    def block_hook(i):
        def hook(_, args, branch):
            nonlocal index
            index = i
            x, = args
            activations[i, "x"].update(x)
            activations[i, "f(x)"].update(branch)
            products[i] += (x * branch).sum(dtype=torch.float64).item()
            if not model.fusion:
                activations[i, "out"].update(x + branch)
        return hook

    def mixer_hook(_, __, output):
        activations[index, "out"].update(output)

    for i, block in enumerate(model.blocks, 1):
        handles.append(block.register_forward_hook(block_hook(i)))
    for mixer in model.fusion:
        handles.append(mixer.register_forward_hook(mixer_hook))
    try:
        for rgb in observations.split(32):
            model(rgb)
    finally:
        for handle in handles:
            handle.remove()

    for i in range(1, len(model.blocks) + 1):
        x, branch = activations[i, "x"], activations[i, "f(x)"]
        product = products[i]
        scale = math.sqrt(x.squares * branch.squares)
        for key in ("x", "f(x)", "out"):
            stats = activations[i, key].stats()
            if key == "out" and not model.fusion:
                expected = x.squares + branch.squares + 2 * product
                assert math.isclose(expected, activations[i, key].squares, rel_tol=1e-6)
            yield dict(block=i, signal=key, **stats,
                       branch_cosine=product / scale if scale else None)


def checkpoints(root):
    for run, modes in (
        ("coinrun-bc-3seed", ("add", "sum", "concat")),
        ("coinrun-direct-3seed", ("add-deep", "concat-direct")),
    ):
        for mode in modes:
            for groups in ((1, 2, 3, 6) if mode == "concat-direct" else (1,)):
                label = f"{mode}-g{groups}" if mode == "concat-direct" else mode
                for seed in range(3):
                    name = f"{label}-s{seed}"
                    yield label, mode, groups, seed, 0, None
                    yield label, mode, groups, seed, 2000, root / run / name / "model.pt"
                    if run == "coinrun-direct-3seed":
                        yield label, mode, groups, seed, 8000, root / "coinrun-direct-8k" / name / "model.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    torch.set_num_threads(4)
    data = ExpertData(args.data)
    observations, _ = data.batch(data.held[:args.samples], "cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ["model", "seed", "step"]
    moments = ["elements", "mean", "mean_abs", "rms"]
    with ((args.output / "parameters.csv").open("w", newline="") as p,
          (args.output / "activations.csv").open("w", newline="") as a):
        params = csv.DictWriter(p, fields + ["parameter"] + moments)
        activations = csv.DictWriter(a, fields + ["block", "signal"] + moments + ["branch_cosine"])
        params.writeheader()
        activations.writeheader()
        for label, mode, groups, seed, step, path in checkpoints(args.runs):
            torch.manual_seed(seed)
            model = Policy(mode, groups).to(memory_format=torch.channels_last).eval()
            if path:
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            prefix = dict(model=label, seed=seed, step=step)
            for name, value in model.named_parameters():
                stats = Moments()
                stats.update(value.detach())
                params.writerow(dict(**prefix, parameter=name, **stats.stats()))
            for record in measure(model, observations):
                activations.writerow(dict(**prefix, **record))
            print(f"{label} seed={seed} step={step} observations={len(observations)}", flush=True)


if __name__ == "__main__":
    main()

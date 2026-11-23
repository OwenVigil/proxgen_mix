"""Parameter-matched residual fusion pilot on GenDGRL expert trajectories."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Fusion(nn.Module):
    def __init__(self, mode, channels):
        super().__init__()
        if channels % 2:
            raise ValueError("Matching requires an even channel count")
        self.concat = mode == "concat"
        inputs = channels * (2 if self.concat else 1)
        hidden = channels if self.concat else channels * 3 // 2
        self.up = nn.Conv2d(inputs, hidden, 1, bias=False)
        self.down = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, x, branch):
        summed = x + branch
        inputs = torch.cat((x, branch), dim=1) if self.concat else summed
        return summed + self.down(F.gelu(self.up(inputs)))


class Block(nn.Module):
    def __init__(self, channels, first_activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = None
        self.first_activation = nn.Tanh() if first_activation == "tanh" else nn.ReLU()

    def forward(self, x):
        x = self.conv2(F.relu(self.conv1(self.first_activation(x))))
        return self.conv3(F.relu(x)) if self.conv3 is not None else x


class Policy(nn.Module):
    def __init__(self, mode, groups, channels=32, first_activation="relu"):
        super().__init__()
        self.down = nn.ModuleList(
            nn.Conv2d(c, channels, 3, padding=1) for c in (3, channels, channels)
        )
        self.blocks = nn.ModuleList(Block(channels, first_activation) for _ in range(6))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * 64, 256),
                                  nn.ReLU(), nn.Linear(256, 15))
        # Keep the shared backbone and head initialization paired.
        if mode == "add-deep":
            self.blocks[-1].conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        # Construct fusion last so paired seeds start with identical backbone weights.
        if mode == "concat-direct":
            self.fusion = nn.ModuleList(nn.Sequential(
                nn.Conv2d(2 * channels, channels, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1, bias=False),
            ) for _ in range(groups))
        else:
            self.fusion = nn.ModuleList(
                Fusion(mode, channels) for _ in range(groups if mode not in ("add", "add-deep") else 0)
            )
        self.mode = mode

    def forward(self, rgb):
        x = rgb.permute(0, 3, 1, 2).float().div_(255)
        for stage, down in enumerate(self.down):
            x = F.max_pool2d(down(x), 3, stride=2, padding=1)
            for i in range(2 * stage, 2 * stage + 2):
                branch = self.blocks[i](x)
                if self.fusion:
                    mixer = self.fusion[i * len(self.fusion) // len(self.blocks)]
                    if self.mode == "concat-direct":
                        residual = torch.cat((x, branch), dim=1)
                        x = mixer(residual)
                    else:
                        x = mixer(x, branch)
                else:
                    x = x + branch
        return self.head(F.relu(x))


class ExpertData:
    def __init__(self, root):
        size = json.loads((root / "meta.json").read_text())["steps"]
        self.observations = np.memmap(root / "rgb.bin", "uint8", "r", shape=(size, 64, 64, 3))
        self.actions = np.memmap(root / "action.bin", "uint8", "r", shape=(size,))
        episodes = np.load(root / "episodes.npy", allow_pickle=False)
        rng = np.random.default_rng(2024)
        held = np.zeros(len(episodes), dtype=bool)
        for level in np.unique(episodes[:, 0]):
            ids = np.flatnonzero(episodes[:, 0] == level)
            if len(ids) > 1:
                held[rng.permutation(ids)[:max(1, len(ids) // 20)]] = True
        mask = np.repeat(held, episodes[:, 1])
        self.train = np.flatnonzero(~mask)
        self.held = rng.permutation(np.flatnonzero(mask))
        if not len(self.held):
            raise ValueError("No episodes available for a within-level holdout")

    def batch(self, indices, device):
        return (torch.from_numpy(self.observations[indices]).to(device),
                torch.from_numpy(self.actions[indices]).to(device, dtype=torch.long))


@torch.inference_mode()
def heldout_loss(model, data, device, batch_size):
    indices = data.held[:8192]
    total = 0.0
    for offset in range(0, len(indices), batch_size):
        rgb, actions = data.batch(indices[offset:offset + batch_size], device)
        total += F.cross_entropy(model(rgb).float(), actions, reduction="sum").item()
    return total / len(indices)


@torch.inference_mode()
def rollout(model, device, env_name, split, episodes, eval_seed):
    from procgen import ProcgenGym3Env

    start, levels = {"train": (0, 200), "val": (200, 50), "test": (250, 0)}[split]
    num = min(32, episodes)
    # Fixed quotas per environment avoid preferentially scoring short episodes.
    quotas = episodes // num + (np.arange(num) < episodes % num)
    env = ProcgenGym3Env(num=num, env_name=env_name, start_level=start, num_levels=levels,
                         distribution_mode="easy", rand_seed=eval_seed, num_threads=4)
    returns = np.zeros(num, dtype=np.float64)
    counts = np.zeros(num, dtype=np.int32)
    scores = [[] for _ in range(num)]
    try:
        _, obs, _ = env.observe()
        seeds = [int(item["level_seed"]) for item in env.get_info()]
        for _ in range(100_000):
            act = model(torch.from_numpy(obs["rgb"]).to(device)).argmax(dim=1)
            env.act(act.cpu().numpy().astype(np.int32))
            reward, obs, first = env.observe()
            returns += reward
            info = env.get_info() if first.any() else None
            for i in np.flatnonzero(first):
                if counts[i] < quotas[i]:
                    seed = seeds[i]
                    if seed < start or (levels and seed >= start + levels):
                        raise ValueError(f"Level {seed} is outside the {split} split")
                    scores[i].append({"level": seed, "return": float(returns[i])})
                    counts[i] += 1
                seeds[i] = int(info[i]["level_seed"])
                returns[i] = 0
            if np.array_equal(counts, quotas):
                return [episode for lane in scores for episode in lane]
        raise RuntimeError("Evaluation did not finish all scheduled episodes")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode", choices=["add", "add-deep", "sum", "concat", "concat-direct"],
                        required=True)
    parser.add_argument("--groups", type=int, choices=[1, 2, 3, 6], default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--first-activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--env", default="coinrun")
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--eval-seed", type=int, default=777)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--warm-start", type=Path, help="Load weights and reset the optimizer")
    parser.add_argument("--start-step", type=int, default=0,
                        help="Recorded steps in the warm-start checkpoint; --steps is the final step")
    parser.add_argument("--snapshot-every", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.rollouts < 0 or args.steps < 1:
        parser.error("batch size and steps must be positive and rollouts nonnegative")
    if args.eval_split == "test" and args.checkpoint is None:
        parser.error("Test levels may only be used with a frozen checkpoint")
    if (args.start_step < 0 or args.start_step >= args.steps or args.snapshot_every < 0
            or (args.start_step and not args.warm_start) or (args.checkpoint and args.warm_start)):
        parser.error("Invalid warm-start or snapshot configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    rng = np.random.default_rng(args.seed)
    data = ExpertData(args.data)
    model = Policy(args.mode, args.groups, first_activation=args.first_activation).to(
        device, memory_format=torch.channels_last
    )
    result = {"config": vars(args), "params": sum(p.numel() for p in model.parameters())}
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    else:
        if args.warm_start:
            model.load_state_dict(torch.load(args.warm_start, map_location=device, weights_only=True))
            for _ in range(args.start_step):
                rng.choice(data.train, args.batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model.train()
        start = time.perf_counter()
        total_loss = torch.zeros((), device=device)
        window_loss = torch.zeros_like(total_loss)
        window_start, window_step = start, args.start_step
        train_seconds, history = 0.0, []
        for step in range(args.start_step + 1, args.steps + 1):
            rgb, acts = data.batch(rng.choice(data.train, args.batch_size), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
                loss = F.cross_entropy(model(rgb), acts)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
            window_loss += loss.detach()
            if step % 100 == 0 or step == args.steps:
                print(f"step={step} loss={loss.item():.4f}", flush=True)
            snapshot = args.snapshot_every and step % args.snapshot_every == 0 and step < args.steps
            if step % 1000 == 0 or step == args.steps or snapshot:
                count = step - window_step
                ce = (window_loss / count).item()
                if not np.isfinite(ce):
                    raise FloatingPointError(f"Non-finite training loss by step {step}")
                seconds = time.perf_counter() - window_start
                train_seconds += seconds
                window = {"from_step": window_step, "step": step, "train_ce": ce,
                          "seconds": seconds, "samples_per_second": count * args.batch_size / seconds}
                history.append(window)
                print(json.dumps(window), flush=True)
                if snapshot:
                    torch.save(model.state_dict(), args.output / f"model-{step}.pt")
                window_loss.zero_()
                window_start, window_step = time.perf_counter(), step
        result["mean_train_ce"] = (total_loss / (args.steps - args.start_step)).item()
        result["train_seconds"] = train_seconds
        result["train_windows"] = history
        if cuda:
            result["peak_cuda_mib"] = torch.cuda.max_memory_allocated() / 2**20
        torch.save(model.state_dict(), args.output / "model.pt")

    model.eval()
    result["heldout_episode_ce"] = heldout_loss(model, data, device, args.batch_size)
    if args.rollouts:
        scores = rollout(model, device, args.env, args.eval_split, args.rollouts, args.eval_seed)
        result["rollout_mean_return"] = float(np.mean([item["return"] for item in scores]))
        result["rollout_episodes"] = scores
    report = json.dumps(result, default=str, indent=2)
    (args.output / "metrics.json").write_text(report + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()

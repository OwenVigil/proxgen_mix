"""Validate and summarize the three-seed CoinRun BC experiment."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev


MODES = ("add", "sum", "concat")
SPLITS = ("train", "val", "test")


def read_result(path):
    return json.loads(path.read_text())


def collect(root):
    rows, budgets, level_sequences = {}, set(), {}
    for seed in (0, 1, 2):
        for mode in MODES:
            directory = root / f"{mode}-s{seed}"
            base = read_result(directory / "metrics.json")
            config = base["config"]
            if config["mode"] != mode or config["seed"] != seed or config["checkpoint"]:
                raise ValueError(f"Unexpected training config in {directory}")
            budgets.add(tuple(config[k] for k in
                              ("data", "groups", "steps", "batch_size", "lr", "env", "device")))
            row = {"params": base["params"], "ce": base["heldout_episode_ce"],
                   "seconds": base["train_seconds"]}
            for split in SPLITS:
                report = base if split == "val" else read_result(directory / split / "metrics.json")
                evaluation = report["config"]
                if (evaluation["eval_split"] != split or evaluation["mode"] != mode
                        or evaluation["seed"] != seed or evaluation["env"] != config["env"]):
                    raise ValueError(f"Unexpected {split} config in {directory}")
                if split != "val" and Path(evaluation["checkpoint"]).resolve() != (directory / "model.pt").resolve():
                    raise ValueError(f"Unexpected checkpoint in {directory}/{split}")
                episodes = report["rollout_episodes"]
                levels = tuple(item["level"] for item in episodes)
                if len(levels) != evaluation["rollouts"] or levels != level_sequences.setdefault(split, levels):
                    raise ValueError(f"Mismatched {split} levels in {directory}")
                row[split] = mean(item["return"] for item in episodes)
            rows[mode, seed] = row
        if rows["sum", seed]["params"] != rows["concat", seed]["params"]:
            raise ValueError("Mixer comparison is not parameter-matched")
    if len(budgets) != 1:
        raise ValueError("The training budgets differ")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    rows = collect(parser.parse_args().root)
    metrics = ("ce", "train", "val", "test", "seconds")
    print("mode seed params heldout_ce train_return val_return test_return train_seconds")
    for (mode, seed), row in rows.items():
        print(mode, seed, row["params"], *(f"{row[k]:.4f}" for k in metrics))
    print("\nmean +/- sample SD over training seeds")
    for mode in MODES:
        values = [[rows[mode, seed][k] for seed in (0, 1, 2)] for k in metrics]
        print(mode, *(f"{mean(v):.4f} +/- {stdev(v):.4f}" for v in values), sep=" | ")
    print("\npaired differences: concat minus baseline, by seed")
    for baseline in ("add", "sum"):
        for metric in ("ce", "test"):
            deltas = [rows["concat", s][metric] - rows[baseline, s][metric] for s in (0, 1, 2)]
            print(baseline, metric, *(f"{value:+.4f}" for value in deltas), f"mean={mean(deltas):+.4f}")


if __name__ == "__main__":
    main()

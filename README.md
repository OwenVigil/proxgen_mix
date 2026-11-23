# Procgen residual mixing pilot

See [EXPERIMENTS.md](EXPERIMENTS.md) for the pure-concatenation results, activation
diagnostics, and the learning-rate and activation experiments.

Procgen supplies the environment, and the [official GenDGRL benchmark](https://github.com/facebookresearch/gen_dgrl/tree/main/procgen)
supplies PPO demonstrations. Start with CoinRun easy and behavior cloning (BC).
BC measures imitation, not reward optimization; rollout return measures the policy.

## Data and setup

- Archive: `data/coinrun-1M-expert.tar.xz` (785,564,900 bytes).
- Source: [GenDGRL CoinRun 1M Expert](https://dl.fbaipublicfiles.com/DGRL/1M/expert/coinrun.tar.xz).
- Local SHA-256: `15d18d3d74bca49f65961b5340766720c33eedfaa900372509d4f0a56eaf7cd4`.
- Preprocessing uses pickle for the official trajectory format. Do not run it on untrusted archives.
- This machine has two RTX 3090s with 24 GiB each. Keep the ~12 GB BC cache on
  the root filesystem, not in the near-full `/workspace` mount.
- The venv is at `/opt/procgen-mixing-venv` (Python 3.10, NumPy 1.26.4,
  Procgen 0.10.7, PyTorch 2.7.1 CPU). The Procgen wheel needs `libglib2.0-0`
  on this Debian image. To train on a GPU, use a separate venv and install a
  CUDA wheel, e.g. `torch==2.7.1` from the official cu128 PyTorch wheel index.

```bash
cd /workspace/procgen-mixing
PY=/opt/procgen-mixing-venv/bin/python
CACHE=/opt/procgen-mixing-cache/coinrun-1M-expert
# Run only if the cache doesn't already exist:
"$PY" prepare.py data/coinrun-1M-expert.tar.xz "$CACHE"
# Alternatively, on a machine with xz 5.4+:
# xz -T16 -dc data/coinrun-1M-expert.tar.xz | "$PY" prepare.py - "$CACHE"
"$PY" -m unittest -v
for mode in add sum concat; do
  "$PY" train.py --data "$CACHE" --mode "$mode" --groups 1 --seed 0 \
    --steps 200 --batch-size 32 --rollouts 32 --device cpu --output "runs/demo-$mode"
done
```

## Controls

Six residual blocks in a three-stage, equal-width IMPALA-style CNN:

| Mode | Output | Mixer per group, width C |
| --- | --- | --- |
| add | `x + f(x)` | none |
| sum | `x + f(x) + MLP(x + f(x))` | C -> 1.5C -> C |
| concat | `x + f(x) + MLP(cat(x, f(x)))` | 2C -> C -> C |

`sum` and `concat` both add exactly 3C^2 weights per group and 3C^2
multiply-accumulates per spatial position. Actual wall time can differ due to
activation size, concatenation, or kernel efficiency. The zero-initialized output
projection makes all three start as the same policy for a paired seed. Both mixers
keep the standard residual path for this first ablation. The group counts
1, 2, 3, 6 cover global sharing through no sharing. Sharing changes parameter count,
but not the per-forward projection count. Keep expert mixing separate for now.

## Evaluation

- Logged loss: held-out trajectories from training levels, not held-out levels.
- Rollout levels follow the upstream split: train 0-199, validation 200-249, test
  sampled from 250 onward. Test evaluation requires a frozen `--checkpoint`.
- The CPU run is a pipeline check. For evidence, tune on validation only,
  then compare paired seeds (at least five) and two games (e.g. CoinRun and BigFish)
  using held-out BC loss, train and test rollout return, throughput and peak memory.
  Report paired test-return differences with a confidence interval, not only
  the best run. Compare sharing groups separately from the matched sum/concat
  contrast. Confirm any BC gains with an offline RL objective on mixed-quality
  demonstrations (e.g. CQL, using the original archive's rewards and terminal flags).
- For a MoE follow-up, hold the experts, router, top-k, and expert weights constant.
  Compare a weighted sum followed by an MLP to concatenated weighted expert outputs
  followed by an equal-budget MLP. Use a fixed slot order and test whether rank or
  expert identity is responsible for any gain.

## Smoke run

CPU, seed 0, first 512 archive trajectories (32,990 transitions), 200 optimizer
steps, batch 32, 32 paired validation episodes. The validation level sequences
were identical across all three models.

| Mode | Trainable parameters | Held-out trajectory CE | Validation return |
| --- | ---: | ---: | ---: |
| add | 658,767 | 2.2750 | 3.1250 |
| sum | 661,839 | 2.2739 | 1.5625 |
| concat | 661,839 | 2.2763 | 2.1875 |

See `runs/smoke-*/metrics.json` for raw scores and `model.pt` for checkpoints.
This short, single-seed run is not evidence of an improvement.

## Three-seed run

CoinRun easy BC, full 1M-expert cache, CPU, seed 0/1/2, global sharing, 2,000
AdamW updates at batch size 32 and learning rate 0.0005. Each model sampled
64,000 transitions with replacement from the training partition. All checkpoints
were frozen at step 2,000. Each split used the same 96 greedy episodes across
all models (environment seed 777); the test split was not used for selection.

| Mode | Held-out CE | Train return | Validation return | Test return |
| --- | ---: | ---: | ---: | ---: |
| add | 2.1984 +/- 0.0081 | 6.701 +/- 0.739 | 4.861 +/- 0.366 | 5.729 +/- 0.908 |
| sum | 2.2024 +/- 0.0106 | 6.875 +/- 1.556 | 4.861 +/- 1.243 | 5.764 +/- 1.243 |
| concat | 2.1948 +/- 0.0077 | 6.701 +/- 0.592 | 4.479 +/- 0.651 | 5.556 +/- 0.867 |

Values are mean +/- sample standard deviation across training seeds. Relative
to the parameter-matched sum mixer, concat reduced CE by 0.0076 on average,
but its paired test-return differences were +0.208, -0.313, and -0.521
(mean -0.208). Relative to plain add, its average test-return difference was
-0.174. This is a modest imitation-loss improvement, not evidence of better
generalization at this budget.

Raw metrics, checkpoints, and logs are in `runs/coinrun-bc-3seed`. Verify
matched budgets and rollout levels and print per-seed results:

```bash
/opt/procgen-mixing-venv/bin/python summarize.py runs/coinrun-bc-3seed
```

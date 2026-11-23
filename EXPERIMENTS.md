# CoinRun residual experiments

## Setup

All completed trials use CoinRun easy, GenDGRL 1M Expert, and behavior cloning
with AdamW, batch size 32, and training seeds 0, 1, 2. The same within-level
held-out trajectories are used for supervised cross entropy (CE). Rollout
returns measure policy performance and are distinct from imitation loss.
The level splits are train 0-199, validation 200-249, test 250 and above.
Runs used CPU PyTorch; throughput is specific to the concurrent workload.

## Earlier residual variants

`add` uses `s = x + f(x)`. The earlier `sum` and `concat` both preserve the
additive path and add an FFN correction: `s + FFN(s)` and
`s + FFN(cat(x, f(x)))`, respectively. Both corrections share one parameter
set across six blocks and have the same parameter and projection MAC counts.
At 2,000 steps, CE / mean test return were 2.1984 / 5.729 for `add`,
2.2024 / 5.764 for `sum`, and 2.1948 / 5.556 for `concat`. The lower CE of
the old concat model did not translate to higher test return.
See [README](README.md#three-seed-run) and [run files](runs/coinrun-bc-3seed).

## Pure concatenation and longer training

`concat-direct` replaces the sum entirely: concatenate the two branches and
apply a bias-free per-position FFN (64 -> 32 -> 32, GELU). Groups 1, 2, 3, 6
share weights over contiguous sets of 6, 3, 2, 1 blocks, respectively.
`add-deep` retains addition and adds a 3x3 convolution to the final branch.
It has 668,015 parameters; the 3-group concat has 667,983, differing by 32.
Other concat group counts are also within 1.4% of the baseline.

| Model | Held-out CE at 2k | Test at 2k | Test at 4k | Test at 6k | Test at 8k | Held-out CE at 8k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| add-deep | 2.1903 | 5.833 | 6.285 | 6.424 | 6.528 | 2.0989 |
| concat-direct g1 | 2.2499 | 3.819 | 5.451 | 5.590 | 5.694 | 2.1668 |
| concat-direct g2 | 2.2549 | 4.097 | 5.347 | 5.417 | 5.069 | 2.1833 |
| concat-direct g3 | 2.2571 | 3.299 | 5.417 | 5.382 | 4.896 | 2.1862 |
| concat-direct g6 | 2.2548 | 4.236 | 5.451 | 5.243 | 5.938 | 2.1640 |

Entries are three-seed means; every corresponding split used the same 96
greedy episodes. The continuation restores the 2,000-step weights, resets
AdamW state for all models, and advances the batch RNG past previously sampled
batches. These are not uninterrupted 8,000-step runs.

At 2k, every concat variant was behind add-deep on all three paired seeds.
The return gap narrowed by 8k but none overtook the baseline on mean test
return. Group 3 lost test return from 4k to 8k even as CE improved. The
final training window averaged CE 2.083 for add-deep and 2.157-2.179 for
concat, with CPU throughput about 227 vs. 167-170 samples/s. The extra
projection work is not matched by matching the parameter count.
See [2k runs](runs/coinrun-direct-3seed) and [8k runs](runs/coinrun-direct-8k).

## Activation observations

[Magnitude script](inspect_magnitudes.py) probed 1,024 fixed held-out RGB
frames at initialization and after training. [Parameter](runs/layer-magnitudes/parameters.csv)
and [activation](runs/layer-magnitudes/activations.csv) CSVs include signed
mean, mean absolute value, and RMS.

- Original add at 2k: block 1 to block 6 mean absolute skip signal grew
  from 0.139 to 0.299; the branch grew from 0.038 to 0.157.
- At 8k, block 6 output magnitude was 0.813 for add-deep and 0.129-0.170 for
  concat-direct. Add-deep skip/branch cosine was +0.42 in block 6, but
  -0.26 and -0.41 in blocks 1 and 2. Growth is not monotonic over all blocks.
- Pure concat already attenuated the signal at initialization: block 6
  output was about 0.003-0.004 versus 0.087 for original add. A low output
  magnitude does not rule out large inputs inside the branch; it also does
  not imply LayerNorm or RMSNorm, which rescale by input statistics.
- A separate counterfactual probe used 512 fixed frames at 8k and counted
  the inputs to each branch's first ReLU where `abs(x) > 1.81845` (where
  tanh's derivative is below 0.1). Block 6: add-deep 5.70%, concat
  0.38-0.78%. Block 3: add-deep 0.01%, concat 6.58-13.26%.
  These percentages describe ReLU-trained models, not tanh-trained models.

The data establish scale differences, not a causal mechanism such as the
branches competing for amplitude. Smaller activations also do not establish
a higher stable learning rate or a benefit from tanh.

## Learning rate and activation trials

Both trials compare add-deep to the parameter-matched 3-group concat-direct
on paired training seeds 0, 1, 2, starting from scratch. Shared baseline:
AdamW at 5e-4 with ReLU.

| Trial | Fixed | Varied |
| --- | --- | --- |
| Learning rate | Existing activations | ReLU at 2.5e-4, 5e-4, 1e-3 |
| Activation | AdamW at 5e-4 | First activation in each residual branch: ReLU vs. tanh |

Only the six first branch activations change in the tanh arm. Other ReLUs
and fusion GELUs are unchanged. There are 24 independent jobs after counting
the shared baseline once. Each job trains for 4,000 updates and records CE
and samples/s every 1,000 steps, plus held-out CE at 2,000 and 4,000.
The frozen 4,000-step model is evaluated on 96 episodes per split using
environment seed 1777, different from the earlier runs.

All 24 jobs completed. The table reports means across the three paired training
seeds. Test return includes the sample standard deviation. The first training
window covers updates 1-1,000. Evaluation used identical level sequences across
configurations within each split.

| Model | Activation | LR | First 1k train CE | Held-out CE at 2k | Held-out CE at 4k | Train return | Validation return | Test return |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add-deep | ReLU | 2.5e-4 | 2.2934 | 2.2085 | 2.1494 | 6.944 | 5.660 | 6.736 +/- 0.060 |
| add-deep | ReLU | 5e-4 | 2.2872 | 2.1903 | 2.1314 | 7.431 | 6.528 | 6.806 +/- 0.366 |
| add-deep | ReLU | 1e-3 | 2.2876 | 2.1874 | 2.1363 | 7.049 | 6.424 | 6.840 +/- 0.421 |
| add-deep | tanh | 5e-4 | 2.2860 | 2.1852 | 2.1283 | 7.188 | 6.215 | 6.736 +/- 0.434 |
| concat-direct g3 | ReLU | 2.5e-4 | 2.3647 | 2.2659 | 2.2282 | 5.278 | 4.410 | 5.590 +/- 0.366 |
| concat-direct g3 | ReLU | 5e-4 | 2.3476 | 2.2571 | 2.2167 | 5.486 | 4.931 | 5.938 +/- 0.276 |
| concat-direct g3 | ReLU | 1e-3 | 2.3626 | 2.2607 | 2.2202 | 5.486 | 4.757 | 5.903 +/- 0.060 |
| concat-direct g3 | tanh | 5e-4 | 2.3443 | 2.2546 | 2.2172 | 5.382 | 5.243 | 5.833 +/- 0.180 |

All recorded training windows were finite. For ReLU, 5e-4 gave the highest
4k validation return in both architectures. Doubling it increased concat's
first-window CE by 0.0150 and 4k held-out CE by 0.0035. Its paired test
changes were -0.208, -0.208, and +0.313 (mean -0.035). The tested grid
does not identify the maximum stable learning rate.

With tanh at 5e-4, concat's held-out CE improved by 0.0025 at 2k and was
0.0005 higher at 4k. Mean validation return rose by 0.313, and paired test
changes were -0.104, -0.313, and +0.104 (mean -0.104). For add-deep, mean
test return changed by -0.069. The activation comparison covers one learning
rate and one placement. Every concat condition had lower mean 4k test return
than its corresponding add-deep condition. The current rollout random seed
is 1777, so the table is a separate evaluation from the earlier seed-777
continuation.

[Launcher](run_lr_activation.sh) | [Run directory](runs/coinrun-lr-activation-4k)

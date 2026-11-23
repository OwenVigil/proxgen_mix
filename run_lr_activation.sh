#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

root=${1:?usage: bash run_lr_activation.sh RUN_DIRECTORY}
python=${PYTHON:-/opt/procgen-mixing-venv/bin/python}
data=${DATA:-/opt/procgen-mixing-cache/coinrun-1M-expert}
mkdir -- "$root"
export OPENBLAS_NUM_THREADS=1

pids=()
for seed in 0 1 2; do
    for mode in add-deep concat-direct; do
        groups=1
        if [[ "$mode" == concat-direct ]]; then groups=3; fi
        for arm in relu-2.5e-4 relu-5e-4 relu-1e-3 tanh-5e-4; do
            name="$root/$mode-$arm-s$seed"
            (
                common=(--data "$data" --mode "$mode" --groups "$groups" --seed "$seed"
                        --first-activation "${arm%%-*}" --lr "${arm#*-}"
                        --batch-size 32 --device cpu --eval-seed 1777)
                "$python" train.py "${common[@]}" --steps 4000 --snapshot-every 2000 \
                    --rollouts 0 --output "$name"
                for split in val train test; do
                    "$python" train.py "${common[@]}" --steps 4000 --eval-split "$split" \
                        --rollouts 96 --checkpoint "$name/model.pt" --output "$name/$split"
                done
                "$python" train.py "${common[@]}" --steps 2000 --rollouts 0 \
                    --checkpoint "$name/model-2000.pt" --output "$name/step-2000"
                printf 'completed %s\n' "$name"
            ) > "$name.log" 2>&1 &
            pids+=("$!")
        done
    done
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
printf 'learning-rate and activation exit status: %s\n' "$status"
exit "$status"

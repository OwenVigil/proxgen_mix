#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"

root=${1:?usage: bash run_direct_comparison.sh RUN_DIRECTORY}
python=${PYTHON:-/opt/procgen-mixing-venv/bin/python}
data=${DATA:-/opt/procgen-mixing-cache/coinrun-1M-expert}
mkdir -- "$root"
export OPENBLAS_NUM_THREADS=1

pids=()
for seed in 0 1 2; do
    for groups in 0 1 2 3 6; do
        if [[ "$groups" == 0 ]]; then
            mode=add-deep
            group_count=1
            name="$root/add-deep-s$seed"
        else
            mode=concat-direct
            group_count=$groups
            name="$root/concat-direct-g$groups-s$seed"
        fi
        (
            common=(--data "$data" --mode "$mode" --groups "$group_count" --seed "$seed"
                    --steps 2000 --batch-size 32 --device cpu --rollouts 96)
            "$python" train.py "${common[@]}" --output "$name"
            for split in train test; do
                "$python" train.py "${common[@]}" --eval-split "$split" \
                    --checkpoint "$name/model.pt" --output "$name/$split"
            done
            printf 'completed %s\n' "$name"
        ) > "$name.log" 2>&1 &
        pids+=("$!")
    done
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
printf 'comparison exit status: %s\n' "$status"
exit "$status"

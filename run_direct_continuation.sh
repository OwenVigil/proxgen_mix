#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"

source_root=${1:?usage: bash run_direct_continuation.sh SOURCE_RUN NEW_RUN}
root=${2:?usage: bash run_direct_continuation.sh SOURCE_RUN NEW_RUN}
python=${PYTHON:-/opt/procgen-mixing-venv/bin/python}
data=${DATA:-/opt/procgen-mixing-cache/coinrun-1M-expert}
mkdir -- "$root"
export OPENBLAS_NUM_THREADS=1
pids=()
for seed in 0 1 2; do
    for groups in 0 1 2 3 6; do
        mode=concat-direct
        group_count=$groups
        run="concat-direct-g$groups-s$seed"
        if [[ "$groups" == 0 ]]; then
            mode=add-deep
            group_count=1
            run="add-deep-s$seed"
        fi
        name="$root/$run"
        (
            common=(--data "$data" --mode "$mode" --groups "$group_count" --seed "$seed"
                    --batch-size 32 --device cpu --rollouts 96)
            "$python" train.py "${common[@]}" --start-step 2000 --steps 8000 \
                --warm-start "$source_root/$run/model.pt" --snapshot-every 2000 --rollouts 0 \
                --output "$name"
            for step in 4000 6000 8000; do
                model="$name/model-$step.pt"
                if [[ "$step" == 8000 ]]; then model="$name/model.pt"; fi
                for split in val train test; do
                    "$python" train.py "${common[@]}" --steps "$step" --eval-split "$split" \
                        --checkpoint "$model" --output "$name/step-$step/$split"
                done
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
printf 'continuation exit status: %s\n' "$status"
exit "$status"

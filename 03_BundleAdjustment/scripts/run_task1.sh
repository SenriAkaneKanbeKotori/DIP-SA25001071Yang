#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pytorch}"
ITERS="${ITERS:-3000}"
POINT_BATCH="${POINT_BATCH:-4096}"
DEVICE="${DEVICE:-auto}"

conda run -n "$CONDA_ENV" python src/ba_torch.py \
    --data-dir data \
    --output-dir outputs/task1 \
    --iters "$ITERS" \
    --point-batch "$POINT_BATCH" \
    --device "$DEVICE"


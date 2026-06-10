#!/usr/bin/env bash
set -euo pipefail

FILE="${1:-maps}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="${SCRIPT_DIR}/datasets"
URL="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/${FILE}.tar.gz"
TAR_FILE="${DATASETS_DIR}/${FILE}.tar.gz"
TARGET_DIR="${DATASETS_DIR}/${FILE}"

mkdir -p "${DATASETS_DIR}"

echo "Downloading ${URL} ..."
wget -N "${URL}" -O "${TAR_FILE}"

echo "Extracting ${TAR_FILE} ..."
tar -zxvf "${TAR_FILE}" -C "${DATASETS_DIR}"
rm "${TAR_FILE}"

pushd "${SCRIPT_DIR}" >/dev/null
if [ ! -d "datasets/${FILE}/train" ]; then
  echo "Cannot find datasets/${FILE}/train after extraction." >&2
  exit 1
fi

VAL_SPLIT="val"
if [ ! -d "datasets/${FILE}/val" ] && [ -d "datasets/${FILE}/test" ]; then
  VAL_SPLIT="test"
fi

find "datasets/${FILE}/train" -type f \( -name "*.jpg" -o -name "*.png" \) | sort -V > train_list.txt
find "datasets/${FILE}/${VAL_SPLIT}" -type f \( -name "*.jpg" -o -name "*.png" \) | sort -V > val_list.txt
popd >/dev/null

echo "Wrote train_list.txt and val_list.txt for ${FILE}."

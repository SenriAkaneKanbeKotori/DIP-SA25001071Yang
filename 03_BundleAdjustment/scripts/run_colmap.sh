#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-data}"
IMAGE_PATH="$DATASET_PATH/images"
COLMAP_PATH="$DATASET_PATH/colmap"
SPARSE_PATH="$COLMAP_PATH/sparse"
DENSE_PATH="$COLMAP_PATH/dense"

mkdir -p "$SPARSE_PATH" "$DENSE_PATH"

echo "=== Step 1: Feature Extraction ==="
colmap feature_extractor \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1

echo "=== Step 2: Feature Matching ==="
colmap exhaustive_matcher \
    --database_path "$COLMAP_PATH/database.db" \
    --SiftMatching.use_gpu 1

echo "=== Step 3: Sparse Reconstruction ==="
colmap mapper \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --output_path "$SPARSE_PATH"

echo "=== Step 4: Image Undistortion ==="
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$SPARSE_PATH/0" \
    --output_path "$DENSE_PATH"

echo "=== Step 5: Dense Reconstruction ==="
colmap patch_match_stereo \
    --workspace_path "$DENSE_PATH"

echo "=== Step 6: Stereo Fusion ==="
colmap stereo_fusion \
    --workspace_path "$DENSE_PATH" \
    --output_path "$DENSE_PATH/fused.ply"

echo "Results:"
echo "  Sparse: $SPARSE_PATH/0"
echo "  Dense:  $DENSE_PATH/fused.ply"


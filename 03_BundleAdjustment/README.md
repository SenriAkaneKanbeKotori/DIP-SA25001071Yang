# Assignment 03: Bundle Adjustment

This folder is organized as a standalone GitHub repository for the Bundle
Adjustment assignment.

## What is included

```text
.
├── data/                    # copied assignment data, added after packaging
├── pics/                    # copied assignment figures, added after packaging
├── src/ba_torch.py          # Task 1: PyTorch bundle adjustment
├── scripts/run_task1.ps1    # Windows runner for Task 1
├── scripts/run_task1.sh     # Unix runner for Task 1
├── scripts/run_colmap.ps1   # Windows COLMAP pipeline for Task 2
├── scripts/run_colmap.sh    # Unix COLMAP pipeline for Task 2
├── outputs/                 # generated results
├── report.md                # Markdown assignment report
├── environment.yml
└── requirements.txt
```

## Environment

The code uses PyTorch, NumPy, and Matplotlib. If the local `pytorch` conda
environment already exists, no new environment is needed.

```powershell
conda activate pytorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

To recreate a compatible conda environment:

```powershell
conda env create -f environment.yml
conda activate pytorch
```

## Task 1: PyTorch Bundle Adjustment

Run the full optimization from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_task1.ps1
```

Equivalent direct command:

```powershell
conda run -n pytorch python src/ba_torch.py --data-dir data --output-dir outputs/task1 --device auto --iters 3000 --point-batch 4096
```

Generated files:

```text
outputs/task1/reconstruction.obj
outputs/task1/loss_curve.png
outputs/task1/pointcloud_preview.png
outputs/task1/camera_params.npz
outputs/task1/points3d_optimized.npy
outputs/task1/metrics.json
```

The OBJ file stores colored vertices as:

```text
v x y z r g b
```

where RGB values are normalized to `[0, 1]`.

## Task 2: COLMAP Reconstruction

The slides state that Task 2 has no code requirement. The provided scripts run
the full COLMAP command-line pipeline for reproducibility.

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_colmap.ps1
```

Sparse-only quick run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_colmap.ps1 -SkipDense
```

Unix:

```bash
bash scripts/run_colmap.sh
```

Expected COLMAP outputs:

```text
data/colmap/sparse/0/
data/colmap/dense/fused.ply
```

## Report

The Markdown report is in `report.md`. After running Task 1 and Task 2, keep the
generated output files in the paths shown above so the report links remain valid.

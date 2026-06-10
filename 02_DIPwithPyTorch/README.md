# Assignment 2 - DIP with PyTorch

This repository folder contains the completed files for Assignment 2:

1. Traditional DIP: Poisson Image Editing implemented with PyTorch.
2. Deep-learning DIP: Pix2Pix-style image-to-image translation with a fully
   convolutional PyTorch network.
3. Markdown report: `REPORT.md`.

## Project Structure

```text
.
├── README.md
├── REPORT.md
├── requirements.txt
├── run_blending_gradio.py
├── data_poisson/
├── assets/
├── scripts/
│   ├── run_poisson_examples.py
│   └── smoke_test.py
└── Pix2Pix/
    ├── FCN_network.py
    ├── facades_dataset.py
    ├── train.py
    ├── download_pix2pix_dataset.sh
    └── download_facades_dataset.sh
```

## Environment

Use the configured PyTorch conda environment:

```bash
conda activate pytorch
pip install -r requirements.txt
```

## Poisson Image Editing

Run the interactive Gradio app:

```bash
python run_blending_gradio.py
```

Generate the fixed examples used by the report:

```bash
python scripts/run_poisson_examples.py
```

The generated images are saved to `assets/poisson_results/`.

## Pix2Pix-style FCN

Download a pix2pix dataset with more samples than the facades template. The
recommended default is `maps`:

```bash
cd Pix2Pix
bash download_pix2pix_dataset.sh maps
python train.py --input-side left --epochs 200 --batch-size 16
```

Training writes validation images to `Pix2Pix/results/val/`, loss history to
`Pix2Pix/results/history.csv`, and checkpoints to `Pix2Pix/checkpoints/`.

## Quick Check

```bash
python scripts/smoke_test.py
```

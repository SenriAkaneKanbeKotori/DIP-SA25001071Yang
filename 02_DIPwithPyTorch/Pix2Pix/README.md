# Pix2Pix-style FCN

This folder implements the deep-learning part of Assignment 2 with a fully
convolutional encoder-decoder network in PyTorch.

## Files

- `FCN_network.py`: fully convolutional encoder-decoder model.
- `facades_dataset.py`: paired-image dataset reader for pix2pix side-by-side images.
- `train.py`: training, validation, checkpoints, and result visualization.
- `download_pix2pix_dataset.sh`: download one official pix2pix dataset.

## Dataset

The assignment asks for a dataset with more samples than the default facades
example. The recommended default here is `maps`.

```bash
cd Pix2Pix
bash download_pix2pix_dataset.sh maps
```

Other official pix2pix dataset names can also be passed, for example:

```bash
bash download_pix2pix_dataset.sh cityscapes
bash download_pix2pix_dataset.sh edges2shoes
```

The script creates `train_list.txt` and `val_list.txt` in this folder.

## Train

```bash
python train.py --input-side left --epochs 200 --batch-size 16
```

If the desired input is on the right half of each paired image, use:

```bash
python train.py --input-side right
```

Training writes:

- `results/history.csv`: train and validation L1 loss per epoch.
- `results/train/` and `results/val/`: input / target / output comparison images.
- `checkpoints/`: model checkpoints.

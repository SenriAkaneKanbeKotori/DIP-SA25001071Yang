from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


class PairedImageDataset(Dataset):
    """
    Dataset for pix2pix-style paired images stored as left/right halves.
    The official facades, maps, cityscapes, and edges2* datasets follow this
    format after extraction from the Berkeley pix2pix dataset archive.
    """

    def __init__(self, list_file, image_size=256, input_side="left"):
        self.list_file = Path(list_file)
        self.image_size = image_size
        self.input_side = input_side

        if input_side not in {"left", "right"}:
            raise ValueError("input_side must be 'left' or 'right'")

        with self.list_file.open("r", encoding="utf-8") as file:
            self.image_filenames = [line.strip() for line in file if line.strip()]

        if not self.image_filenames:
            raise ValueError(f"No image paths were found in {self.list_file}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = Path(self.image_filenames[idx])
        if not img_path.is_absolute():
            img_path = self.list_file.parent / img_path
        paired = Image.open(img_path).convert("RGB")
        width, height = paired.size
        half_width = width // 2

        left = paired.crop((0, 0, half_width, height))
        right = paired.crop((half_width, 0, width, height))

        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            left = left.resize(size, Image.BICUBIC)
            right = right.resize(size, Image.BICUBIC)

        if self.input_side == "left":
            image_input, image_target = left, right
        else:
            image_input, image_target = right, left

        return image_to_tensor(image_input), image_to_tensor(image_target)


class FacadesDataset(PairedImageDataset):
    """Backward-compatible name used by the assignment template."""

    pass

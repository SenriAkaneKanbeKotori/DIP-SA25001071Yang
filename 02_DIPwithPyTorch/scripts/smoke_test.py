from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Pix2Pix"))

from run_blending_gradio import cal_laplacian_loss, create_mask_from_points  # noqa: E402
from FCN_network import FullyConvNetwork  # noqa: E402


def test_poisson_core():
    points = np.array([(2, 2), (13, 3), (11, 13), (3, 12)])
    mask = create_mask_from_points(points, 16, 16)
    assert mask.shape == (16, 16)
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask.min() == 0

    fg = torch.rand(1, 3, 16, 16)
    bg = torch.rand(1, 3, 16, 16, requires_grad=True)
    fg_mask = torch.from_numpy(mask.copy()).view(1, 1, 16, 16).float() / 255.0
    bg_mask = fg_mask.clone()
    loss = cal_laplacian_loss(fg, fg_mask, bg, bg_mask)
    assert loss.ndim == 0
    loss.backward()
    assert bg.grad is not None


def test_pix2pix_network():
    model = FullyConvNetwork().eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert y.min().item() >= -1.0
    assert y.max().item() <= 1.0


if __name__ == "__main__":
    test_poisson_core()
    test_pix2pix_network()
    print("Smoke tests passed.")

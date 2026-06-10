"""Bundle Adjustment from scratch using PyTorch.

The script optimizes a shared focal length, per-view Euler-angle camera
extrinsics, and all 3D point coordinates from masked 2D observations.

Run from the repository root:
    python src/ba_torch.py --data-dir data --output-dir outputs/task1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    device: str
    seed: int
    iters: int
    point_batch: int
    log_every: int
    eval_chunk: int
    image_width: int
    image_height: int
    init_focal: float
    init_depth: float
    init_yaw_range_deg: float
    lr_points: float
    lr_pose: float
    lr_focal: float
    huber_delta: float
    center_reg: float
    focal_reg: float


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="PyTorch Bundle Adjustment")
    parser.add_argument("--data-dir", default="data", help="Directory containing points2d.npz and points3d_colors.npy")
    parser.add_argument("--output-dir", default="outputs/task1", help="Directory for OBJ, plots, and metrics")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Training device")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--point-batch", type=int, default=4096, help="Number of 3D points sampled per step; <=0 uses all")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-chunk", type=int, default=4096)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--image-height", type=int, default=1024)
    parser.add_argument("--init-focal", type=float, default=1024.0)
    parser.add_argument("--init-depth", type=float, default=3.0)
    parser.add_argument(
        "--init-yaw-range-deg",
        type=float,
        default=70.0,
        help="Initialize view yaw angles linearly in [-range, range]. Use 0 for identity cameras.",
    )
    parser.add_argument("--lr-points", type=float, default=2e-3)
    parser.add_argument("--lr-pose", type=float, default=1e-3)
    parser.add_argument("--lr-focal", type=float, default=5e-4)
    parser.add_argument("--huber-delta", type=float, default=8.0)
    parser.add_argument("--center-reg", type=float, default=1e-3)
    parser.add_argument("--focal-reg", type=float, default=1e-4)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        seed=args.seed,
        iters=args.iters,
        point_batch=args.point_batch,
        log_every=args.log_every,
        eval_chunk=args.eval_chunk,
        image_width=args.image_width,
        image_height=args.image_height,
        init_focal=args.init_focal,
        init_depth=args.init_depth,
        init_yaw_range_deg=args.init_yaw_range_deg,
        lr_points=args.lr_points,
        lr_pose=args.lr_pose,
        lr_focal=args.lr_focal,
        huber_delta=args.huber_delta,
        center_reg=args.center_reg,
        focal_reg=args.focal_reg,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_observations(data_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    points2d_path = data_dir / "points2d.npz"
    colors_path = data_dir / "points3d_colors.npy"
    if not points2d_path.exists():
        raise FileNotFoundError(f"Missing {points2d_path}")
    if not colors_path.exists():
        raise FileNotFoundError(f"Missing {colors_path}")

    points2d = np.load(points2d_path)
    keys = sorted(points2d.files)
    observations = np.stack([points2d[key] for key in keys], axis=0).astype(np.float32)
    colors = np.load(colors_path).astype(np.float32)
    if observations.ndim != 3 or observations.shape[-1] != 3:
        raise ValueError(f"Expected observations with shape (views, points, 3), got {observations.shape}")
    if colors.shape[0] != observations.shape[1]:
        raise ValueError("Color count does not match point count")
    return keys, observations, colors


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles in radians to rotation matrices.

    Args:
        euler: Tensor of shape (..., 3), ordered as rotation about X, Y, Z.

    Returns:
        Tensor of shape (..., 3, 3).
    """
    x, y, z = euler.unbind(dim=-1)
    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return rx @ ry @ rz


def initialize_points(
    xy: torch.Tensor,
    visible: torch.Tensor,
    focal: float,
    depth: float,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Triangulation-free initialization from mean visible image coordinates."""
    cx = image_width / 2.0
    cy = image_height / 2.0
    weights = visible.float()
    counts = weights.sum(dim=0).clamp_min(1.0)
    mean_xy = (xy * weights[..., None]).sum(dim=0) / counts[:, None]

    x = (mean_xy[:, 0] - cx) * depth / focal
    y = (cy - mean_xy[:, 1]) * depth / focal
    z = torch.zeros_like(x)
    points = torch.stack([x, y, z], dim=-1)
    points += 0.01 * torch.randn_like(points)
    return points


def initialize_cameras(num_views: int, depth: float, yaw_range_deg: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    euler = torch.zeros(num_views, 3, device=device)
    if yaw_range_deg != 0:
        yaw = torch.linspace(-math.radians(yaw_range_deg), math.radians(yaw_range_deg), num_views, device=device)
        euler[:, 1] = yaw

    translation = torch.zeros(num_views, 3, device=device)
    translation[:, 2] = -depth
    return euler, translation


def project_points(
    points: torch.Tensor,
    euler: torch.Tensor,
    translation: torch.Tensor,
    focal: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Project 3D points with the assignment's camera convention.

    X_c = R @ X + T
    u = -f * X_c / Z_c + cx
    v =  f * Y_c / Z_c + cy
    """
    rotation = euler_xyz_to_matrix(euler)
    camera_points = torch.einsum("vij,nj->vni", rotation, points) + translation[:, None, :]
    z = camera_points[..., 2]
    eps = torch.tensor(1e-6, device=z.device, dtype=z.dtype)
    z = torch.where(z.abs() < eps, -eps.expand_as(z), z)

    u = -focal * camera_points[..., 0] / z + image_width / 2.0
    v = focal * camera_points[..., 1] / z + image_height / 2.0
    return torch.stack([u, v], dim=-1)


def robust_reprojection_loss(pred: torch.Tensor, target: torch.Tensor, visible: torch.Tensor, huber_delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    residual = pred - target
    residual = residual[visible]
    if residual.numel() == 0:
        raise RuntimeError("Sampled batch contains no visible observations")

    loss = F.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        beta=huber_delta,
        reduction="mean",
    )
    rmse = torch.sqrt(torch.mean(torch.sum(residual * residual, dim=-1)))
    return loss, rmse


@torch.no_grad()
def evaluate_rmse(
    points: torch.Tensor,
    euler: torch.Tensor,
    translation: torch.Tensor,
    focal: torch.Tensor,
    xy: torch.Tensor,
    visible: torch.Tensor,
    cfg: TrainConfig,
) -> dict[str, float]:
    sq_sum = 0.0
    abs_sum = 0.0
    obs_count = 0
    for start in range(0, points.shape[0], cfg.eval_chunk):
        end = min(start + cfg.eval_chunk, points.shape[0])
        pred = project_points(points[start:end], euler, translation, focal, cfg.image_width, cfg.image_height)
        target = xy[:, start:end]
        mask = visible[:, start:end]
        residual = pred - target
        residual = residual[mask]
        if residual.numel() == 0:
            continue
        sq_sum += torch.sum(torch.sum(residual * residual, dim=-1)).item()
        abs_sum += torch.sum(torch.linalg.norm(residual, dim=-1)).item()
        obs_count += residual.shape[0]

    if obs_count == 0:
        return {"rmse_px": float("nan"), "mae_px": float("nan"), "visible_observations": 0}
    return {
        "rmse_px": math.sqrt(sq_sum / obs_count),
        "mae_px": abs_sum / obs_count,
        "visible_observations": int(obs_count),
    }


def write_obj(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    colors = colors.astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    with path.open("w", encoding="utf-8") as f:
        for point, color in zip(points, colors):
            f.write(
                "v "
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )


def save_loss_plot(path: Path, history: list[dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [entry["iter"] for entry in history]
    losses = [entry["loss"] for entry in history]
    rmses = [entry["batch_rmse_px"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(iterations, losses, color="#1f77b4")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Smooth L1 loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, rmses, color="#d62728")
    axes[1].set_title("Batch reprojection RMSE")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Pixels")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_pointcloud_preview(path: Path, points: np.ndarray, colors: np.ndarray, max_points: int = 12000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = colors.astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)

    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        ids = rng.choice(points.shape[0], size=max_points, replace=False)
        plot_points = points[ids]
        plot_colors = colors[ids]
    else:
        plot_points = points
        plot_colors = colors

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        plot_points[:, 2],
        c=plot_colors,
        s=0.5,
        linewidths=0,
    )
    ax.view_init(elev=10, azim=-90)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Optimized 3D point cloud")

    mins = plot_points.min(axis=0)
    maxs = plot_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys, obs_np, colors_np = load_observations(Path(cfg.data_dir))
    device = torch.device(cfg.device)
    obs = torch.from_numpy(obs_np).to(device)
    xy = obs[..., :2]
    visible = obs[..., 2] > 0.5
    num_views, num_points = visible.shape

    init_points = initialize_points(
        xy=xy,
        visible=visible,
        focal=cfg.init_focal,
        depth=cfg.init_depth,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
    )
    init_euler, init_translation = initialize_cameras(num_views, cfg.init_depth, cfg.init_yaw_range_deg, device)

    points = torch.nn.Parameter(init_points)
    euler = torch.nn.Parameter(init_euler)
    translation = torch.nn.Parameter(init_translation)
    log_focal_scale = torch.nn.Parameter(torch.zeros((), device=device))

    optimizer = torch.optim.Adam(
        [
            {"params": [points], "lr": cfg.lr_points},
            {"params": [euler, translation], "lr": cfg.lr_pose},
            {"params": [log_focal_scale], "lr": cfg.lr_focal},
        ]
    )

    valid_point_ids = torch.where(visible.any(dim=0))[0]
    history: list[dict[str, float]] = []
    start_time = time.time()

    print(f"Device: {device}")
    print(f"Views: {num_views}, points: {num_points}, visible observations: {int(visible.sum().item())}")
    print(f"Output: {output_dir}")

    for iteration in range(1, cfg.iters + 1):
        if cfg.point_batch <= 0 or cfg.point_batch >= valid_point_ids.numel():
            point_ids = valid_point_ids
        else:
            rand_ids = torch.randint(0, valid_point_ids.numel(), (cfg.point_batch,), device=device)
            point_ids = valid_point_ids[rand_ids]

        focal = cfg.init_focal * torch.exp(log_focal_scale)
        pred = project_points(points[point_ids], euler, translation, focal, cfg.image_width, cfg.image_height)
        loss_data, batch_rmse = robust_reprojection_loss(pred, xy[:, point_ids], visible[:, point_ids], cfg.huber_delta)

        center_loss = points.mean(dim=0).square().sum()
        focal_loss = log_focal_scale.square()
        loss = loss_data + cfg.center_reg * center_loss + cfg.focal_reg * focal_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % cfg.log_every == 0 or iteration == cfg.iters:
            entry = {
                "iter": iteration,
                "loss": float(loss.detach().cpu()),
                "data_loss": float(loss_data.detach().cpu()),
                "batch_rmse_px": float(batch_rmse.detach().cpu()),
                "focal": float(focal.detach().cpu()),
                "elapsed_sec": time.time() - start_time,
            }
            history.append(entry)
            print(
                f"[{iteration:5d}/{cfg.iters}] "
                f"loss={entry['loss']:.6f} "
                f"rmse={entry['batch_rmse_px']:.3f}px "
                f"f={entry['focal']:.2f}"
            )

    focal = cfg.init_focal * torch.exp(log_focal_scale)
    final_metrics = evaluate_rmse(points, euler, translation, focal, xy, visible, cfg)
    final_points_np = points.detach().cpu().numpy()
    euler_np = euler.detach().cpu().numpy()
    translation_np = translation.detach().cpu().numpy()
    focal_value = float(focal.detach().cpu())

    write_obj(output_dir / "reconstruction.obj", final_points_np, colors_np)
    save_loss_plot(output_dir / "loss_curve.png", history)
    save_pointcloud_preview(output_dir / "pointcloud_preview.png", final_points_np, colors_np)
    np.save(output_dir / "points3d_optimized.npy", final_points_np)
    np.savez(
        output_dir / "camera_params.npz",
        view_keys=np.array(keys),
        euler_xyz=euler_np,
        translation=translation_np,
        focal=np.array(focal_value, dtype=np.float32),
        image_size=np.array([cfg.image_width, cfg.image_height], dtype=np.int32),
    )

    metrics = {
        "config": asdict(cfg),
        "final": final_metrics,
        "focal": focal_value,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Final metrics:")
    print(json.dumps(final_metrics, indent=2))
    print(f"Saved OBJ: {output_dir / 'reconstruction.obj'}")
    print(f"Saved loss curve: {output_dir / 'loss_curve.png'}")
    print(f"Saved point cloud preview: {output_dir / 'pointcloud_preview.png'}")


if __name__ == "__main__":
    train(parse_args())

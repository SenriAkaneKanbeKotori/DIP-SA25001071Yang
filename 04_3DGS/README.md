# Assignment 04 - Simplified 3D Gaussian Splatting

这是按照作业 README 和 `作业04-3DGS.pptx` 流程整理的 GitHub-ready 版本。核心 TODO 已补全，示例数据位于 `data/chair/images` 与 `data/lego/images`。

## 已实现内容

- `gaussian_model.py`: 由四元数旋转和 log-scale 构造 3D Gaussian covariance。
- `gaussian_renderer.py`: 实现 3D 到 2D 投影、2D Gaussian 密度计算、按深度排序的 alpha blending。
- `data_utils.py` / `gaussian_model.py`: 对 `pytorch3d`、`natsort` 做可选依赖处理；缺少这些包时会使用纯 PyTorch/标准库 fallback。

## 环境

本机建议直接使用已配置好的 conda PyTorch 环境：

```powershell
conda activate pytorch
```

如果系统默认 `python` 不是 conda 环境，可以显式调用：

```powershell
C:\Users\13746\miniconda3\envs\pytorch\python.exe train.py --help
```

主要依赖见 `requirements.txt`。COLMAP 需要在 `PATH` 中可用，或使用本机已安装的 COLMAP 路径。

## 运行流程

以下以 `lego` 为例，`chair` 同理。

### Task 1: COLMAP SfM

```powershell
python mvs_with_colmap.py --data_dir data/lego
python debug_mvs_by_projecting_pts.py --data_dir data/lego
```

输出：

- `data/lego/database.db`
- `data/lego/sparse/0_text/{cameras.txt, images.txt, points3D.txt}`
- `data/lego/projections/*.png`

### Task 2: 训练简化版 3DGS

```powershell
python train.py --colmap_dir data/lego --checkpoint_dir data/lego/checkpoints --num_epochs 80 --device cuda
```

输出：

- `data/lego/checkpoints/checkpoint_*.pt`
- `data/lego/checkpoints/debug_images/epoch_*.png`
- `data/lego/checkpoints/debug_rendering.mp4`

### Optional: 渲染环绕视频

```powershell
python render_3dgs_mv.py --colmap_dir data/lego --checkpoint data/lego/checkpoints/checkpoint_000060.pt --num_frames 240 --fps 30
```

默认输出：`data/lego/render_mv.mp4`

### Task 3: 与官方 3DGS 对比

使用相同 scene 跑官方实现，然后把渲染质量、训练速度、显存占用填入 `REPORT.md` 的对比表。

官方仓库：https://github.com/graphdeco-inria/gaussian-splatting

## 辅助脚本

`scripts/` 下提供 PowerShell 包装脚本：

- `run_colmap.ps1`: 跑 COLMAP 与投影验证。
- `train_scene.ps1`: 训练当前纯 PyTorch 简化版。
- `render_video.ps1`: 使用 checkpoint 渲染多视角视频。

## GitHub 提交建议

`.gitignore` 已忽略 COLMAP 数据库、重建输出、checkpoint、debug 图像和视频。若课程要求提交结果图或视频，可以把少量精选结果复制到 `results/` 后加入 Git，并在 `REPORT.md` 中引用。

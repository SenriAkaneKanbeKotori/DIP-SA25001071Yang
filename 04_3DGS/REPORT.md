# 作业 04 - Simplified 3D Gaussian Splatting 实验报告

姓名：  
学号：  
实验场景：`lego` / `chair`  
运行环境：Windows, conda `pytorch`, PyTorch + CUDA  

## 1. Structure-from-Motion with COLMAP

运行命令：

```powershell
python mvs_with_colmap.py --data_dir data/<scene>
python debug_mvs_by_projecting_pts.py --data_dir data/<scene>
```

输出文件：

- `data/<scene>/sparse/0_text/cameras.txt`
- `data/<scene>/sparse/0_text/images.txt`
- `data/<scene>/sparse/0_text/points3D.txt`
- `data/<scene>/projections/*.png`

结果记录：

| 指标 | 数值 |
| --- | --- |
| 注册图像数量 | 待填写 |
| 稀疏点数量 | 待填写 |
| 平均重投影误差 | 待填写 |

投影验证图：

![COLMAP projection](results/colmap_projection_r0.png)

分析：COLMAP 恢复了相机内外参和稀疏点云。投影验证图中，彩色稀疏点应与输入视图中的物体轮廓和纹理区域基本对齐；若出现整体偏移，通常需要检查相机模型、图像顺序和内参缩放。

## 2. Simplified 3D Gaussian Splatting

### 2.1 核心实现

本实现使用 SfM 点作为 Gaussian 均值，颜色来自 COLMAP 点颜色，不透明度、旋转和尺度作为可优化参数。完成的核心公式如下：

- 3D covariance：`Sigma = R S S^T R^T`
- 2D covariance：`Sigma' = J W Sigma W^T J^T`
- 2D Gaussian：按归一化二维高斯密度计算每个像素处的响应
- Alpha blending：按深度从近到远排序，使用 `T_i = prod_{j<i}(1 - alpha_j)` 合成颜色

### 2.2 训练

运行命令：

```powershell
python train.py --colmap_dir data/<scene> --checkpoint_dir data/<scene>/checkpoints --num_epochs 80 --device cuda
```

训练记录：

| Epoch | Loss | 备注 |
| --- | --- | --- |
| 0 | 待填写 | 初始化渲染 |
| 20 | 待填写 | 待填写 |
| 40 | 待填写 | 待填写 |
| 60 | 待填写 | 待填写 |

可视化结果：

![debug image](results/simplified_3dgs_epoch_0060.png)

观察与分析：简化版实现能通过可微投影和 alpha blending 优化点的位置、颜色、透明度、尺度和旋转。由于没有 tile-based rasterizer、可见性裁剪、adaptive densification 和更高效的 CUDA kernel，训练速度和细节重建能力会明显弱于官方实现；稀疏 SfM 点不足时，物体边缘和细小结构通常更容易变模糊或缺失。

## 3. 与官方 3DGS 实现对比

官方实现仓库：https://github.com/graphdeco-inria/gaussian-splatting

对比设置：使用相同 scene、相同输入图像和 COLMAP 数据，训练到相近可视质量或相同迭代数后比较。

| 方法 | 渲染质量 | 训练速度 | 显存占用 |
| --- | --- | --- | --- |
| 本作业纯 PyTorch 简化版 | 待填写 | 待填写 | 待填写 |
| 官方 3DGS | 待填写 | 待填写 | 待填写 |

差异来源：

1. 官方实现使用 CUDA rasterizer 和 tile-based splatting，能显著减少无效 Gaussian 与像素对的计算。
2. 官方实现包含 adaptive densification、pruning 和 opacity reset，可动态增加有效 Gaussian 并移除低贡献 Gaussian。
3. 官方实现支持更完整的可见性处理、屏幕空间半径控制和球谐颜色表示，因此细节、边缘和视角相关外观通常更好。
4. 本作业版本使用纯 PyTorch 张量广播，逻辑清晰但会为大量 Gaussian 和像素生成中间张量，速度和显存效率都较低。

## 4. 结论

本实验完成了从多视角图像到稀疏 SfM、再到简化 3D Gaussian Splatting 的完整 pipeline。通过实现 covariance 构造、透视投影、2D Gaussian 评估和 alpha blending，可以直观看到 3DGS 的核心可微渲染思想；与官方实现相比，本版本更适合理解原理，而官方版本更适合高质量和实时渲染。

param(
    [string]$Scene = "lego",
    [string]$Checkpoint = "data/lego/checkpoints/checkpoint_000060.pt",
    [int]$Frames = 240,
    [int]$Fps = 30,
    [string]$Device = "cuda",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python render_3dgs_mv.py `
    --colmap_dir "data/$Scene" `
    --checkpoint $Checkpoint `
    --num_frames $Frames `
    --fps $Fps `
    --device $Device

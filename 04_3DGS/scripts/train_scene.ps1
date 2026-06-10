param(
    [string]$Scene = "lego",
    [int]$Epochs = 80,
    [string]$Device = "cuda",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$DataDir = "data/$Scene"
$CheckpointDir = "$DataDir/checkpoints"

& $Python train.py `
    --colmap_dir $DataDir `
    --checkpoint_dir $CheckpointDir `
    --num_epochs $Epochs `
    --device $Device

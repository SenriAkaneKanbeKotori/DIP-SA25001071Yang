param(
    [string]$CondaEnv = "pytorch",
    [int]$Iters = 3000,
    [int]$PointBatch = 4096,
    [string]$Device = "auto"
)

conda run -n $CondaEnv python src/ba_torch.py `
    --data-dir data `
    --output-dir outputs/task1 `
    --iters $Iters `
    --point-batch $PointBatch `
    --device $Device


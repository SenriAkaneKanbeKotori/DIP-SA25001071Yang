param(
    [string]$Scene = "lego",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$DataDir = "data/$Scene"

& $Python mvs_with_colmap.py --data_dir $DataDir
& $Python debug_mvs_by_projecting_pts.py --data_dir $DataDir

param(
    [string]$DataDir = "data",
    [switch]$SkipDense
)

$ImagePath = Join-Path $DataDir "images"
$ColmapPath = Join-Path $DataDir "colmap"
$SparsePath = Join-Path $ColmapPath "sparse"
$DensePath = Join-Path $ColmapPath "dense"
$DatabasePath = Join-Path $ColmapPath "database.db"

New-Item -ItemType Directory -Force $SparsePath | Out-Null
New-Item -ItemType Directory -Force $DensePath | Out-Null

Write-Host "=== Step 1: Feature Extraction ==="
colmap feature_extractor `
    --database_path $DatabasePath `
    --image_path $ImagePath `
    --ImageReader.camera_model PINHOLE `
    --ImageReader.single_camera 1 `
    --SiftExtraction.use_gpu 1

Write-Host "=== Step 2: Feature Matching ==="
colmap exhaustive_matcher `
    --database_path $DatabasePath `
    --SiftMatching.use_gpu 1

Write-Host "=== Step 3: Sparse Reconstruction ==="
colmap mapper `
    --database_path $DatabasePath `
    --image_path $ImagePath `
    --output_path $SparsePath

if ($SkipDense) {
    Write-Host "Sparse reconstruction saved to $SparsePath"
    exit 0
}

Write-Host "=== Step 4: Image Undistortion ==="
colmap image_undistorter `
    --image_path $ImagePath `
    --input_path (Join-Path $SparsePath "0") `
    --output_path $DensePath

Write-Host "=== Step 5: Dense Reconstruction ==="
colmap patch_match_stereo `
    --workspace_path $DensePath

Write-Host "=== Step 6: Stereo Fusion ==="
colmap stereo_fusion `
    --workspace_path $DensePath `
    --output_path (Join-Path $DensePath "fused.ply")

Write-Host "Results:"
Write-Host "  Sparse: $SparsePath"
Write-Host "  Dense:  $(Join-Path $DensePath 'fused.ply')"


param(
    [string]$PbrtRoot = "C:\Users\Arthur\Desktop\pbrt-v4",
    [int]$SPP = 512,
    [int]$Seed = 1
)

$ErrorActionPreference = "Stop"

$scene = Join-Path $PSScriptRoot "..\scene\broken-normals.pbrt"
$resultsDir = Join-Path $PSScriptRoot "..\results"

$oldExe = Join-Path $PbrtRoot "build-cuda124-scripted\Release\pbrt.exe"
$newExe = Join-Path $PbrtRoot "build-new\Release\pbrt.exe"

if (!(Test-Path $oldExe)) { throw "Old binary not found: $oldExe" }
if (!(Test-Path $newExe)) { throw "New binary not found: $newExe" }
if (!(Test-Path $scene)) { throw "Scene not found: $scene" }

New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

$oldOut = Join-Path $resultsDir "correct-old.png"
$newOut = Join-Path $resultsDir "correct-new.png"

Write-Host "[1/2] Render old/original version..."
& $oldExe --seed $Seed --spp $SPP --outfile $oldOut $scene

Write-Host "[2/2] Render modified version..."
& $newExe --seed $Seed --spp $SPP --outfile $newOut $scene

Write-Host "Done."
Write-Host "Old: $oldOut"
Write-Host "New: $newOut"

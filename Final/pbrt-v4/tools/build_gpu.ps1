param(
    [string]$BuildDir = "build-gpu",
    [string]$Configuration = "Release",
    [string]$OptiXPath = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0",
    [string]$CudaPath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$cmakeArgs = @(
    '-S', '.',
    '-B', $BuildDir,
    '-G', 'Visual Studio 17 2022',
    '-A', 'x64',
    "-T", "v143,cuda=$CudaPath",
    "-DPBRT_OPTIX_PATH=$OptiXPath"
)

Write-Host "[1/4] Configuring GPU build..."
& cmake @cmakeArgs

$vcxproj = Join-Path $repoRoot "$BuildDir\pbrt_lib.vcxproj"
if (!(Test-Path $vcxproj)) { throw "Missing $vcxproj" }

Write-Host "[2/4] Applying nvcc host-flag workaround..."
$content = Get-Content -Raw $vcxproj
$content = $content.Replace('/EHsc /MP', '-EHsc -MP')
Set-Content -Path $vcxproj -Value $content -Encoding UTF8

Write-Host "[3/4] Building pbrt_exe ($Configuration)..."
& cmake --build $BuildDir --config $Configuration --target pbrt_exe -j 8

Write-Host "[4/4] Done: $repoRoot\$BuildDir\$Configuration\pbrt.exe"

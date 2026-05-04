param(
    [string]$BuildDir = "build-cuda124-customtoolset",
    [string]$Configuration = "Release",
    [string]$Target = "build_uniform_grid_gpu"
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$cmakeArgs = @(
    '-S', '.',
    '-B', $BuildDir,
    '-G', 'Visual Studio 17 2022',
    '-A', 'x64',
    '-T', 'v143,version=14.38,cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4',
    '-DPBRT_OPTIX_PATH=C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0'
)

Write-Host "[1/3] Configuring with CUDA 12.4 + MSVC 14.38..."
& cmake @cmakeArgs

$vcxproj = Join-Path $repoRoot "$BuildDir\pbrt_lib.vcxproj"
if (!(Test-Path $vcxproj)) {
    throw "Could not find $vcxproj"
}

Write-Host "[2/3] Applying nvcc host-flag workaround to pbrt_lib.vcxproj..."
$content = Get-Content -Raw $vcxproj
$content = $content.Replace('/EHsc /MP', '-EHsc -MP')
Set-Content -Path $vcxproj -Value $content -Encoding UTF8

Write-Host "[3/3] Building $Target ($Configuration)..."
& cmake --build $BuildDir --config $Configuration --target $Target -j 8

Write-Host "Done. Binary: $repoRoot\$BuildDir\$Configuration\$Target.exe"
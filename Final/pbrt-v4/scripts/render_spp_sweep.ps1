param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Scene,
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][int[]]$SppValues
)

$Pbrt = if ($env:PBRT) { $env:PBRT } else { "build/Release/pbrt.exe" }
$ScenePath = Resolve-Path $Scene

foreach ($spp in $SppValues) {
    $out = "render_spp${spp}.exr"
    Write-Host "Rendering $ScenePath at $spp spp -> $out"
    & $Pbrt $ScenePath "--pixelsamples=$spp" "--outfile=$out"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

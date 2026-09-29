$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectDir

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name MyLabel `
    --icon "icon.ico" `
    --collect-all cv2 `
    --hidden-import numpy `
    --add-data "tracker_worker.py;." `
    --add-data "icon_app.png;." `
    main.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "Build complete: $projectDir\dist\MyLabel.exe"

$legacyModelTarget = Join-Path $projectDir "dist\model"
if (Test-Path -LiteralPath $legacyModelTarget -PathType Container) {
    Remove-Item -LiteralPath $legacyModelTarget -Recurse -Force
    Write-Host "Removed model files left by an older build: $legacyModelTarget"
}

Write-Host "MCITrack code and weights are external and are not included in this build."

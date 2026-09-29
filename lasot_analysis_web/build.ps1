$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $projectRoot
try {
    python -m PyInstaller --noconfirm --clean --onefile --windowed --name "LaSOT_Analysis" main.py
    Write-Host "Build complete: $projectRoot\dist\LaSOT_Analysis.exe"
}
finally {
    Pop-Location
}

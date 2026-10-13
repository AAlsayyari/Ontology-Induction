$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Run setup_environment.ps1 first to create the project environment.'
}

& $projectPython (Join-Path $PSScriptRoot 'web_ui.py') --open @args
exit $LASTEXITCODE

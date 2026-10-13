$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -c "import sys; assert sys.version_info[:2] == (3, 12), 'Install Python 3.12 first.'"
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required for the tested environment.' }
        & python -m venv .venv
    } else {
        throw 'Install Python 3.12, then run this script again.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}

& $projectPython -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12 for the tested environment.'"
if ($LASTEXITCODE -ne 0) { throw 'The project environment must use Python 3.12.' }

& $projectPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'Could not bootstrap pip in the project environment.' }

& $projectPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip.' }

& $projectPython -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Could not install CUDA PyTorch.' }

& $projectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install project dependencies.' }

& $projectPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable: check your NVIDIA driver and GPU.'; assert torch.cuda.is_bf16_supported(), 'A BF16-capable NVIDIA GPU is required for the default inference mode.'; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('BF16:', torch.cuda.is_bf16_supported())"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }

Write-Host 'Setup complete. Run .\start_web_ui.ps1 to open the app.'

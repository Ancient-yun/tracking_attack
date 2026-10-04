param(
    [switch]$DownloadAssets,
    [ValidateSet('cu128','cu121','cpu')][string]$TorchPlatform='cu128',
    [string]$PythonVersion='3.12',
    [int]$ClipLimit=0
)
$ErrorActionPreference='Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskTools = Join-Path $taskRoot '.tools'
$taskUv = Join-Path $taskTools 'uv\uv.exe'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $taskTools 'python'
$env:UV_CACHE_DIR = Join-Path $taskRoot '.cache\uv'
$env:HF_HOME = Join-Path $taskRoot '.cache\huggingface'
$env:TORCH_HOME = Join-Path $taskRoot '.cache\torch'
$env:MPLCONFIGDIR = Join-Path $taskRoot '.cache\matplotlib'
New-Item -ItemType Directory -Force -Path $taskTools,$env:UV_CACHE_DIR | Out-Null
if (!(Test-Path -LiteralPath $taskUv)) {
    $taskUvArchive = Join-Path $taskTools 'uv.zip'
    Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.12.22/uv-x86_64-pc-windows-msvc.zip' -OutFile $taskUvArchive
    Expand-Archive -LiteralPath $taskUvArchive -DestinationPath (Join-Path $taskTools 'uv') -Force
}
& $taskUv python install $PythonVersion --no-bin --no-registry
if ($LASTEXITCODE -ne 0) { throw 'Python download failed' }
$taskVenv = Join-Path $taskRoot '.venv'
$taskPython = Join-Path $taskVenv 'Scripts\python.exe'
if (!(Test-Path -LiteralPath $taskPython)) {
    & $taskUv venv --python $PythonVersion $taskVenv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
if ($TorchPlatform -eq 'cu121') {
    & $taskUv pip install --python $taskPython torch==2.5.1 torchvision==0.20.1 --index-url 'https://download.pytorch.org/whl/cu121'
} else {
    & $taskUv pip install --python $taskPython torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/$TorchPlatform"
}
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $taskUv pip install --python $taskPython -r (Join-Path $PSScriptRoot 'requirements-runtime.txt')
if ($LASTEXITCODE -ne 0) { throw 'Experiment dependency installation failed' }
& $taskUv pip install --python $taskPython --no-deps -e $taskRoot
if ($LASTEXITCODE -ne 0) { throw 'Experiment package installation failed' }
& $taskPython (Join-Path $PSScriptRoot 'verify_environment.py')
if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed' }
& $taskUv pip freeze --python $taskPython | Set-Content -Encoding UTF8 (Join-Path $taskRoot 'assets\environment-freeze.txt')
if ($DownloadAssets) {
    & $taskPython (Join-Path $PSScriptRoot 'download_assets.py') --limit $ClipLimit
    if ($LASTEXITCODE -ne 0) { throw 'Some asset downloads failed; inspect assets/download_manifest.json and rerun to resume' }
}
Write-Host "Ready: $taskPython"

param(
    [string]$RunName = ('po_validation_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskManifest = Join-Path $taskRoot 'manifests\po_validation.json'
if (!(Test-Path -LiteralPath $taskPython)) { throw 'Run scripts/bootstrap.ps1 -DownloadAssets first.' }
if ([IO.Path]::GetFileName($RunName) -ne $RunName -or $RunName -in '.', '..') {
    throw 'RunName must be a single folder name.'
}
if (!(Test-Path -LiteralPath $taskManifest)) {
    & $taskPython -m st4rtrack_pgd.make_manifest --datasets po_mini --count 2 --num-frames 16 --output $taskManifest
    if ($LASTEXITCODE -ne 0) { throw 'Manifest creation failed.' }
}
$taskArgs = @('-m', 'st4rtrack_pgd.run_attack', '--config', (Join-Path $taskRoot 'configs\validation.json'),
              '--manifest', $taskManifest, '--run-dir', (Join-Path $taskRoot "runs\$RunName"))
if ($Resume) { $taskArgs += '--resume' }
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { throw 'Validation incomplete; inspect failures.jsonl and summary.json.' }

param(
    [ValidateSet('Build','Smoke','Test','Validation','Pilot','Full','PGD','LossStudy','Shell')]
    [string]$Action = 'Smoke',
    [string]$RunName,
    [switch]$Resume,
    [string[]]$Objectives,
    [int]$Limit = 0,
    [int]$Steps = 0,
    [string]$LossConfig = 'loss_components_8clips_allframes.json',
    [string]$LossManifest = 'loss_components_8clips_allframes.json',
    [switch]$FailFast
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskCompose = Join-Path $taskRoot 'compose.yaml'
$taskComposeArgs = @('compose', '--project-directory', $taskRoot, '-f', $taskCompose)
if ($Action -eq 'LossStudy') {
    $taskComposeArgs += @('-f', (Join-Path $taskRoot 'compose.loss.yaml'))
}
if ($Limit -lt 0 -or $Steps -lt 0) {
    throw 'Limit and Steps must be positive when supplied.'
}
if (($Objectives -or $Limit -gt 0 -or $Steps -gt 0 -or $FailFast) -and $Action -ne 'LossStudy') {
    throw 'Objectives, Limit, Steps and FailFast are only supported for LossStudy.'
}
if (($PSBoundParameters.ContainsKey('LossConfig') -or $PSBoundParameters.ContainsKey('LossManifest')) -and $Action -ne 'LossStudy') {
    throw 'LossConfig and LossManifest are only supported for LossStudy.'
}
foreach ($taskLossFile in @($LossConfig, $LossManifest)) {
    if ([IO.Path]::GetFileName($taskLossFile) -ne $taskLossFile -or $taskLossFile -in '.', '..') {
        throw 'LossConfig and LossManifest must be filenames in configs or docker/manifests.'
    }
}
if ($Resume -and $Action -notin 'Validation','Pilot','Full','PGD','LossStudy') {
    throw 'Resume is only valid for an explicitly selected experiment action.'
}
foreach ($taskDir in @('runs/docker', '.cache/docker')) {
    New-Item -ItemType Directory -Path (Join-Path $taskRoot $taskDir) -Force | Out-Null
}

if ($Action -eq 'Build') {
    $taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $taskPython) {
        & $taskPython (Join-Path $taskRoot 'scripts\prepare_docker_manifests.py')
        if ($LASTEXITCODE -ne 0) { throw 'Docker manifest preparation failed.' }
    } else {
        foreach ($taskManifestName in @('po_validation.json','po_pilot.json','synthetic_full.json')) {
            if (!(Test-Path -LiteralPath (Join-Path $taskRoot "docker\manifests\$taskManifestName"))) {
                throw 'Prepared Docker manifests are missing. Run scripts/prepare_docker_manifests.py with any Python 3 interpreter.'
            }
        }
    }
    & docker @taskComposeArgs build experiment
} elseif ($Action -eq 'Smoke') {
    & docker @taskComposeArgs run --rm --no-deps experiment
} elseif ($Action -eq 'Test') {
    & docker @taskComposeArgs run --rm --no-deps experiment python -m pytest -q -p no:cacheprovider
} elseif ($Action -eq 'Shell') {
    & docker @taskComposeArgs run --rm --no-deps experiment bash
} else {
    if (!$RunName) {
        if ($Resume) { throw 'Specify RunName when resuming.' }
        $RunName = $Action.ToLowerInvariant() + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
    }
    if ([IO.Path]::GetFileName($RunName) -ne $RunName -or $RunName -in '.', '..') {
        throw 'RunName must be a single folder name.'
    }
    $taskSelection = @{
        Validation = @('validation.json','po_validation.json')
        Pilot = @('pilot.json','po_pilot.json')
        Full = @('full.json','synthetic_full.json')
        PGD = @('pgd_once.json','pgd_4clips.json')
        LossStudy = @($LossConfig,$LossManifest)
    }[$Action]
    $taskModule = if ($Action -eq 'LossStudy') { 'st4rtrack_pgd.run_component_study' } else { 'st4rtrack_pgd.run_attack' }
    $taskRunArgs = @('run','--rm','--no-deps','experiment','python','-m',$taskModule,
        '--config', "configs/$($taskSelection[0])",
        '--manifest', "docker/manifests/$($taskSelection[1])",
        '--run-dir', "runs/docker/$RunName")
    if ($Resume) { $taskRunArgs += '--resume' }
    if ($Objectives) { $taskRunArgs += @('--objectives') + $Objectives }
    if ($Limit -gt 0) { $taskRunArgs += @('--limit', [string]$Limit) }
    if ($Steps -gt 0) { $taskRunArgs += @('--steps', [string]$Steps) }
    if ($FailFast) { $taskRunArgs += '--fail-fast' }
    & docker @taskComposeArgs @taskRunArgs
}
if ($LASTEXITCODE -ne 0) { throw "Docker $Action failed with exit code $LASTEXITCODE." }

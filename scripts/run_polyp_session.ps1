param(
    [string]$ExperimentName = "Polyp_PVT_GCASCADE_img_size352bs4_Run2",
    [string]$ExperimentRoot = "",
    [string]$PythonExecutable = "",
    [switch]$DryRun,
    [Nullable[int]]$CompletedEpochForDryRun = $null
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($ExperimentRoot)) {
    $ExperimentRoot = Join-Path $projectRoot (Join-Path "experiments" $ExperimentName)
} elseif (-not [System.IO.Path]::IsPathRooted($ExperimentRoot)) {
    $ExperimentRoot = Join-Path $projectRoot $ExperimentRoot
}
$ExperimentRoot = [System.IO.Path]::GetFullPath($ExperimentRoot)

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    if ($env:CONDA_PREFIX -and (Split-Path $env:CONDA_PREFIX -Leaf) -ieq "gcascade") {
        $PythonExecutable = Join-Path $env:CONDA_PREFIX "python.exe"
    } else {
        $PythonExecutable = "python"
    }
}

$branch = (& git -C $projectRoot branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "abu-resumable-training") {
    throw "Expected Git branch abu-resumable-training; found '$branch'."
}

$requiredPaths = @(
    "train_polyp.py",
    "utils\training_checkpoint.py",
    "scripts\training_progress.py",
    "pretrained_pth\pvt\pvt_v2_b2.pth",
    "data\polyp\TrainDataset\images",
    "data\polyp\TrainDataset\masks",
    "data\polyp\TestDataset"
)
foreach ($relativePath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $relativePath))) {
        throw "Required path is missing: $relativePath"
    }
}

$environmentJson = & $PythonExecutable -c "import json, pathlib, sys, torch; print(json.dumps({'python':sys.version.split()[0], 'prefix':str(pathlib.Path(sys.prefix)), 'torch':torch.__version__, 'cuda':torch.cuda.is_available(), 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
if ($LASTEXITCODE -ne 0) {
    throw "Python environment verification failed."
}
$environment = $environmentJson | ConvertFrom-Json
if ((Split-Path $environment.prefix -Leaf) -ine "gcascade") {
    throw "Expected the gcascade environment; Python prefix is '$($environment.prefix)'."
}
if (-not $environment.cuda) {
    throw "CUDA is not available in the selected Python environment."
}

$checkpointPath = Join-Path $ExperimentRoot "checkpoints\training-resume.pth"
$modelDirectory = Join-Path $ExperimentRoot "models"
$progressArguments = @(
    "-m", "scripts.training_progress",
    "--checkpoint", $checkpointPath,
    "--json"
)
if ($null -ne $CompletedEpochForDryRun) {
    if (-not $DryRun) {
        throw "-CompletedEpochForDryRun is permitted only with -DryRun."
    }
    $progressArguments += @(
        "--completed-epoch",
        ([Convert]::ToInt32($CompletedEpochForDryRun)).ToString()
    )
}

Push-Location $projectRoot
try {
    $progressJson = & $PythonExecutable @progressArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read training progress."
    }
} finally {
    Pop-Location
}
$progress = $progressJson | ConvertFrom-Json

if ($progress.completed) {
    Write-Host "Experiment is already complete at epoch 199. No training started."
    exit 0
}

if (-not $progress.checkpoint_exists -and -not $progress.simulated -and (Test-Path -LiteralPath $ExperimentRoot)) {
    $existing = @(Get-ChildItem -LiteralPath $ExperimentRoot -Force -ErrorAction SilentlyContinue)
    if ($existing.Count -gt 0) {
        throw "Experiment directory is non-empty but has no valid resume checkpoint: $ExperimentRoot"
    }
}

Write-Host "Verified Python $($environment.python), PyTorch $($environment.torch), GPU $($environment.gpu)"
if ($progress.simulated) {
    Write-Host "SIMULATED PROGRESS FOR DRY RUN"
}
Write-Host "Session $($progress.session_number) of 40"
Write-Host "Epochs $($progress.start_epoch)-$($progress.end_epoch) (exclusive --epoch $($progress.exclusive_epoch_limit))"
Write-Host "Checkpoint: $checkpointPath"

$trainArguments = @(
    (Join-Path $projectRoot "train_polyp.py"),
    "--encoder", "PVT",
    "--skip_aggregation", "additive",
    "--epoch", [string]$progress.exclusive_epoch_limit,
    "--lr", "0.0001",
    "--optimizer", "AdamW",
    "--batchsize", "4",
    "--img_size", "352",
    "--clip", "0.5",
    "--decay_rate", "0.1",
    "--decay_epoch", "300",
    "--train_path", (Join-Path $projectRoot "data\polyp\TrainDataset"),
    "--test_path", (Join-Path $projectRoot "data\polyp\TestDataset"),
    "--experiment_name", $ExperimentName,
    "--train_save", $modelDirectory,
    "--save_resume_checkpoint", $checkpointPath
)
if ($progress.completed_epoch -gt 0) {
    $trainArguments += @("--resume", $checkpointPath)
}

$displayCommand = @($PythonExecutable) + $trainArguments | ForEach-Object {
    if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
}
Write-Host ($displayCommand -join " ")

if ($DryRun) {
    Write-Host "Dry run only: no directories created and no training started."
    Write-Host "After a completed session, shut down normally. Run this same script later to continue from the next checkpointed epoch."
    exit 0
}

New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $checkpointPath -Parent) | Out-Null
Push-Location $projectRoot
try {
    & $PythonExecutable @trainArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training process failed with exit code $LASTEXITCODE."
    }
    & $PythonExecutable -m scripts.training_progress --checkpoint $checkpointPath --expect-completed-epoch $progress.end_epoch
    if ($LASTEXITCODE -ne 0) {
        throw "Post-session checkpoint verification failed."
    }
} finally {
    Pop-Location
}

Write-Host "Session completed and checkpoint metadata verified at epoch $($progress.end_epoch)."
Write-Host "It is safe to stop the machine. Run this same script later to continue."

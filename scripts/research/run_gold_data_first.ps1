param(
    [string]$Dataset = "VietTung04/vbpl-vn",
    [string]$Split = "train",
    [int]$Limit = 0,
    [string]$GoldOutput = "data/gold/vbpl_gold.parquet",
    [string]$GoldArchiveOutput = "",
    [string]$AsOfDate = "",
    [int]$MonitorExpiringDays = 90,
    [ValidateSet("strict", "balanced", "broad")]
    [string]$Profile = "balanced",
    [int]$MinYearSoft = 0,
    [int]$TargetMinLaws = 0,
    [int]$BackfillMinYear = 0,
    [int]$MaxPrimaryRecords = 0,
    [switch]$AbortOnLowLaws
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "web_app\backend\requirements.txt"
$IngestScriptPath = Join-Path $ProjectRoot "scripts\load_vbpl_vn_dual_index.py"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[SETUP] Creating virtual environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv $VenvPath
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $VenvPath
    } else {
        throw "Python launcher not found. Install Python 3.12+ and retry."
    }
}

Write-Host "[SETUP] Installing dependencies..."
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -r $RequirementsPath --extra-index-url "https://download.pytorch.org/whl/cu124"

Write-Host "[RUN] Building gold dataset (data-only)..."
$argsList = @(
    $IngestScriptPath,
    "--dataset", $Dataset,
    "--split", $Split,
    "--monitor-expiring-days", $MonitorExpiringDays,
    "--profile", $Profile,
    "--gold-output", $GoldOutput,
    "--data-only"
)

if ($Limit -gt 0) {
    $argsList += @("--limit", $Limit)
}

if ($AsOfDate -ne "") {
    $argsList += @("--as-of-date", $AsOfDate)
}

if ($GoldArchiveOutput -ne "") {
    $argsList += @("--gold-archive-output", $GoldArchiveOutput)
}

if ($MinYearSoft -gt 0) {
    $argsList += @("--min-year-soft", $MinYearSoft)
}

if ($TargetMinLaws -gt 0) {
    $argsList += @("--target-min-laws", $TargetMinLaws)
}

if ($BackfillMinYear -gt 0) {
    $argsList += @("--backfill-min-year", $BackfillMinYear)
}

if ($MaxPrimaryRecords -gt 0) {
    $argsList += @("--max-primary-records", $MaxPrimaryRecords)
}

if ($AbortOnLowLaws.IsPresent) {
    $argsList += "--abort-on-low-laws"
}

& $VenvPython $argsList

Write-Host "[DONE] Gold data pipeline finished."

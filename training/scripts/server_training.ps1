[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Upload", "Bootstrap", "Start", "Status", "Logs", "Collect", "Tunnel")]
    [string]$Action,
    [string]$Server = "muhasadhabib@10.33.33.11",
    [string]$Base = "/home/muhasadhabib/tanaman-rimpang",
    [string]$SourceId,
    [string]$RunId = "rimpang-v1",
    [string]$JobId,
    [ValidateSet("train", "annotate")]
    [string]$Environment = "train",
    [string]$PythonModule,
    [string[]]$Arguments = @(),
    [string]$Gpu = "",
    [switch]$ResumeJob,
    [switch]$WithCuda,
    [ValidateSet("https://download.pytorch.org/whl/cu128", "https://pypi.org/simple")]
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [switch]$AllowIncomplete,
    [ValidateRange(1024, 65535)]
    [int]$Port = 8087,
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SshOptions = @("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30")

function Assert-Identifier([string]$Value, [string]$Name) {
    if ($Value -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$' -or $Value -in @(".", "..")) {
        throw "$Name must be a nonempty workspace identifier, not a path."
    }
}

function Quote-Shell([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Invoke-Remote([string]$Script) {
    $Script.Replace("`r", "") | & ssh @SshOptions -T $Server "tr -d '\r' | bash -s"
    if ($LASTEXITCODE -ne 0) { throw "Remote operation failed with exit code $LASTEXITCODE." }
}

function Invoke-RemotePython([string]$Code, [string[]]$Values) {
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
    $program = "import base64;exec(base64.b64decode('$encoded'))"
    $command = "python3.11 -B -c " + (Quote-Shell $program)
    foreach ($value in $Values) { $command += " " + (Quote-Shell $value) }
    Invoke-Remote ("set -eu`n" + $command)
}

if ($Server -notmatch '^[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+$') {
    throw "Use an explicit user@host SSH destination."
}
if ($Base -notmatch '^/home/[a-zA-Z0-9_.-]+/tanaman-rimpang$') {
    throw "Base must be an isolated /home/<user>/tanaman-rimpang workspace."
}
Assert-Identifier $RunId "RunId"
$LocalTransfers = Join-Path $Root "artifacts\transfers"

if ($Action -eq "Tunnel") {
    & ssh @SshOptions -o ExitOnForwardFailure=yes -N -L "127.0.0.1:${Port}:127.0.0.1:${Port}" $Server
    if ($LASTEXITCODE -ne 0) { throw "SSH tunnel stopped with exit code $LASTEXITCODE." }
    return
}

if ($Action -eq "Upload") {
    if (-not $SourceId) {
        $revision = (& git -C $Root rev-parse --short HEAD).Trim()
        if ($LASTEXITCODE -ne 0) { throw "Cannot determine the source revision." }
        $SourceId = $revision + "-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    }
    Assert-Identifier $SourceId "SourceId"
    New-Item -ItemType Directory -Force -Path $LocalTransfers | Out-Null
    $archive = Join-Path $LocalTransfers "$SourceId.tar.gz"
    if (Test-Path -LiteralPath $archive) { throw "Source archive already exists: $archive" }
    $relative = [Collections.Generic.List[string]]::new()
    foreach ($folder in @("training", "shared", "docs")) {
        foreach ($file in Get-ChildItem -LiteralPath (Join-Path $Root $folder) -Recurse -File) {
            if ($file.FullName -match '[\\/]__pycache__[\\/]' -or
                $file.Extension -notin @(".py", ".json", ".txt", ".md", ".ps1", ".sh", ".xml")) { continue }
            $relative.Add($file.FullName.Substring($Root.Length + 1))
        }
    }
    foreach ($file in @("README.md", "LICENSE", "data\README.md", "data\sources.json")) {
        $relative.Add($file)
    }
    $files = foreach ($name in $relative) {
        $file = Get-Item -LiteralPath (Join-Path $Root $name)
        [ordered]@{ path = $name.Replace("\", "/"); bytes = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    }
    $metadata = [ordered]@{
        schemaVersion = 1; sourceId = $SourceId
        gitRevision = (& git -C $Root rev-parse HEAD).Trim()
        createdAt = [DateTime]::UtcNow.ToString("o"); files = @($files)
    }
    $scratch = Join-Path $LocalTransfers (".source-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $scratch | Out-Null
    try {
        $manifestPath = Join-Path $scratch "source-manifest.json"
        [IO.File]::WriteAllText($manifestPath, ($metadata | ConvertTo-Json -Depth 8),
            [Text.UTF8Encoding]::new($false))
        $tarArguments = @("-czf", $archive, "-C", $Root) + @($relative.ToArray()) + @("-C", $scratch, "source-manifest.json")
        & tar @tarArguments
        if ($LASTEXITCODE -ne 0) { throw "Source packaging failed." }
    }
    finally {
        if (Test-Path -LiteralPath (Join-Path $scratch "source-manifest.json")) {
            Remove-Item -LiteralPath (Join-Path $scratch "source-manifest.json")
        }
        Remove-Item -LiteralPath $scratch
    }
    $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    Invoke-Remote ("set -eu`numask 077`nmkdir -p " + (Quote-Shell "$Base/transfers"))
    & scp @SshOptions $archive "${Server}:$Base/transfers/$SourceId.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw "Source upload failed; the local archive is retained." }
    $extract = @'
import hashlib, json, os, pathlib, sys, tarfile, tempfile
base, source_id, expected = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
archive = base / "transfers" / (source_id + ".tar.gz")
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
if sha(archive) != expected:
    raise ValueError("Source archive checksum mismatch")
destination = base / "code" / source_id
if destination.exists():
    raise ValueError("Source snapshot already exists; use a new SourceId")
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix=".source-", dir=destination.parent) as temporary:
    stage = pathlib.Path(temporary) / "source"
    stage.mkdir()
    with tarfile.open(archive, "r:gz") as bundle:
        seen = set()
        for member in bundle.getmembers():
            name = member.name
            path = pathlib.PurePosixPath(name)
            if path.is_absolute() or "\\" in name or ":" in name or ".." in path.parts:
                raise ValueError("Unsafe source archive path: " + name)
            if not member.isfile() or name in seen:
                raise ValueError("Unexpected source archive member: " + name)
            seen.add(name)
        bundle.extractall(stage, filter="data")
    manifest = json.loads((stage / "source-manifest.json").read_text())
    if manifest["sourceId"] != source_id:
        raise ValueError("Source identity mismatch")
    expected_files = {"source-manifest.json"}
    for item in manifest["files"]:
        path = stage / item["path"]
        if not path.resolve().is_relative_to(stage.resolve()) or not path.is_file():
            raise ValueError("Missing or unsafe source file")
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("Source file checksum mismatch: " + item["path"])
        expected_files.add(item["path"])
    if seen != expected_files:
        raise ValueError("Source archive inventory mismatch")
    stage.rename(destination)
for name in ("envs", "cache", "data", "runs", "services"):
    (base / name).mkdir(exist_ok=True)
os.chmod(base, 0o700)
print(json.dumps({"sourceId": source_id, "source": str(destination), "sha256": expected}))
'@
    Invoke-RemotePython $extract @($Base, $SourceId, $hash)
    return
}

Assert-Identifier $SourceId "SourceId"
$Source = "$Base/code/$SourceId"
$Python = "$Base/envs/$Environment-v1/bin/python"
$Run = "$Base/runs/$RunId"

if ($Action -eq "Bootstrap") {
    $setup = @"
set -eu
umask 077
test -d $(Quote-Shell $Source)
export PIP_CACHE_DIR=$(Quote-Shell "$Base/cache/pip")
export XDG_CACHE_HOME=$(Quote-Shell "$Base/cache")
if ! test -x $(Quote-Shell "$Base/envs/train-v1/bin/python"); then python3.11 -m venv $(Quote-Shell "$Base/envs/train-v1"); fi
$(Quote-Shell "$Base/envs/train-v1/bin/python") -m pip install --disable-pip-version-check --no-input -r $(Quote-Shell "$Source/training/requirements-data.txt")
if ! test -x $(Quote-Shell "$Base/envs/annotate-v1/bin/python"); then python3.11 -m venv $(Quote-Shell "$Base/envs/annotate-v1"); fi
$(Quote-Shell "$Base/envs/annotate-v1/bin/python") -m pip install --disable-pip-version-check --no-input -r $(Quote-Shell "$Source/training/requirements-annotations.txt")
"@
    if ($WithCuda) {
        $setup += @"

$(Quote-Shell "$Base/envs/train-v1/bin/python") -m pip install --disable-pip-version-check --no-input torch==2.8.0 torchvision==0.23.0 --index-url $(Quote-Shell $TorchIndex)
$(Quote-Shell "$Base/envs/train-v1/bin/python") -m pip install --disable-pip-version-check --no-input -r $(Quote-Shell "$Source/training/requirements.txt")
CUDA_VISIBLE_DEVICES=0 $(Quote-Shell "$Base/envs/train-v1/bin/python") -c 'import torch; assert torch.cuda.is_available() and torch.version.cuda == "12.8" and torch.cuda.device_count() == 1, "CUDA 12.8 / GPU 0 runtime unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
"@
    }
    Invoke-Remote $setup
    return
}

if ($Action -in @("Start", "Status", "Logs")) {
    Assert-Identifier $JobId "JobId"
    $Job = "$Run/jobs/$JobId"
}
if ($Action -eq "Start") {
    if ($PythonModule -notmatch '^training\.scripts\.[a-z_]+$') {
        throw "Start accepts an explicit training.scripts module."
    }
    if ($Gpu -and $Gpu -notmatch '^(0|GPU-[a-fA-F0-9-]+)$') {
        throw "Only GPU 0 or its explicitly reviewed GPU UUID may be selected."
    }
    $inspect = "cd " + (Quote-Shell $Source) + "`nif test -f " + (Quote-Shell "$Job/status.json") +
        "; then " + (Quote-Shell $Python) + " -B -m training.scripts.server_job status --job-dir " +
        (Quote-Shell $Job) + "; else printf 'null\n'; fi"
    $previous = ((Invoke-Remote ("set -eu`n" + $inspect)) -join "`n") | ConvertFrom-Json
    if ($previous -and (-not $ResumeJob -or $previous.status -notin @("failed", "interrupted"))) {
        throw "This job already exists. Use a new JobId, or -ResumeJob only for failed/interrupted jobs."
    }
    if ($ResumeJob -and -not $previous) { throw "There is no previous job to resume." }
    $expectedAttempt = 1
    if ($previous) { $expectedAttempt = [int]$previous.attempt + 1 }
    $session = "rimpang-$RunId-$JobId"
    $jobArguments = @($Python, "-B", "-u", "-m", "training.scripts.server_job", "run",
        "--base", $Base, "--cwd", $Source, "--job-dir", $Job, "--stage", $JobId,
        "--gpu", $Gpu)
    if ($ResumeJob) { $jobArguments += "--resume-job" }
    $jobArguments += @("--", $Python, "-B", "-u", "-m", $PythonModule) + $Arguments
    $command = "cd " + (Quote-Shell $Source) + "; exec " + (($jobArguments | ForEach-Object { Quote-Shell $_ }) -join " ")
    Invoke-Remote ("set -eu`numask 077`ntest -x " + (Quote-Shell $Python) +
        "`ntmux new-session -d -s " + (Quote-Shell $session) + " " + (Quote-Shell $command))
    Start-Sleep -Seconds 2
    $state = ((Invoke-Remote ("set -eu`ncd " + (Quote-Shell $Source) + "`n" + (Quote-Shell $Python) +
        " -m training.scripts.server_job status --job-dir " + (Quote-Shell $Job))) -join "`n") | ConvertFrom-Json
    $state | ConvertTo-Json -Depth 8
    if ($state.attempt -ne $expectedAttempt -or $state.status -notin @("running", "completed")) {
        throw "Job did not start successfully. Inspect $Job/stdout.log before retrying."
    }
    return
}
if ($Action -eq "Status") {
    Invoke-Remote ("set -eu`ncd " + (Quote-Shell $Source) + "`n" + (Quote-Shell $Python) +
        " -m training.scripts.server_job status --job-dir " + (Quote-Shell $Job))
    return
}
if ($Action -eq "Logs") {
    Invoke-Remote ("set -eu`ntail -n 70 " + (Quote-Shell "$Job/stdout.log"))
    return
}
if ($Action -eq "Collect") {
    $command = "cd " + (Quote-Shell $Source) + "`n" + (Quote-Shell "$Base/envs/train-v1/bin/python") +
        " -m training.scripts.package_artifacts pack --run " + (Quote-Shell $Run) +
        " --output " + (Quote-Shell "$Base/transfers")
    if ($AllowIncomplete) { $command += " --allow-incomplete" }
    $result = (Invoke-Remote ("set -eu`n" + $command)) -join "`n" | ConvertFrom-Json
    if ($result.archive -notmatch ('^' + [regex]::Escape("$Base/transfers/") + '[a-zA-Z0-9_.-]+\.tar\.gz$') -or
        $result.sha256 -notmatch '^[a-f0-9]{64}$') { throw "Invalid result transfer metadata." }
    New-Item -ItemType Directory -Force -Path $LocalTransfers | Out-Null
    $archive = Join-Path $LocalTransfers ([IO.Path]::GetFileName($result.archive))
    & scp @SshOptions "${Server}:$($result.archive)" $archive
    if ($LASTEXITCODE -ne 0) { throw "Result transfer failed; retry from the recorded remote archive." }
    if (-not $Destination) { $Destination = Join-Path $Root "artifacts\$RunId" }
    Push-Location $Root
    try {
        & python -B -m training.scripts.package_artifacts receive --archive $archive --output $Destination --sha256 $result.sha256
        if ($LASTEXITCODE -ne 0) { throw "Result verification failed; no active model was changed." }
    }
    finally { Pop-Location }
}

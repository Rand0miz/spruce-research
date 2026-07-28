param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$workspacePath = (Resolve-Path -LiteralPath $Workspace).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stage = Join-Path $workspacePath ".colab_zip_staging_$stamp"
$rebuiltZip = Join-Path $workspacePath "spruce_colab_train_source.rebuild.zip"
$currentZip = Join-Path $workspacePath "spruce_colab_train_source.zip"
$backupZip = Join-Path $workspacePath "spruce_colab_train_source.pre_rebuild_$stamp.zip"

foreach ($target in @($stage, $rebuiltZip, $currentZip, $backupZip)) {
    $full = [IO.Path]::GetFullPath($target)
    $prefix = $workspacePath + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing out-of-workspace target: $full"
    }
}
if (-not (Test-Path -LiteralPath $currentZip)) {
    throw "Missing existing source archive: $currentZip"
}
$workspacePrefix = $workspacePath + [IO.Path]::DirectorySeparatorChar
Get-ChildItem -LiteralPath $workspacePath -Directory `
    -Filter ".colab_zip_staging_*" |
    ForEach-Object {
        $staleStage = [IO.Path]::GetFullPath($_.FullName)
        if (-not $staleStage.StartsWith(
                $workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing unsafe stale-staging cleanup: $staleStage"
        }
        Remove-Item -LiteralPath $staleStage -Recurse -Force
    }
if (Test-Path -LiteralPath $stage) {
    throw "Staging path unexpectedly exists: $stage"
}
if (Test-Path -LiteralPath $rebuiltZip) {
    Remove-Item -LiteralPath $rebuiltZip -Force
}

New-Item -ItemType Directory -Path $stage | Out-Null
Expand-Archive -LiteralPath $currentZip -DestinationPath $stage -Force

# Preserve the prior archive's benchmark evidence, but refresh every source
# tree from the current workspace so removed files do not linger.
$replaceDirs = @(
    "configs",
    "eval",
    "interfaces",
    "kernels",
    "scripts",
    "selector",
    "sparse",
    "teacher",
    "tests",
    "colab"
)
foreach ($dir in $replaceDirs) {
    $stageDir = Join-Path $stage $dir
    if (Test-Path -LiteralPath $stageDir) {
        $resolved = [IO.Path]::GetFullPath($stageDir)
        $stagePrefix = $stage + [IO.Path]::DirectorySeparatorChar
        if (-not $resolved.StartsWith(
                $stagePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing unsafe staging delete: $resolved"
        }
        Remove-Item -LiteralPath $stageDir -Recurse -Force
    }
}

$benchmarkStage = Join-Path $stage "benchmarks"
Get-ChildItem -LiteralPath $benchmarkStage -File | Remove-Item -Force

$sourceFiles = @()
$sourceFiles += Get-ChildItem `
    -LiteralPath (Join-Path $workspacePath "benchmarks") -File |
    Where-Object { $_.Extension -in @(".py", ".json", ".md") }
foreach ($dir in $replaceDirs) {
    $sourceDir = Join-Path $workspacePath $dir
    if (Test-Path -LiteralPath $sourceDir) {
        $sourceFiles += Get-ChildItem -LiteralPath $sourceDir -Recurse -File |
            Where-Object {
                $_.Extension -in @(
                    ".py", ".json", ".md", ".ipynb", ".ps1"
                ) -and
                $_.FullName -notmatch "[\\/]__pycache__[\\/]"
            }
    }
}
foreach ($file in $sourceFiles) {
    $relative = [IO.Path]::GetRelativePath($workspacePath, $file.FullName)
    $destination = Join-Path $stage $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) `
        -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
}

# LOG.md records the archive hash after packaging, so embedding it would make
# that hash self-referential and inevitably stale. Keep the authoritative log
# in the workspace and exclude any root copy inherited from an older archive.
$stagedLog = Join-Path $stage "LOG.md"
if (Test-Path -LiteralPath $stagedLog) {
    $resolvedLog = [IO.Path]::GetFullPath($stagedLog)
    $stagePrefix = $stage + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedLog.StartsWith(
            $stagePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing unsafe staged-log delete: $resolvedLog"
    }
    Remove-Item -LiteralPath $stagedLog -Force
}

Compress-Archive -Path (Join-Path $stage "*") `
    -DestinationPath $rebuiltZip -CompressionLevel Optimal

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($rebuiltZip)
try {
    $names = @(
        $archive.Entries |
            ForEach-Object { $_.FullName.Replace("\", "/") }
    )
    $required = @(
        "benchmarks/all_blocks_equivalence.py",
        "benchmarks/diagnose_residual_summaries.py",
        "benchmarks/evaluate_evidence_compiler.py",
        "benchmarks/benchmark_pre_qwen_e2e.py",
        "benchmarks/benchmark_pre_qwen_natural_yarn_length.py",
        "benchmarks/report_pre_qwen_natural_yarn.py",
        "benchmarks/run_pre_qwen_natural_yarn_suite.py",
        "interfaces/evidence_compiler.py",
        "interfaces/pre_qwen_selector_spec.md",
        "interfaces/residual_summaries.py",
        "selector/evidence.py",
        "selector/pre_qwen.py",
        "sparse/summaries.py",
        "sparse/attention.py",
        "tests/test_all_blocks_equivalence.py",
        "tests/test_evidence_compiler.py",
        "tests/test_evidence_compiler_benchmark.py",
        "tests/test_pre_qwen_benchmark.py",
        "tests/test_pre_qwen_natural_yarn_suite.py",
        "tests/test_pre_qwen_selector.py",
        "tests/test_residual_diagnostics.py",
        "tests/test_residual_summaries.py",
        "tests/test_summary_pooling.py",
        "colab/run_all_blocks_equivalence.ipynb",
        "colab/run_evidence_compiler_gate.ipynb",
        "colab/run_pre_qwen_beam16_followup.ipynb",
        "colab/run_pre_qwen_e2e.ipynb",
        "colab/run_pre_qwen_natural_yarn_paper.ipynb",
        "colab/run_residual_summary_gate.ipynb",
        "scripts/prompt_banks/natural_paper_untouched.json",
        "colab/rebuild_source_archive.ps1"
    )
    foreach ($entry in $required) {
        if ($entry -notin $names) {
            throw "Rebuilt archive is missing $entry"
        }
    }
    if ($names | Where-Object { $_ -match "__pycache__|\.pyc$" }) {
        throw "Rebuilt archive contains Python cache files"
    }
    $entryCount = $archive.Entries.Count
}
finally {
    $archive.Dispose()
}

Copy-Item -LiteralPath $currentZip -Destination $backupZip
Move-Item -LiteralPath $rebuiltZip -Destination $currentZip -Force
$hash = (Get-FileHash -LiteralPath $currentZip -Algorithm SHA256).Hash
$size = (Get-Item -LiteralPath $currentZip).Length

$resolvedStage = [IO.Path]::GetFullPath($stage)
$expectedPrefix = $workspacePath + [IO.Path]::DirectorySeparatorChar
if (-not $resolvedStage.StartsWith(
        $expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing unsafe staging cleanup: $resolvedStage"
}
Remove-Item -LiteralPath $stage -Recurse -Force

[pscustomobject]@{
    Archive = $currentZip
    Backup = $backupZip
    Entries = $entryCount
    Bytes = $size
    SHA256 = $hash
}

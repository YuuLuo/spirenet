# host\make-dist.ps1 -Label <label> [-NoBuild]
#
# Builds the integrated host and makes an IMMUTABLE deployment directory
#
#     host\dist\<yyyymmdd-HHmmss>-<label>\
#         SpireHost.exe + every assembly (+ .pdb, .deps.json, .runtimeconfig.json)
#         vocab\vocab.json, vocab\catalog.json
#         FIDELITY.md, DRIVER-API.md, README.md
#         dist.json      label, stamp, DRIVER-API version, SHA256 of sts2.dll and of every shipped file, the production
#                        patch list with its hash, vocabulary and catalog hashes, how the executable finds the game
#
# Training and verification run only from dist directories, so later rebuilds never disturb running processes.
# A dist directory is never modified after creation: every start of its executable verifies it against dist.json
# (README.md, "Deploy"). The game is not copied: the executable finds host\game by walking up from its own directory
# (or --game-dir / SPIREHOST_GAME_DIR). On a failure after the directory was created it is left behind WITHOUT a
# dist.json (it is then not a dist) and has to be deleted by hand.
param(
    [Parameter(Mandatory = $true)][string]$Label,
    [switch]$NoBuild
)
$ErrorActionPreference = "Stop"
if ($Label -notmatch '^[A-Za-z0-9._-]+$') { throw "make-dist: the label may contain letters, digits, '.', '_' and '-' only" }
$hostDir = $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$out = Join-Path $hostDir "out\bin\SpireHost\release"
$dist = Join-Path $hostDir "dist\$stamp-$Label"
if (Test-Path $dist) { throw "make-dist: $dist exists" }

Push-Location $hostDir
try {
    if (-not $NoBuild) {
        Write-Output "make-dist: building ..."
        dotnet build SpireHost\SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false -v:q -nologo
        if ($LASTEXITCODE -ne 0) { throw "make-dist: build failed" }
    }
    if (-not (Test-Path (Join-Path $out "SpireHost.exe"))) { throw "make-dist: no build output in $out" }

    # A build whose ledger does not describe its patch set is not shippable.
    Write-Output "make-dist: checking registry == ledger ..."
    & (Join-Path $out "SpireHost.exe") --check-ledger | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "make-dist: registry != ledger (SpireHost --check-ledger exit $LASTEXITCODE); no dist made" }

    New-Item -ItemType Directory -Path $dist | Out-Null
    Copy-Item -Path (Join-Path $out "*") -Destination $dist -Recurse
    Copy-Item -Path (Join-Path $hostDir "FIDELITY.md"), (Join-Path $hostDir "DRIVER-API.md"), (Join-Path $hostDir "README.md") -Destination $dist

    # The copied executable describes itself: only it knows its patch set and vocabulary.
    Write-Output "make-dist: writing dist.json ..."
    & (Join-Path $dist "SpireHost.exe") --write-dist-manifest --label $Label --stamp $stamp | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "make-dist: the manifest was not written (exit $LASTEXITCODE); an INCOMPLETE directory is left at $dist" }

    # read-only from here on
    Get-ChildItem -Path $dist -Recurse -File | ForEach-Object { $_.IsReadOnly = $true }

    Write-Output "make-dist: verifying ..."
    & (Join-Path $dist "SpireHost.exe") --verify-dist | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "make-dist: verification of the new dist failed (exit $LASTEXITCODE): $dist" }

    Write-Output "dist: $dist"
    Write-Output "exe:  $dist\SpireHost.exe"
}
finally {
    Pop-Location
}

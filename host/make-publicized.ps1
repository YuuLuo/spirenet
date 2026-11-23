# Regenerates host\publicized\sts2.dll (compile-time reference only) from the pinned host\game\sts2.dll.
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
dotnet run --project "$here\tools\Publicizer\Publicizer.csproj" -c Release -- `
  "$here\game\sts2.dll" "$here\publicized\sts2.dll" `
  --expect-sha256 62ecbe91093e1fa862de4329aac865209dcb297804caf0f0520d92d8c0b4a9c3
if ($LASTEXITCODE -ne 0) { throw "Publicizer failed with exit code $LASTEXITCODE" }

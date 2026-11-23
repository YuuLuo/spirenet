#!/usr/bin/env bash
# Regenerates host/publicized/sts2.dll (compile-time reference only) from the pinned host/game/sts2.dll.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dotnet run --project "$here/tools/Publicizer/Publicizer.csproj" -c Release -- \
  "$here/game/sts2.dll" "$here/publicized/sts2.dll" \
  --expect-sha256 62ecbe91093e1fa862de4329aac865209dcb297804caf0f0520d92d8c0b4a9c3

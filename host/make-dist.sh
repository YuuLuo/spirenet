#!/usr/bin/env bash
# host/make-dist.sh <label> [--no-build]
#
# Builds the integrated host and makes an IMMUTABLE deployment directory
#
#     host/dist/<yyyymmdd-HHmmss>-<label>/
#         SpireHost.exe + every assembly (+ .pdb, .deps.json, .runtimeconfig.json)
#         vocab/vocab.json, vocab/catalog.json
#         FIDELITY.md, DRIVER-API.md, README.md
#         dist.json      label, stamp, DRIVER-API version, SHA256 of sts2.dll and of every shipped file, the production
#                        patch list with its hash, vocabulary and catalog hashes, how the executable finds the game
#
# Training and verification run only from dist directories, so later rebuilds never disturb running processes.
# A dist directory is never modified after creation: every start of its executable verifies it against dist.json
# (README.md, "Deploy"). The game is not copied: the executable finds host/game by walking up from its own directory
# (or --game-dir / SPIREHOST_GAME_DIR). On a failure after the directory was created it is left behind WITHOUT a
# dist.json (it is then not a dist) and has to be deleted by hand.
set -eu
HOST="$(cd "$(dirname "$0")" && pwd)"
LABEL="${1:?usage: make-dist.sh <label> [--no-build]}"
NOBUILD="${2:-}"
case "$LABEL" in
  *[!A-Za-z0-9._-]*) echo "make-dist: the label may contain letters, digits, '.', '_' and '-' only" >&2; exit 1 ;;
esac
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HOST/out/bin/SpireHost/release"
DIST="$HOST/dist/$STAMP-$LABEL"
if [ -e "$DIST" ]; then echo "make-dist: $DIST exists" >&2; exit 1; fi
cd "$HOST"

if [ "$NOBUILD" != "--no-build" ]; then
  echo "make-dist: building ..."
  dotnet build SpireHost/SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false -v:q -nologo
fi
# The apphost: SpireHost.exe on Windows, SpireHost on Linux.
EXE=SpireHost.exe
if [ "$(uname -s)" = Linux ]; then EXE=SpireHost; fi
if [ ! -f "$OUT/$EXE" ]; then echo "make-dist: no build output in $OUT" >&2; exit 1; fi

# A build whose ledger does not describe its patch set is not shippable (exit 4 from --check-ledger stops here).
echo "make-dist: checking registry == ledger ..."
"$OUT/$EXE" --check-ledger > /dev/null

mkdir -p "$DIST"
cp -r "$OUT/." "$DIST/"
cp "$HOST/FIDELITY.md" "$HOST/DRIVER-API.md" "$HOST/README.md" "$DIST/"

# The copied executable describes itself: only it knows its patch set and vocabulary.
echo "make-dist: writing dist.json ..."
"$DIST/$EXE" --write-dist-manifest --label "$LABEL" --stamp "$STAMP" > /dev/null

# read-only from here on
chmod -R a-w "$DIST"

echo "make-dist: verifying ..."
"$DIST/$EXE" --verify-dist > /dev/null

echo "dist: $DIST"
echo "exe:  $DIST/$EXE"

#!/usr/bin/env python3
"""Builds host/game/ from your own Slay the Spire 2 v0.113.1 install.

    python host/tools/prepare_game_dir.py <STS2 install directory> [--out DIR]

The install directory is the one that contains SlayTheSpire2.pck, release_info.json and the data_sts2_<platform>
directory (Steam: Library -> Slay the Spire 2 -> Manage -> Browse local files). Output (default: host/game/):

    sts2.dll + the game's other managed assemblies the host references or resolves at run time (copied unchanged)
    release_info.json                       (copied unchanged)
    fair-localization-eng.json              the English localisation tables, assembled from the game's own
                                            res://localization/eng/*.json inside SlayTheSpire2.pck

The host refuses any other sts2.dll (SHA256 pin, SpireHost.Core/Boot/GameAssemblies.cs) and any other bundle
(SpireHost.Core/Boot/Bootstraps.cs, LocalizationBootstrap); this script checks both pins before it writes anything.
Standard library only; nothing is downloaded and nothing from the game is part of the repository.
"""

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
from collections import Counter
from pathlib import Path

PINNED_VERSION = "v0.113.1"
PINNED_STS2_SHA256 = "62ecbe91093e1fa862de4329aac865209dcb297804caf0f0520d92d8c0b4a9c3"
# LocalizationBootstrap accepts this bundle (the one this script builds from the v0.113.1 SlayTheSpire2.pck).
PINNED_BUNDLE_SHA256 = "7ed2a457ca4073b5f7b2b8abe497b53da428e63a02b838aefb6230106ccb669e"
BUNDLE_NAME = "fair-localization-eng.json"

# The game's own managed assemblies the host needs (the .NET runtime assemblies of the data directory are not copied:
# the host runs on the installed .NET 9 runtime).
ASSEMBLIES = [
    "0Harmony.dll",
    "GodotSharp.dll",
    "JetBrains.Annotations.dll",
    "MonoMod.Backports.dll",
    "MonoMod.ILHelpers.dll",
    "Sentry.Godot.dll",
    "Sentry.dll",
    "SharpGen.Runtime.COM.dll",
    "SharpGen.Runtime.dll",
    "SmartFormat.ZString.dll",
    "SmartFormat.dll",
    "Steamworks.NET.dll",
    "System.IO.Hashing.dll",
    "Vortice.DXGI.dll",
    "Vortice.DirectX.dll",
    "Vortice.Mathematics.dll",
    "sts2.dll",
]

# Godot 4.5 pack format 3: header "GDPC", format, engine major / minor / patch, pack flags, file base, directory offset.
PCK_HEADER = struct.Struct("<4sIIIIIQQ")
PCK_FORMAT = 3
PCK_REL_FILEBASE = 2
LOC_ENTRY = re.compile(r"localization/eng/([a-z0-9_]+)\.json")


class PrepareError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_data_dir(install: Path) -> Path:
    dirs = sorted(d for d in install.glob("data_sts2_*") if d.is_dir())
    if len(dirs) != 1:
        raise PrepareError(f"expected exactly one data_sts2_<platform> directory in {install}, found {[d.name for d in dirs]}")
    return dirs[0]


def read_loc_entries(pck: Path) -> dict[str, bytes]:
    """The contents of every localization/eng/<table>.json in the pack, by table name (MD5 checked against the pack)."""
    entries: dict[str, bytes] = {}
    with pck.open("rb") as f:
        magic, fmt, major, minor, patch, flags, file_base, dir_offset = PCK_HEADER.unpack(f.read(PCK_HEADER.size))
        if magic != b"GDPC" or fmt != PCK_FORMAT or flags != PCK_REL_FILEBASE:
            raise PrepareError(f"{pck}: not an unencrypted Godot pack of format {PCK_FORMAT} "
                               f"(magic {magic!r}, format {fmt}, flags {flags}, engine {major}.{minor}.{patch})")
        f.seek(dir_offset)
        (count,) = struct.unpack("<I", f.read(4))
        located = []
        for _ in range(count):
            (length,) = struct.unpack("<I", f.read(4))
            path = f.read(length).rstrip(b"\0").decode("utf-8")
            offset, size, md5, entry_flags = struct.unpack("<QQ16sI", f.read(36))
            m = LOC_ENTRY.fullmatch(path)
            if m:
                if entry_flags != 0:
                    raise PrepareError(f"{pck}: {path} is encrypted or marked removed (flags {entry_flags})")
                located.append((m.group(1), path, file_base + offset, size, md5))
        for table, path, offset, size, md5 in located:
            f.seek(offset)
            data = f.read(size)
            if hashlib.md5(data).digest() != md5:
                raise PrepareError(f"{pck}: {path} does not match its MD5 in the pack directory")
            entries[table] = data
    if not entries:
        raise PrepareError(f"{pck}: no localization/eng/*.json entries")
    return entries


def no_duplicates(pairs: list) -> dict:
    d = dict(pairs)
    if len(d) != len(pairs):
        dups = sorted(k for k, n in Counter(k for k, _ in pairs).items() if n > 1)
        raise PrepareError("duplicate keys in a localisation table: " + ", ".join(dups))
    return d


def build_bundle(entries: dict[str, bytes], release_info: dict) -> bytes:
    tables = {}
    for table, data in entries.items():
        t = json.loads(data.decode("utf-8"), object_pairs_hook=no_duplicates)
        bad = [k for k, v in t.items() if not isinstance(v, str)]
        if bad:
            raise PrepareError(f"localization/eng/{table}.json: non-string values for {bad[:5]}")
        tables[table] = t
    bundle = {"commit": release_info["commit"], "language": "eng", "tables": tables, "version": release_info["version"]}
    return (json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("install", type=Path, help="Slay the Spire 2 v0.113.1 install directory (contains SlayTheSpire2.pck)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "game",
                    help="output directory, must not exist or be empty (default: host/game)")
    args = ap.parse_args()
    install: Path = args.install.resolve()
    out: Path = args.out.resolve()

    if out.exists() and any(out.iterdir()):
        raise PrepareError(f"{out} exists and is not empty; delete it first")

    data_dir = find_data_dir(install)
    sts2_sha = sha256_file(data_dir / "sts2.dll")
    if sts2_sha != PINNED_STS2_SHA256:
        raise PrepareError(f"{data_dir / 'sts2.dll'} has SHA256 {sts2_sha}; the host needs the unmodified {PINNED_VERSION} "
                           f"({PINNED_STS2_SHA256}). Install exactly that game version.")
    release_info_path = install / "release_info.json"
    release_info = json.loads(release_info_path.read_text(encoding="utf-8"))
    if release_info["version"] != PINNED_VERSION:
        raise PrepareError(f"{release_info_path}: version {release_info['version']}, expected {PINNED_VERSION}")

    bundle = build_bundle(read_loc_entries(install / "SlayTheSpire2.pck"), release_info)
    bundle_sha = hashlib.sha256(bundle).hexdigest()
    if bundle_sha != PINNED_BUNDLE_SHA256:
        raise PrepareError(f"the localisation bundle built from {install / 'SlayTheSpire2.pck'} has SHA256 {bundle_sha}, "
                           f"expected {PINNED_BUNDLE_SHA256}")

    out.mkdir(parents=True, exist_ok=True)
    for name in ASSEMBLIES:
        shutil.copyfile(data_dir / name, out / name)
    shutil.copyfile(release_info_path, out / "release_info.json")
    (out / BUNDLE_NAME).write_bytes(bundle)

    for name in sorted(p.name for p in out.iterdir()):
        print(f"{sha256_file(out / name)}  {name}")
    print(f"game directory ready: {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PrepareError as e:
        print("prepare_game_dir: " + str(e), file=sys.stderr)
        sys.exit(1)

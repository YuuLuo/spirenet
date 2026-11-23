"""Run-config migration for the calc features (notes/calc/contract.md 4): a copy of a run's config (the
supervisor's ``config.json`` or a TOML) with ``tokenizer.calc = true``, ``actor.reset_flags |= 512`` and the host
launch spec of the calc dist. Never overwrites; the source is not touched. The forecast setting is kept.

    python -B scripts/migrate_calc_config.py --config runs/X/config.json --host-config configs/linux_host_calc.toml \
        --output runs/X/config.calc.json

Then start the run once with ``--config <output> --resume --migrate-calc`` (later resumes without the flag: the
supervisor persists the config with ``migrate_calc = false``).
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from spire.host import real_host_spec
from spire.schema import RESET_FLAGS_DEFAULT, ResetFlag
from spire.train import TrainConfig, build, check_forecast_flags
from spire.vocab import Vocab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--host-config", type=Path, required=True, help="launch spec of the calc dist")
    ap.add_argument("--old-host-config", type=Path, help="default: the config's hosts.real_config")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), "output exists; never overwrites"
    text = args.config.read_text(encoding="utf-8")
    raw = tomllib.loads(text) if args.config.suffix == ".toml" else json.loads(text)
    tok = raw.setdefault("tokenizer", {})
    assert not tok.get("calc", False), "already a calc config"
    actor = raw.setdefault("actor", {})
    flags = int(actor.get("reset_flags", RESET_FLAGS_DEFAULT))
    assert not flags & int(ResetFlag.CALC) and flags & int(ResetFlag.TARGET_PREVIEWS)
    hosts = raw.setdefault("hosts", {})
    old = real_host_spec(args.old_host_config or hosts["real_config"])
    new = real_host_spec(args.host_config)
    # the calc dist speaks the same vocabulary / catalog (contract 4: the files must hash the same)
    assert Vocab.load(old.vocab).hash == Vocab.load(new.vocab).hash
    assert old.catalog.read_bytes() == new.catalog.read_bytes()
    tok["calc"] = True
    actor["reset_flags"] = flags | int(ResetFlag.CALC)
    hosts["real_config"] = str(args.host_config.resolve())
    for key in ("vocab", "catalog"):  # explicit paths move to the new dist's files; "host" follows real_config
        if raw.get(key) not in (None, "host", "synthetic", "none"):
            raw[key] = str(new.vocab if key == "vocab" else new.catalog)
    for key in ("resume_from", "migrate_public_history", "migrate_seat_symmetry", "migrate_coop_features",
                "migrate_forecast", "migrate_calc", "reset_seat_symmetry_optimizer"):
        raw.pop(key, None)
    check_forecast_flags(build(TrainConfig, raw))  # the result must be a valid config
    args.output.write_text(json.dumps(raw, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tokenizer.calc": True, "actor.reset_flags": actor["reset_flags"],
                      "hosts.real_config": hosts["real_config"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

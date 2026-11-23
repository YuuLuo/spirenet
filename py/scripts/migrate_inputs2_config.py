"""Run-config migration for inputs v2a (notes/inputs2/v2a.md 3): a copy of a run's config (the supervisor's
``config.json`` or a TOML) with ``tokenizer.map_tokens = "no_combat"``, ``route_summary = true``,
``second_boss_table = true``, ``actor.reset_flags |= 2`` (MAP_IN_COMBAT; 793 -> 795) and ``ppo.kickstart`` off. Never
overwrites; the source is not touched. No host change: the calc dist of the run is kept.

    python -B scripts/migrate_inputs2_config.py --config runs/X/config.json --output runs/X/config.inputs2.json

Then start the run once with ``--config <output> --resume --migrate-inputs2`` (later resumes without the flag: the
supervisor persists the config with ``migrate_inputs2 = false``), or resume a checkpoint migrated by
``scripts/migrate_inputs2.py`` with plain ``--resume``.
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from spire.inputs2_migration import check_source
from spire.schema import RESET_FLAGS_DEFAULT, ResetFlag
from spire.tokenize import TokenizerConfig
from spire.train import TrainConfig, build, check_forecast_flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), "output exists; never overwrites"
    text = args.config.read_text(encoding="utf-8")
    raw = tomllib.loads(text) if args.config.suffix == ".toml" else json.loads(text)
    tok = raw.setdefault("tokenizer", {})
    check_source(build(TokenizerConfig, tok))
    actor = raw.setdefault("actor", {})
    flags = int(actor.get("reset_flags", RESET_FLAGS_DEFAULT))
    assert not flags & int(ResetFlag.MAP_IN_COMBAT) and flags & int(ResetFlag.CALC), f"reset_flags {flags}"
    tok.update(map_tokens="no_combat", route_summary=True, second_boss_table=True)
    actor["reset_flags"] = flags | int(ResetFlag.MAP_IN_COMBAT)
    ppo = raw.setdefault("ppo", {})
    kick_before = ppo.pop("kickstart", None)
    for key in ("resume_from", "migrate_public_history", "migrate_seat_symmetry", "migrate_coop_features",
                "migrate_forecast", "migrate_calc", "migrate_inputs2", "reset_seat_symmetry_optimizer"):
        raw.pop(key, None)
    cfg = build(TrainConfig, raw)  # the result must be a valid config
    check_forecast_flags(cfg)
    assert not cfg.ppo.kickstart.on
    args.output.write_text(json.dumps(raw, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tokenizer": {k: tok[k] for k in (
        "map_tokens", "route_summary", "second_boss_table")}, "actor.reset_flags": actor["reset_flags"],
        "ppo.kickstart_before": kick_before, "ppo.kickstart": "off"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

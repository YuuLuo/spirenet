"""Run-config migration for the critic tower (notes/critic/contract.md 5): a copy of a run's config (the supervisor's
``config.json`` or a TOML) with ``model.critic`` and ``ppo.critic`` at their defaults and the host launch specs of the
critic dist (``--training-hidden``). Never overwrites; the source (config.pre-critic.json) is not touched. Every
other key (tokenizer incl. inputs v2a and rules mode, practice, kickstart) is kept.

    python -B scripts/migrate_critic_config.py --config runs/X/config.json \\
        --host-config DIST/linux_host_calc_hidden_v01131.toml \\
        [--practice-host-config DIST/linux_host_calc_search_hidden_v01131.toml] --output runs/X/config.critic.json

Then start the run once with ``--config <output> --resume --migrate-critic``.

RESET bit 10 is NOT written into ``actor.reset_flags``: ``schema.reset_flags_for`` refuses a base with it, and the
training actor alone adds it (spire.critic; build-v1.md 2.1).
"""
from __future__ import annotations

import argparse
import copy
import json
import tomllib
from dataclasses import asdict
from pathlib import Path

from spire.critic import CriticConfig, CriticPPOConfig, carries_hidden
from spire.host import real_host_spec
from spire.schema import RESET_FLAGS_DEFAULT
from spire.train import TrainConfig, build, check_forecast_flags
from spire.vocab import Vocab


def _same_dist_files(old: Path, new: Path) -> None:
    a, b = real_host_spec(old), real_host_spec(new)
    assert Vocab.load(a.vocab).hash == Vocab.load(b.vocab).hash, f"{new}: another vocabulary"
    assert a.catalog.read_bytes() == b.catalog.read_bytes(), f"{new}: another catalog"
    assert "--training-hidden" in b.command, f"{new}: the launch spec lacks --training-hidden"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--host-config", type=Path, required=True, help="the plain hidden launch spec of the critic dist")
    ap.add_argument("--practice-host-config", type=Path, help="the fork-capable hidden spec (runs with practice)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), "output exists; never overwrites"
    text = args.config.read_text(encoding="utf-8")
    raw = tomllib.loads(text) if args.config.suffix == ".toml" else json.loads(text)
    source = copy.deepcopy(raw)
    model = raw.setdefault("model", {})
    ppo = raw.setdefault("ppo", {})
    assert "critic" not in model and "critic" not in ppo, "already a critic config"
    assert not (model.get("override") or {}).get("critic"), "already a critic config"
    flags = int(raw.get("actor", {}).get("reset_flags", RESET_FLAGS_DEFAULT))
    assert not carries_hidden(flags), "actor.reset_flags must not carry bit 10 (1024)"
    hosts = raw.setdefault("hosts", {})
    _same_dist_files(Path(hosts["real_config"]), args.host_config)
    model["critic"] = asdict(CriticConfig())
    ppo["critic"] = json.loads(json.dumps(asdict(CriticPPOConfig())))
    hosts["real_config"] = str(args.host_config.resolve())
    pr = raw.get("practice", {})
    if pr.get("enabled", False):
        assert args.practice_host_config is not None, "a practice run needs --practice-host-config"
        _same_dist_files(Path(pr["host_config"]), args.practice_host_config)
        pr["host_config"] = str(args.practice_host_config.resolve())
    else:
        assert args.practice_host_config is None, "--practice-host-config without practice"
    for key in ("resume_from", "migrate_public_history", "migrate_seat_symmetry", "migrate_coop_features",
                "migrate_forecast", "migrate_calc", "migrate_inputs2", "migrate_rules", "rules_free",
                "reset_seat_symmetry_optimizer"):
        raw.pop(key, None)
    # the result must be a valid config. Until spire.learner.PPOConfig has the field ``critic`` (package P4) the
    # section is validated on its own and the rest without it.
    build(CriticConfig, model["critic"])
    build(CriticPPOConfig, ppo["critic"])
    rest = copy.deepcopy(raw)
    from spire.learner import PPOConfig
    import dataclasses
    if "critic" not in {f.name for f in dataclasses.fields(PPOConfig)}:
        del rest["ppo"]["critic"]
    cfg = build(TrainConfig, rest)
    check_forecast_flags(cfg)
    assert cfg.model.config().critic == CriticConfig()
    changed = sorted(k for k in set(source) | set(raw) if source.get(k) != raw.get(k))
    assert set(changed) <= {"model", "ppo", "hosts", "practice", "resume_from", "migrate_public_history",
                            "migrate_seat_symmetry", "migrate_coop_features", "migrate_forecast", "migrate_calc",
                            "migrate_inputs2", "migrate_rules", "rules_free", "reset_seat_symmetry_optimizer"}, changed
    args.output.write_text(json.dumps(raw, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "model.critic": model["critic"], "ppo.critic": ppo["critic"],
                      "hosts.real_config": hosts["real_config"],
                      "practice.host_config": pr.get("host_config") if pr.get("enabled", False) else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

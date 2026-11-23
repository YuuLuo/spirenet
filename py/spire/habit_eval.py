"""Greedy 4-seat panel evaluation with one habit forced on top of the policy (notes/habit; ``spire.habit``).

Writes into ``--out``: the evaluation result (``<panel>/<tag>.json`` + ``.episodes.jsonl``), ``games.jsonl`` (one row
per game: result, floor, bosses, death location, habit counters) and ``gamelog/`` (full decision records of the games
with seed id < ``--gamelog-games``)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from .actor import RULES_KEYS, TorchPolicy
from .evalpanel import evaluate, load_catalog, load_panel, load_policy_net
from .gamelog import GameLogger
from .habit import HabitOverride, HabitRecorder, write_rows
from .host import real_host_spec
from .proto import Frame
from .train import HostSection, make_hosts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host-config", required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--habit", required=True, help="baseline | combo | combo2 | smith:0.5 | no_elite:1-2 | ... (+)")
    ap.add_argument("--commit-select", choices=("on", "off"), required=True,
                    help="tokenizer.commit_select for this evaluation (CANCEL of upgrade / enchant prompts masked)")
    ap.add_argument("--removal-rule", required=True,
                    help="off | <removal-ratings.json>: tokenizer.removal_rule for this evaluation (spire.removal)")
    ap.add_argument("--rules-mode", choices=("free",), default=None,
                    help="free: play a rules_mode teach checkpoint with the removal prompt offered in full (spire.rules; "
                         "default: the checkpoint's mode)")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", choices=("ema", "online"), default="ema")
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--max-episode-decisions", type=int, default=20000)
    ap.add_argument("--round-budget-s", type=float, default=36000.0)
    ap.add_argument("--gamelog-games", type=int, default=5,
                    help="keep full decision records of seed ids below this (0: no game logger; spire.gamelog logs "
                         "multi-seat games only)")
    ap.add_argument("--no-graphs", action="store_true", help="eager inference (no CUDA graphs)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    assert 1 <= a.envs <= 16
    torch.set_num_threads(2)
    spec = real_host_spec(a.host_config)
    net, tok, version, extra = load_policy_net(a.checkpoint, device=a.device, weights=a.weights,
                                             vocab_path=spec.vocab, catalog_path=spec.catalog)
    trained_commit = tok.config.commit_select
    assert tok.config.cycle_ban, "commit_select uses the cycle-ban mask"
    trained_removal = tok.config.removal_rule
    rule = "" if a.removal_rule == "off" else a.removal_rule
    tok.config = dataclasses.replace(tok.config, commit_select=a.commit_select == "on", removal_rule=bool(rule),
                                     removal_ratings=rule)
    trained_rules = tok.config.rules_mode
    if a.rules_mode == "free":
        assert trained_rules in ("teach", "free"), f"--rules-mode free needs a teach checkpoint, not {trained_rules}"
        tok.config = dataclasses.replace(tok.config, rules_mode="free")
        net.rules_mode = "free"
    panel = load_panel(a.panel)
    catalog = load_catalog(str(spec.catalog), tok.vocab)
    assert catalog is not None
    a.out.mkdir(parents=True, exist_ok=True)
    override = HabitOverride(a.habit, tok.vocab, catalog)
    recorder = HabitRecorder(override)
    glog_dir = a.out / "gamelog"
    gamelog = GameLogger(tok.vocab, glog_dir, {"checkpoint": str(a.checkpoint), "version": version,
                                              "weights": a.weights, "mode": "greedy", "panel": panel.name,
                                              "habit": a.habit}) if a.gamelog_games > 0 else None

    def on_decision(env_id: int, f: Frame, action: int) -> None:
        recorder.observe(env_id, f, action)
        if gamelog is not None:
            gamelog.observe(env_id, f, action)

    policy = TorchPolicy(net, tok, mode="greedy", seed=a.seed, version=version,
                         use_graphs=a.device != "cpu" and not a.no_graphs)
    hosts = make_hosts(HostSection(real=True, real_config=a.host_config, transport="stdio", priority="below_normal",
                                   max_restarts=0), a.envs, seed=a.seed, log_dir=a.out / "host-logs", name="habit",
                       expected_hashes=(tok.vocab.hash, int(extra["catalog_hash"])))
    info: dict[str, Any] = {"checkpoint": str(a.checkpoint), "version": version, "mode": "greedy",
                            "weights": a.weights, "habit": a.habit, "commit_select": a.commit_select == "on",
                            "commit_select_trained": trained_commit, "removal_rule": rule or "off",
                            "removal_rule_trained": trained_removal, "rules_mode": tok.config.rules_mode,
                            "rules_mode_trained": trained_rules}
    try:
        result, path = evaluate(panel, policy, tok, hosts, tag=f"v{version}-{a.habit.replace(':', '')}",
                                policy_info=info, out_dir=a.out, hosts_info=a.host_config, max_batch=a.envs,
                                max_episode_decisions=a.max_episode_decisions, round_budget_s=a.round_budget_s,
                                on_decision=on_decision, override=override, on_job=override.on_job)
    finally:
        policy.close()
        for h in hosts:
            h.close()
    rows = recorder.rows(result.episodes, a.gamelog_games)
    write_rows(rows, a.out / "games.jsonl")
    index = gamelog.finish(result.episodes) if gamelog is not None else []
    kept = 0
    for r in index:
        if r["seed_id"] is not None and int(r["seed_id"]) < a.gamelog_games:
            kept += 1
        else:
            (glog_dir / r["path"]).unlink()
    summary = {"info": info, "panel": panel.name, "result": str(path), "summary": result.summary,
               "partial": result.partial, "games": len(rows), "gamelogs_kept": kept,
               "seconds": result.seconds, "decisions": result.decisions,
               "games_at_cap": sum(e["result"] == "decision_cap" for e in result.episodes)}
    if tok.config.rules_mode != "mask":  # spire.rules telemetry: sums over the games (per game: the episodes)
        summary["rules"] = {k: sum(e["rules"][k] for e in result.episodes) for k in RULES_KEYS}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    shutil.rmtree(a.out / "host-logs")  # kept only when the evaluation failed
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

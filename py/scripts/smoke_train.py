"""End-to-end smoke run against fake SpireHost processes: train -> checkpoint -> panel eval -> dashboard.

    python -B scripts/smoke_train.py [--minutes 4] [--actors 1] [--envs 16] [--name smoke]

* fake hosts are real subprocesses (``scripts/fake_spirehost.py``) speaking the binary protocol in
  ``learnable`` mode (losing HP for END_TURN while a card is playable), so pipes, process
  management and a policy-dependent outcome are all exercised;
* training uses the ``processes`` topology (actor process(es) + learner process + supervisor);
* afterwards the frozen panel ``eval/panels/smoke-fake.json`` (created on first use, before
  training, so its seeds are excluded from training) is played greedily with the checkpoint's EMA
  and online weights and with the uniform-random baseline, compared pairwise (McNemar), and the
  dashboard is written to ``runs/<name>/dashboard.html``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from spire.actor import TorchPolicy, UniformPolicy  # noqa: E402
from spire.dashboard import write_dashboard  # noqa: E402
from spire.evalpanel import (  # noqa: E402
    PANEL_DIR,
    evaluate,
    load_panel,
    load_policy_net,
    make_panel,
    paired_compare,
    save_panel,
)
from spire.metrics import read_jsonl  # noqa: E402
from spire.synth import FakeHostConfig  # noqa: E402
from spire.tokenize import Tokenizer  # noqa: E402
from spire.train import REPO, HostSection, Supervisor, TrainConfig, build, make_hosts, panel_seeds  # noqa: E402
from spire.vocab import Vocab, make_synthetic_vocab  # noqa: E402

PANEL = "smoke-fake"
FAKE = {"learnable": True, "mean_decisions": 150, "hashes": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--actors", type=int, default=1)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--name", default=f"smoke-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--panel-seeds", type=int, default=64)
    ap.add_argument("--transport", choices=("stdio", "pipe"), default="pipe", help="host frame transport")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not (PANEL_DIR / f"{PANEL}.json").exists():
        save_panel(make_panel(PANEL, a.panel_seeds, ["IRONCLAD", "SILENT"], [0], rng_seed=20261114,
                              exclude=panel_seeds()))
    panel = load_panel(PANEL)

    cfg: TrainConfig = build(TrainConfig, {
        "run_name": a.name, "topology": "processes", "duration_s": a.minutes * 60, "seed": 1,
        "metrics_interval_s": 10.0, "gpu_interval_s": 10.0,
        "model": {"override": {"d_model": 128, "n_layers": 2, "n_heads": 4, "d_ff": 256, "d_id": 32, "d_cond": 32,
                               "d_static": 64}},
        "ppo": {"minibatch": 2048, "micro_batch": 1024, "lr": 1e-3, "ent_coef": 0.003},
        "rollout": {"segment_len": 128},
        "episodes": {"lineups": [["IRONCLAD", 0.5], ["SILENT", 0.5]]},
        "actor": {"actors": a.actors, "device": device, "max_batch": 64, "report_interval": 5.0},
        "hosts": {"envs_per_actor": a.envs, "fake": FAKE, "transport": a.transport},
        "learner": {"device": device, "iteration_samples": 8192, "checkpoint_interval_s": 60.0},
        "shm_mb": 128,
    })
    t0 = time.time()
    code = Supervisor(cfg).run()
    print(f"training finished with code {code} after {time.time() - t0:.0f} s")
    run = cfg.run_dir
    ckpt = run / "ckpt" / "latest.pt"

    # ---- panel evaluation: EMA / online weights (greedy) and the uniform baseline, paired
    vocab = Vocab(make_synthetic_vocab())
    fake = build(FakeHostConfig, FAKE)
    results = {}
    for kind in ("ema", "online", "uniform"):
        hosts = make_hosts(HostSection(fake=fake, transport=a.transport), a.envs, seed=99, log_dir=run / "host-logs",
                           name=f"eval-{kind}")
        try:
            if kind == "uniform":
                policy: TorchPolicy | UniformPolicy = UniformPolicy(0)
                tok = Tokenizer(vocab)
                info: dict[str, object] = {"kind": "uniform"}
                tag = f"{a.name}-uniform"
            else:
                net, tok, version, extra = load_policy_net(ckpt, device=device, weights=kind)  # type: ignore[arg-type]
                policy = TorchPolicy(net, tok, mode="greedy", version=version)
                info = {"kind": "checkpoint", "checkpoint": str(ckpt), "version": version, "weights": kind,
                        "mode": "greedy", "run": a.name}
                tag = f"{a.name}-v{version}-{kind}-greedy"
            res, path = evaluate(panel, policy, tok, hosts, tag=tag, policy_info=info,
                                 hosts_info="fake_spirehost --learnable (subprocess)")
        finally:
            for h in hosts:
                h.close()
        results[kind] = res
        s = res.summary["all"]
        print(f"eval {kind:7s}: {s['wins']}/{s['n']} = {s['win_rate']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}] -> {path}")
    cmp = {k: paired_compare(results[k].episodes, results["uniform"].episodes)["all"] for k in ("ema", "online")}
    for k, c in cmp.items():
        print(f"paired {k} vs uniform: +{c['a_only']} / -{c['b_only']} of {c['pairs']}, McNemar p = {c['p_mcnemar']:.3g}")

    dash = write_dashboard(run)
    train = [r for r in read_jsonl(run / "metrics.jsonl") if r["kind"] == "train"]
    first, last = train[0], train[-1]
    summary = {
        "run": str(run), "code": code, "minutes": a.minutes, "actors": a.actors, "envs_per_actor": a.envs,
        "transport": a.transport,
        "decisions": last["decisions"], "games": last["games"], "updates": last["updates"],
        "decisions_per_s_last": last["decisions_per_s"],
        "decisions_per_s_mean": last["decisions"] / max(1.0, last["elapsed"]),
        "stage_us_last": last["stage_us"], "transfer_us_per_step_last": last.get("transfer_us_per_step"),
        "counts": last["counts"],
        "win_rate_first_window": {k: v["win_rate"] for k, v in first["outcomes"].items()},
        "win_rate_last_window": {k: v["win_rate"] for k, v in last["outcomes"].items()},
        "eval": {k: r.summary["all"] for k, r in results.items()}, "paired_vs_uniform": cmp,
        "dashboard": str(dash),
    }
    (run / "smoke_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"dashboard: {dash}")
    return code


if __name__ == "__main__":
    sys.exit(main())

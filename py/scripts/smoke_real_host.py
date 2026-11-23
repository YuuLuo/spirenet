"""End-to-end smoke test against the REAL SpireHost encoder/server (``SpireHost.Obs --serve``).

Plays N episodes per host with uniformly random legal actions through ``spire.host.HostProcess`` (stdio or named
pipe), checks sampled frames with ``spire.proto.frame_problems`` plus the header hashes against the exported
vocab.json / catalog.json, optionally tokenizes sampled frames with ``spire.tokenize.Tokenizer``, exercises the
ERROR / resend path and the "RESET while an episode is pending" (abandon) path, and prints a JSON summary with
decisions/s per host and every protocol disagreement it saw.

    python -B scripts/smoke_real_host.py --episodes 150 --hosts 2 [--transport stdio|pipe|mixed] [--hp 99999]
        [--check-every 1] [--tokenize] [--inject-errors] [--abandon-prob 0.01]

Owned by the obs team (notes/host-h1-obs.md). Read-only use of py/spire: nothing there is modified.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "py"))

from spire.host import ActionRejected, HostConfig, HostError, HostProcess  # noqa: E402
from spire.proto import ProtocolError, frame_problems  # noqa: E402
from spire.schema import Kind, Result, SchemaError  # noqa: E402
from spire.vocab import Catalog, Vocab  # noqa: E402

OBS_DLL = ROOT / "host" / "SpireHost.Obs" / "bin" / "Release" / "net9.0" / "SpireHost.Obs.dll"
DATA = ROOT / "host" / "SpireHost.Obs" / "data"
CHARACTERS = ("IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT", "FOOL")
G_ELITES_WON, G_BOSSES_WON, G_TERMINAL_REASON, G_POLICY_DECISIONS = 45, 46, 49, 51  # errata-v1 GLOBAL slots
SEED_SPACE = 34 ** 12  # wire seeds: base-34 numbers of the 12-character game seed (errata P-4)


def run_host(idx: int, transport: str, args: argparse.Namespace, vocab: Vocab, catalog: Catalog, out: dict) -> None:
    rng = random.Random(args.seed * 1000 + idx)
    cmd = ["dotnet", str(OBS_DLL), "--serve", "--max-decisions", str(args.max_decisions)]
    if args.hp:
        cmd += ["--hp", str(args.hp), "--turn-cap", str(args.turn_cap)]
    cfg = HostConfig(command=tuple(cmd), name=f"smoke{idx}", transport=transport, step_timeout=60.0,
                     reset_timeout=120.0, expected_hashes=(vocab.hash, catalog.hash), verify_hashes=False,
                     log_path=str(ROOT / "bench" / "h1-obs" / "logs" / f"smoke{idx}.stderr.log"))
    tok = None
    if args.tokenize:
        from spire.tokenize import Tokenizer
        tok = Tokenizer(vocab)
    stats: collections.Counter = collections.Counter()
    by_kind: collections.Counter = collections.Counter()
    terminal_reason: collections.Counter = collections.Counter()
    problems: collections.Counter = collections.Counter()
    disagreements: list[str] = []
    first_problem: list[str] = []
    decisions = 0
    checked = 0
    host = HostProcess(cfg)
    t_boot = time.perf_counter()
    t0 = None
    try:
        for ep in range(args.episodes):
            ch = CHARACTERS[(idx + ep) % len(CHARACTERS)]
            seed = None if ep % 3 == 0 else rng.randrange(SEED_SPACE)
            asc = rng.choice([0, 0, 0, 10])
            try:
                f = host.reset(seed, [vocab.id("character", ch)], ascension=asc)
                if t0 is None:
                    t0 = time.perf_counter()  # first frame: the engine has booted
                    stats["boot_s_x10"] = int((t0 - t_boot) * 10)
                steps = 0
                abandon_at = rng.randrange(5, 200) if rng.random() < args.abandon_prob else -1
                while not f.is_terminal:
                    by_kind[Kind(f.kind).name] += 1
                    if steps % args.check_every == 0:
                        checked += 1
                        for p in frame_problems(f):
                            problems[p] += 1
                            if len(first_problem) < 5:
                                first_problem.append(f"ep {ep} step {f.step_id}: {p}")
                        if tok is not None:
                            tok.encode(f)
                    if steps == abandon_at:
                        stats["abandoned_by_reset"] += 1
                        break  # the next reset() abandons this episode (the host sends no TERMINAL for it)
                    n = f.n_actions
                    if args.inject_errors and rng.random() < 0.002:
                        try:  # ERROR + resend: bad index, then stale step id
                            host.step(n + 3)
                        except ActionRejected as e:
                            stats["error_bad_index"] += int(e.code == 1)
                            assert e.frame is not None  # a rejected STEP: the host re-sent its OBS frame
                            f = e.frame
                        try:
                            host.step(0, step_id=f.step_id + 5)
                        except ActionRejected as e:
                            stats["error_stale_step"] += int(e.code == 2)
                            assert e.frame is not None
                            f = e.frame
                        continue
                    f = host.step(rng.randrange(n))
                    steps += 1
                    decisions += 1
                if not f.is_terminal:
                    continue
                res = Result(f.result).name
                stats[res] += 1
                reason = int(f.glob[G_TERMINAL_REASON])
                terminal_reason[reason] += 1
                if int(f.glob[G_POLICY_DECISIONS]) != f.step_id or f.step_id != steps:
                    disagreements.append(f"ep {ep}: TERMINAL step_id {f.step_id} / G[51] {int(f.glob[G_POLICY_DECISIONS])} / steps {steps}")
                stats["elites_won"] += int(f.glob[G_ELITES_WON])
                stats["bosses_won"] += int(f.glob[G_BOSSES_WON])
                for p in frame_problems(f):
                    problems["terminal: " + p] += 1
            except (HostError, ProtocolError, SchemaError) as e:  # a protocol disagreement: counted, the host restarts
                stats["host_errors"] += 1
                if len(disagreements) < 20:
                    disagreements.append(f"ep {ep}: {type(e).__name__}: {str(e)[:200]}")
        stats["episodes"] = args.episodes
    finally:
        host.close()
    wall = time.perf_counter() - (t0 or t_boot)
    out[idx] = {
        "transport": transport, "stats": dict(stats), "decisions": decisions, "frames_checked": checked,
        "wall_s": round(wall, 1), "decisions_per_s": round(decisions / max(wall, 1e-9), 1), "by_kind": dict(by_kind),
        "terminal_reason": {str(k): v for k, v in terminal_reason.items()}, "problems": dict(problems),
        "first_problems": first_problem, "disagreements": disagreements, "host_stats": host.stats.as_dict(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50, help="episodes per host")
    ap.add_argument("--hosts", type=int, default=1)
    ap.add_argument("--transport", default="stdio", choices=("stdio", "pipe", "mixed"))
    ap.add_argument("--hp", type=int, default=0)
    ap.add_argument("--turn-cap", type=int, default=100)
    ap.add_argument("--max-decisions", type=int, default=20000)
    ap.add_argument("--check-every", type=int, default=1)
    ap.add_argument("--tokenize", action="store_true")
    ap.add_argument("--inject-errors", action="store_true")
    ap.add_argument("--abandon-prob", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.hosts > 4:
        raise SystemExit("at most 4 host processes")
    vocab = Vocab.load(DATA / "vocab.json")
    catalog = Catalog.load(DATA / "catalog.json", vocab)
    (ROOT / "bench" / "h1-obs" / "logs").mkdir(parents=True, exist_ok=True)
    out: dict = {}
    transports = [("stdio", "pipe")[i % 2] if a.transport == "mixed" else a.transport for i in range(a.hosts)]
    threads = [threading.Thread(target=run_host, args=(i, transports[i], a, vocab, catalog, out)) for i in range(a.hosts)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total: collections.Counter = collections.Counter()
    for r in out.values():
        total.update(r["stats"])
    decisions = sum(r["decisions"] for r in out.values())
    summary = {"transport": a.transport, "hp": a.hp, "hosts": a.hosts, "episodes_per_host": a.episodes,
               "check_every": a.check_every, "tokenize": a.tokenize, "totals": dict(total), "decisions": decisions,
               "wall_s": round(time.perf_counter() - t0, 1),
               "decisions_per_s_sum_of_hosts": round(sum(r["decisions_per_s"] for r in out.values()), 1),
               "problems": sum(sum(r["problems"].values()) for r in out.values()),
               "disagreements": sum(len(r["disagreements"]) for r in out.values()), "per_host": out}
    text = json.dumps(summary, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    ok = (len(out) == a.hosts and summary["problems"] == 0 and summary["disagreements"] == 0
          and all(r["host_stats"]["crashes"] == 0 and r["host_stats"]["timeouts"] == 0 for r in out.values()))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

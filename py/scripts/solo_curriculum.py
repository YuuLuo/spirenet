"""Solo ascension curriculum of one run: every ``interval`` seconds read the newest train row of
runs/<run>/metrics.jsonl and, per character of the run's single-seat lineups, rewrite that lineup's ascension mix in
runs/<run>/sampling.json (the run polls it every 30 s).

    python scripts/solo_curriculum.py <run> [interval_s=1200] [--taste]      (from py/; runs in $SPIRE_RUN_ROOT, default ../runs)

--taste (runs with ppo.taste, spire.taste / notes/solo/taste-spec.md v1.1 F): read the outcome window of the "none"
seat-episodes only (train row taste.none_outcomes, same "<lineup>|A<asc>" keys and n / win_rate fields): half of the
games of a taste run are played by the taste behaviour. Without the flag the behaviour is unchanged.

a* = the highest ascension with at least 200 finished games in the outcome window and win rate >= 0.15 (0 if none);
weights a*+1 30%, a*+2 20%, a* 20%, a*-1 10%, A10 10%, 10% spread evenly over the ascensions below a*-1 (none: on
A0); clipped to 0..10, duplicates merged, renormalised. The file is replaced atomically; one line per character goes
to runs/<run>/curriculum.log (UTC time, character, a*, the window's n / win rate per ascension).
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

RUNS = Path(os.environ.get("SPIRE_RUN_ROOT", "../runs"))
MIN_GAMES = 200
MIN_WIN = 0.15


def newest_train_row(path: Path) -> dict | None:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 8_000_000))
        lines = f.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        if '"kind": "train"' in line:
            return json.loads(line)
    return None


def mix(a_star: int) -> list[list[float]]:
    w: dict[int, float] = defaultdict(float)
    for a, p in ((a_star + 1, 0.3), (a_star + 2, 0.2), (a_star, 0.2), (a_star - 1, 0.1), (10, 0.1)):
        w[min(10, max(0, a))] += p
    lower = list(range(0, a_star - 1)) or [0]
    for a in lower:
        w[a] += 0.1 / len(lower)
    total = sum(w.values())
    return [[a, round(w[a] / total, 6)] for a in sorted(w)]


def step(run: str, taste: bool = False) -> None:
    rd = RUNS / run
    row = newest_train_row(rd / "metrics.jsonl")
    if row is None:
        return
    sampling = json.loads((rd / "sampling.json").read_text(encoding="utf-8"))
    outcomes = row["taste"]["none_outcomes"] if taste else row["outcomes"]
    stamp = time.strftime("%F %T", time.gmtime())
    new_la = []
    log = []
    for lineup, _ in sampling["lineups"]:
        assert "+" not in lineup, lineup
        stats = {int(k.split("|A")[1]): v for k, v in outcomes.items() if k.split("|A")[0] == lineup}
        ok = [a for a, v in stats.items() if v["n"] >= MIN_GAMES and v["win_rate"] >= MIN_WIN]
        a_star = max(ok, default=0)
        new_la.append([lineup, mix(a_star)])
        rates = " ".join(f"A{a}:{stats[a]['n']}/{stats[a]['win_rate']:.3f}" for a in sorted(stats))
        log.append(f"{stamp} v{row['version']} {lineup} a*={a_star} window {rates}\n")
    sampling["lineup_ascensions"] = new_la
    sampling["comment"] = f"solo_curriculum.py {stamp} (train row version {row['version']})"
    tmp = rd / "sampling.json.tmp"
    tmp.write_text(json.dumps(sampling, indent=1), encoding="utf-8")
    os.replace(tmp, rd / "sampling.json")
    with open(rd / "curriculum.log", "a", encoding="utf-8") as f:
        f.writelines(log)


def main() -> None:
    run = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1200.0
    assert len(sys.argv) <= 3 or sys.argv[3] == "--taste", sys.argv
    taste = len(sys.argv) > 3
    while True:
        time.sleep(interval)
        step(run, taste)


if __name__ == "__main__":
    main()

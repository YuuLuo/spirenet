"""Fixed seed panels, checkpoint evaluation and the statistics for claims (DESIGN 3.3, design-review 4).

* Panel = a frozen list of 12-character game seeds (alphabet of ``SeedHelper``) x a set of
  configurations (lineup x ascension); every seed is played once per configuration. Panels live
  in ``eval/panels/<name>.json`` and are never overwritten; their seeds are excluded from training
  (``spire.train.panel_seeds``).
* :func:`evaluate` plays a panel with a policy (EMA weights of a checkpoint by default, greedy or
  with a temperature) on real or fake hosts through :class:`spire.actor.Actor` in eval mode.
  Every episode is appended to ``<result>.episodes.jsonl`` as it ends; the summary goes to
  ``eval/results/<panel>/<tag>.json``. Failures (aborted / lost / ERROR / overflow) stay in the
  denominator: the reported win rate is wins / all episodes.
* Statistics: Wilson intervals (two-sided, and the one-sided lower bound for "rate > target"
  claims), McNemar's exact test for two checkpoints on the same panel (pairs by seed and
  configuration), Wald's SPRT for sequential claims.

CLI::

    python -B -m spire.evalpanel make --name dev-a0 --seeds 512 --lineups IRONCLAD SILENT --asc 0
    python -B -m spire.evalpanel eval --panel dev-a0 --checkpoint runs/x/ckpt/latest.pt [--fake]
    python -B -m spire.evalpanel compare A.json B.json
    python -B -m spire.evalpanel sprt result.json --p0 0.9 --p1 0.95
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch

from .actor import (
    Actor,
    ActorConfig,
    EpisodeJob,
    JobList,
    Policy,
    TorchPolicy,
    UniformPolicy,
    canonical_seed,
    parse_lineup,
    random_seed,
)
from .host import Host, real_host_spec
from .metrics import EpisodeRecord, JsonlWriter, canonical_lineup, lineup_key, read_jsonl, wilson_interval, wilson_lower
from .model import ModelConfig, SpireNet
from .proto import Frame
from .schema import reset_flags_for
from .synth import FakeHost, FakeHostConfig
from .tokenize import Tokenizer, TokenizerConfig
from .train import REPO, HostSection, TrainConfig, build, load_vocab, make_hosts, panel_seeds, select_device
from .vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab

PANEL_DIR = REPO / "eval" / "panels"
RESULT_DIR = REPO / "eval" / "results"
__all__ = ["Panel", "make_panel", "save_panel", "load_panel", "evaluate", "summarize", "paired_compare",
           "mcnemar_exact", "SPRT", "wilson_interval", "wilson_lower", "load_policy_net", "load_catalog"]


# =========================================================================== panels


@dataclass(frozen=True)
class Panel:
    name: str
    seeds: tuple[str, ...]
    lineups: tuple[str, ...]  # "IRONCLAD" / "IRONCLAD+SILENT+..."
    ascensions: tuple[int, ...]
    created: str = ""
    generator: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lineups = tuple(canonical_lineup(lineup) for lineup in self.lineups)
        assert len(set(lineups)) == len(lineups), "duplicate canonical panel lineups"
        object.__setattr__(self, "lineups", lineups)

    @property
    def configs(self) -> list[tuple[str, int]]:
        return [(lu, a) for lu in self.lineups for a in self.ascensions]

    def jobs(self, vocab: Vocab, configs: Sequence[tuple[str, int]] | None = None) -> list[EpisodeJob]:
        """Every seed once per configuration, seed-major (configurations interleave over time)."""
        cfgs = [(canonical_lineup(lu), a) for lu, a in configs] if configs is not None else self.configs
        assert len(set(cfgs)) == len(cfgs), "duplicate canonical panel configurations"
        ids = {lu: parse_lineup(lu, vocab) for lu, _ in cfgs}
        return [EpisodeJob(ids[lu], lu, a, s, f"{i}") for i, s in enumerate(self.seeds) for lu, a in cfgs]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seeds": list(self.seeds), "lineups": list(self.lineups),
                "ascensions": list(self.ascensions), "created": self.created, "generator": dict(self.generator)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Panel":
        return cls(d["name"], tuple(d["seeds"]), tuple(d["lineups"]), tuple(int(a) for a in d["ascensions"]),
                   d.get("created", ""), d.get("generator", {}))


def make_panel(name: str, n_seeds: int, lineups: Sequence[str], ascensions: Sequence[int], *,
               rng_seed: int | None = None, exclude: Iterable[str] = ()) -> Panel:
    """``n_seeds`` distinct seeds not in ``exclude``; ``rng_seed=None`` draws from the OS CSPRNG."""
    if n_seeds < 1 or not lineups or not ascensions:
        raise ValueError("a panel needs seeds, lineups and ascensions")
    taken = {canonical_seed(s) for s in exclude}
    rng = None if rng_seed is None else np.random.default_rng(rng_seed)
    seeds: list[str] = []
    while len(seeds) < n_seeds:
        s = random_seed(rng)
        if s not in taken:
            taken.add(s)
            seeds.append(s)
    return Panel(name, tuple(seeds), tuple(lineups), tuple(int(a) for a in ascensions),
                 dt.datetime.now().isoformat(timespec="seconds"),
                 {"source": "csprng" if rng_seed is None else "pcg64", "rng_seed": rng_seed})


def save_panel(panel: Panel, directory: Path = PANEL_DIR) -> Path:
    """Write ``<directory>/<name>.json``; an existing panel of that name is never replaced."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{panel.name}.json"
    if path.exists():
        if load_panel(path).seeds != panel.seeds:
            raise FileExistsError(f"panel {panel.name!r} exists with different seeds: panels are frozen")
        return path
    path.write_text(json.dumps(panel.to_dict(), indent=1), encoding="utf-8")
    return path


def load_panel(name_or_path: str | Path, directory: Path = PANEL_DIR) -> Panel:
    p = Path(name_or_path)
    if not p.suffix:
        p = directory / f"{name_or_path}.json"
    return Panel.from_dict(json.loads(p.read_text(encoding="utf-8")))


# =========================================================================== statistics


def binom_cdf(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2), exact (integer arithmetic)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) for i in range(k + 1)) / 2**n


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for ``b`` / ``c`` discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(b, c), n))


@dataclass
class SPRT:
    """Wald's sequential probability ratio test of H0: p = p0 against H1: p = p1 (p1 > p0).

    Feed outcomes with :meth:`update`; ``decision`` becomes ``"H1"`` (accept p1: the claim
    holds), ``"H0"``, or stays ``"continue"``. Error rates ``alpha`` (false H1) and ``beta``.
    """

    p0: float
    p1: float
    alpha: float = 0.05
    beta: float = 0.05
    n: int = 0
    k: int = 0
    llr: float = 0.0
    decision: Literal["continue", "H0", "H1"] = "continue"

    def __post_init__(self) -> None:
        if not 0 < self.p0 < self.p1 < 1:
            raise ValueError("need 0 < p0 < p1 < 1")
        self.upper = math.log((1 - self.beta) / self.alpha)
        self.lower = math.log(self.beta / (1 - self.alpha))
        self._w = math.log(self.p1 / self.p0)
        self._l = math.log((1 - self.p1) / (1 - self.p0))

    def update(self, win: bool) -> str:
        if self.decision != "continue":
            return self.decision
        self.n += 1
        self.k += int(win)
        self.llr += self._w if win else self._l
        if self.llr >= self.upper:
            self.decision = "H1"
        elif self.llr <= self.lower:
            self.decision = "H0"
        return self.decision

    def run(self, outcomes: Iterable[bool]) -> str:
        for w in outcomes:
            if self.update(w) != "continue":
                break
        return self.decision

    def as_dict(self) -> dict[str, Any]:
        return {"p0": self.p0, "p1": self.p1, "alpha": self.alpha, "beta": self.beta, "n": self.n, "k": self.k,
                "llr": self.llr, "upper": self.upper, "lower": self.lower, "decision": self.decision}


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per configuration (``lineup|A<asc>``) and pooled (``"all"``): wins / all episodes with
    Wilson 95% interval and one-sided 95% lower bound; failures counted and kept as non-wins."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        groups.setdefault(lineup_key(r["lineup"], r["ascension"]), []).append(r)
    groups["all"] = list(records)
    out: dict[str, dict[str, Any]] = {}
    for key, rs in groups.items():
        n = len(rs)
        wins = sum(r["result"] == "win" for r in rs)
        lo, hi = wilson_interval(wins, n)
        fails: dict[str, int] = {}
        for r in rs:
            if r["result"] not in ("win", "loss"):
                fails[r["result"]] = fails.get(r["result"], 0) + 1
        out[key] = {"n": n, "wins": wins, "win_rate": wins / n if n else None, "lo": lo, "hi": hi,
                    "lower_one_sided": wilson_lower(wins, n), "failures": fails,
                    "mean_floor": float(np.mean([r["floor"] for r in rs])) if n else None}
    return out


def summarize_strata(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """:func:`summarize` per ``lineup|A<asc>|<act1>`` (the Act 1 variant the episode got, from the frame's act
    id; records without it under ``?``) plus, per configuration, the act-1 stratified win rate
    ``sum_v share_v * rate_v`` with the panel's own shares (``stratified``) - comparable between checkpoints even
    when failures make the realised act-1 mix differ."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        groups.setdefault(f"{lineup_key(r['lineup'], r['ascension'])}|{r.get('act1') or '?'}", []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for key, rs in sorted(groups.items()):
        out[key] = {k: v for k, v in summarize(rs).items() if k == "all"}["all"]
    by_cfg: dict[str, list[str]] = {}
    for key in out:
        by_cfg.setdefault(key.rsplit("|", 1)[0], []).append(key)
    for cfg_key, keys in by_cfg.items():
        n = sum(out[k]["n"] for k in keys)
        if n:
            out[cfg_key + "|stratified"] = {"n": n, "win_rate": sum(out[k]["n"] / n * (out[k]["win_rate"] or 0.0)
                                                                    for k in keys),
                                            "act1_share": {k.rsplit("|", 1)[1]: out[k]["n"] / n for k in keys}}
    return out


def _pair_key(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return r["seed"], canonical_lineup(r["lineup"]), int(r["ascension"])


def paired_compare(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """McNemar on episodes matched by (seed, lineup, ascension); per configuration and pooled.
    ``a_only`` = pairs A won and B did not, ``b_only`` the reverse; ``diff`` = A rate - B rate."""
    ia = {_pair_key(r): r["result"] == "win" for r in a}
    ib = {_pair_key(r): r["result"] == "win" for r in b}
    common = sorted(set(ia) & set(ib))
    groups: dict[str, list[tuple[str, str, int]]] = {"all": common}
    for k in common:
        groups.setdefault(lineup_key(k[1], k[2]), []).append(k)
    out: dict[str, Any] = {"unmatched_a": len(ia) - len(common), "unmatched_b": len(ib) - len(common)}
    for key, ks in groups.items():
        a_only = sum(ia[k] and not ib[k] for k in ks)
        b_only = sum(ib[k] and not ia[k] for k in ks)
        n = len(ks)
        out[key] = {"pairs": n, "both": sum(ia[k] and ib[k] for k in ks), "neither": sum(not ia[k] and not ib[k]
                                                                                        for k in ks),
                    "a_only": a_only, "b_only": b_only, "a_rate": sum(ia[k] for k in ks) / n if n else None,
                    "b_rate": sum(ib[k] for k in ks) / n if n else None,
                    "diff": (a_only - b_only) / n if n else None, "p_mcnemar": mcnemar_exact(a_only, b_only)}
    return out


# =========================================================================== evaluation


def load_catalog(src: str, vocab: Vocab) -> Catalog | None:
    """A checkpoint's catalog source (``extra["catalog"]``): ``"synthetic"``, ``"none"`` or a catalog.json path."""
    if src == "synthetic":
        return Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
    return None if src == "none" else Catalog.load(src, vocab)


def load_policy_net(checkpoint: str | Path, *, device: str = "cuda", weights: Literal["ema", "online"] = "ema",
                    vocab_path: str | Path | None = None, catalog_path: str | Path | None = None
                    ) -> tuple[SpireNet, Tokenizer, int, dict[str, Any]]:
    """Rebuild the network of a training checkpoint (``Learner.save`` with the train ``extra``)."""
    ck = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    extra = ck.get("extra")
    if not extra:
        raise ValueError(f"{checkpoint}: no model description ('extra'); not a spire.train checkpoint")
    vocab_src = str(vocab_path) if vocab_path is not None else extra["vocab"]
    catalog_src = str(catalog_path) if catalog_path is not None else extra["catalog"]
    if "host" in (vocab_src, catalog_src):  # checkpoints written before the paths were resolved at save time
        spec = real_host_spec()
        vocab_src = str(spec.vocab) if vocab_src == "host" else vocab_src
        catalog_src = str(spec.catalog) if catalog_src == "host" else catalog_src
    vocab = Vocab(make_synthetic_vocab()) if vocab_src == "synthetic" else Vocab.load(vocab_src)
    if vocab.hash != extra["vocab_hash"]:
        raise ValueError("vocab differs from the one the checkpoint was trained with")
    catalog = load_catalog(catalog_src, vocab)
    if catalog_path is not None:
        assert (0 if catalog is None else catalog.hash) == extra["catalog_hash"], \
            "catalog override differs from the checkpoint catalog"
    tok = Tokenizer(vocab, build(TokenizerConfig, extra["tokenizer"]))
    net = SpireNet(tok.layout, build(ModelConfig, extra["model_config"]), catalog)
    net.load_state_dict(ck["ema" if weights == "ema" else "model"])
    return net.to(torch.device(device)).eval(), tok, int(ck["counters"]["version"]), extra


@dataclass
class EvalResult:
    panel: str
    tag: str
    policy: dict[str, Any]
    episodes: list[dict[str, Any]]
    summary: dict[str, dict[str, Any]]
    started: str
    seconds: float
    hosts: str
    expected_episodes: int
    completed_episodes: int
    partial: bool
    stop_reason: str
    strata: dict[str, dict[str, Any]] = field(default_factory=dict)  # summarize_strata (act-1 variant)
    decisions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _NoticeJobs(JobList):
    """:class:`JobList` that tells ``on_job(env_id, job)`` which job an env starts (before its RESET is sent)."""

    def __init__(self, jobs: Sequence[EpisodeJob], on_job: Callable[[int, EpisodeJob | None], None]) -> None:
        super().__init__(jobs)
        self.on_job = on_job

    def next_job(self, env_id: int) -> EpisodeJob | None:
        job = super().next_job(env_id)
        self.on_job(env_id, job)
        return job


def evaluate(panel: Panel, policy: Policy, tokenizer: Tokenizer, hosts: Sequence[Host], *, tag: str,
             policy_info: Mapping[str, Any], configs: Sequence[tuple[str, int]] | None = None,
             out_dir: Path = RESULT_DIR, hosts_info: str = "", max_batch: int = 128,
             max_episode_decisions: int = 0, round_budget_s: float = 1200.0,
             stop: Callable[[], bool] | None = None,
             on_decision: Callable[[int, Frame, int], None] | None = None,
             override: Any = None,
             on_job: Callable[[int, EpisodeJob | None], None] | None = None) -> tuple[EvalResult, Path]:
    """Play the panel up to ``round_budget_s`` or ``stop()`` (polled by the actor loop); ``on_decision`` is
    :class:`~spire.actor.Actor`'s per-decision callback (env id, frame, chosen action row); ``override`` its
    evaluation-only action override (:data:`spire.actor.Override`); ``on_job(env_id, job)`` is told which job an env
    starts (before its RESET; None = the env goes idle).

    Summaries contain completed episodes only and explicitly mark partial rounds. Interrupted episodes remain
    in the JSONL log as ``shutdown``; all decisions, including unfinished games, count toward evaluation cost.
    The caller closes the hosts after the round, including any requests still in flight at interruption.
    """
    assert max_episode_decisions >= 0
    assert round_budget_s > 0
    out = out_dir / panel.name
    out.mkdir(parents=True, exist_ok=True)
    ep_log = JsonlWriter(out / f"{tag}.episodes.jsonl")
    episodes: list[dict[str, Any]] = []

    def on_episode(rec: EpisodeRecord) -> None:
        d = rec.as_dict()
        if rec.result != "shutdown":
            episodes.append(d)
        ep_log.write(d)

    jobs = panel.jobs(tokenizer.vocab, configs)
    started = dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.monotonic()
    stop_reason = "complete"

    def round_stopped() -> bool:
        nonlocal stop_reason
        if stop is not None and stop():
            stop_reason = "shutdown"
            return True
        if time.monotonic() - t0 >= round_budget_s:
            stop_reason = "round_budget"
            return True
        return False

    actor = Actor(hosts, policy, tokenizer, JobList(jobs) if on_job is None else _NoticeJobs(jobs, on_job),
                  ActorConfig(mode="eval", max_batch=max_batch, max_episode_decisions=max_episode_decisions,
                              reset_flags=reset_flags_for(tokenizer.config.forecast, calc=tokenizer.config.calc,
                                                          map_in_combat=tokenizer.config.route_summary)),
                  on_episode=on_episode, on_decision=on_decision, override=override)
    try:
        counters = actor.run(round_stopped)
    finally:
        ep_log.close()
    partial = len(episodes) < len(jobs)
    assert len(episodes) <= len(jobs)
    assert not partial or stop_reason != "complete", f"evaluation incomplete: {len(episodes)} of {len(jobs)}"
    if not partial:
        stop_reason = "complete"
    res = EvalResult(panel.name, tag, dict(policy_info), episodes, summarize(episodes), started,
                     time.monotonic() - t0, hosts_info, len(jobs), len(episodes), partial, stop_reason,
                     summarize_strata(episodes), counters.decisions)
    path = out / f"{tag}.json"
    path.write_text(json.dumps(res.to_dict(), indent=1), encoding="utf-8")
    return res, path


# =========================================================================== periodic evaluation (training)


def ensure_eval_panel(cfg: TrainConfig) -> Panel:
    """The panel of ``cfg.eval``; created (CSPRNG seeds, excluding every existing panel's) if it does not exist
    yet - before training starts, so its seeds are excluded from training like every other panel's."""
    ev = cfg.eval
    if ev.panel.endswith(".json"):  # an explicit path (tests, ad-hoc panels outside eval/panels)
        path = Path(ev.panel)
        name = path.stem
    else:
        name = ev.panel or f"{cfg.run_name}-eval"
        path = PANEL_DIR / f"{name}.json"
    if path.exists():
        return load_panel(path)
    lineups = list(ev.lineups) or [lu for lu, _ in cfg.episodes.lineups]
    ascs = list(ev.ascensions) or [int(a) for a, _ in cfg.episodes.ascensions]
    panel = make_panel(name, ev.panel_seeds, lineups, ascs, exclude=panel_seeds())
    save_panel(panel, path.parent)
    return panel


def eval_main(cfg_d: dict[str, Any], msg_q: Any, stop: Any, train_decisions: Any) -> None:
    """Evaluator process of a training run (``cfg.eval``): plays the panel with the latest checkpoint's weights
    (EMA, greedy by default) on its own ``eval.envs`` hosts, never feeding training. A round starts when (a)
    ``first_after_s`` passed, (b) a checkpoint newer than the last evaluated one exists, (c) ``interval_s``
    passed since the previous round and (d) the evaluator's decisions so far are at most ``max_fraction`` of the
    training decisions. Each result goes to ``eval/results/<panel>/<run>-v<version>-<weights>-<mode>.json`` and,
    summarised, to the supervisor (``eval`` records in metrics.jsonl)."""
    from .sysutil import apply_placement
    cfg = build(TrainConfig, cfg_d)
    ev = cfg.eval
    torch.set_num_threads(1)
    select_device(ev.device)
    apply_placement(ev.priority, "")
    panel = ensure_eval_panel(cfg)
    ckpt = cfg.run_dir / "ckpt" / "latest.pt"
    vocab, catalog = load_vocab(cfg)
    t0 = time.monotonic()
    t_last = -1e18
    last_version = -1
    eval_decisions = 0
    mode: Literal["sample", "greedy"] = "greedy" if ev.temperature <= 0 else "sample"
    while not stop.is_set():
        if stop.wait(2.0):
            break
        now = time.monotonic()
        if now - t0 < ev.first_after_s or now - t_last < ev.interval_s or not ckpt.exists():
            continue
        if eval_decisions > ev.max_fraction * max(1, int(train_decisions.value)):
            continue
        net, tok, version, extra = load_policy_net(ckpt, device=ev.device, weights=ev.weights)
        if version == last_version:
            continue
        policy = TorchPolicy(net, tok, mode=mode, temperature=ev.temperature if ev.temperature > 0 else 1.0,
                             version=version, use_graphs=ev.device != "cpu")
        hosts = make_hosts(cfg.hosts, ev.envs, seed=(cfg.seed * 100 + 99) * 100 + cfg.run_start,
                           log_dir=cfg.run_dir / "host-logs", name="eval",
                           expected_hashes=(vocab.hash, catalog.hash if catalog is not None else 0)
                           if cfg.hosts.real else None)
        tag = f"{cfg.run_name}-v{version}-{ev.weights}-{mode}"
        t_round = time.monotonic()
        try:
            res, path = evaluate(panel, policy, tok, hosts, tag=tag, stop=stop.is_set,
                                 max_episode_decisions=ev.max_episode_decisions,
                                 round_budget_s=ev.round_budget_s,
                                 out_dir=Path(ev.results_dir) if ev.results_dir else RESULT_DIR,
                                 policy_info={"kind": "checkpoint", "checkpoint": str(ckpt), "version": version,
                                              "weights": ev.weights, "mode": mode, "temperature": ev.temperature,
                                              "run": cfg.run_name, "periodic": True},
                                 hosts_info=f"{ev.envs} evaluator hosts of the training run")
        finally:
            policy.close()
            for h in hosts:
                h.close()
        t_last = time.monotonic()
        last_version = version
        eval_decisions += res.decisions
        msg_q.put(("eval", 0, {"panel": panel.name, "version": version, "weights": ev.weights, "mode": mode,
                               "seconds": round(t_last - t_round, 1), "decisions": res.decisions,
                               "episodes": len(res.episodes), "summary": res.summary, "strata": res.strata,
                               "completed_episodes": res.completed_episodes,
                               "expected_episodes": res.expected_episodes,
                               "partial": res.partial, "stop_reason": res.stop_reason,
                               "path": str(path), "eval_decisions_total": eval_decisions,
                               "train_decisions": int(train_decisions.value)}))


def load_result(path: str | Path) -> dict[str, Any]:
    d: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return d


def load_episodes(path: str | Path) -> list[dict[str, Any]]:
    """Episodes of a result (summary JSON or its ``.episodes.jsonl``)."""
    p = Path(path)
    return list(read_jsonl(p)) if p.suffix == ".jsonl" else load_result(p)["episodes"]


# =========================================================================== CLI


def _fake_hosts(n: int, cfg: FakeHostConfig, seed: int) -> list[Host]:
    return [FakeHost(seed=seed + i, config=cfg) for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="seed panels and checkpoint evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="create a frozen seed panel")
    mk.add_argument("--name", required=True)
    mk.add_argument("--seeds", type=int, required=True)
    mk.add_argument("--lineups", nargs="+", required=True)
    mk.add_argument("--asc", type=int, nargs="+", default=[0])
    mk.add_argument("--rng-seed", type=int, default=None, help="reproducible panel (default: OS CSPRNG)")
    ev = sub.add_parser("eval", help="evaluate a checkpoint (or the uniform policy) on a panel")
    ev.add_argument("--panel", required=True)
    ev.add_argument("--checkpoint", default=None)
    ev.add_argument("--uniform", action="store_true", help="uniformly random legal actions (baseline)")
    ev.add_argument("--weights", choices=("ema", "online"), default="ema")
    ev.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ev.add_argument("--envs", type=int, default=16)
    ev.add_argument("--host-cmd", nargs="+", default=None, help="SpireHost command line")
    ev.add_argument("--fake-proc", action="store_true", help="fake hosts as subprocesses (default: in-process)")
    ev.add_argument("--fake-learnable", action="store_true")
    ev.add_argument("--fake-mean-decisions", type=float, default=300.0)
    ev.add_argument("--tag", default=None)
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--max-episode-decisions", type=int, default=0, help="RESET decision cap (0 = host default)")
    ev.add_argument("--round-budget-s", type=float, default=1200.0, help="wall-clock limit; report a partial round")
    cp = sub.add_parser("compare", help="paired McNemar comparison of two results on one panel")
    cp.add_argument("a")
    cp.add_argument("b")
    sp = sub.add_parser("sprt", help="sequential test of a result's episodes (in play order)")
    sp.add_argument("result")
    sp.add_argument("--p0", type=float, required=True)
    sp.add_argument("--p1", type=float, required=True)
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--beta", type=float, default=0.05)
    sp.add_argument("--config", default=None, help="only episodes of this lineup|A<asc>")
    a = ap.parse_args(argv)

    if a.cmd == "make":
        panel = make_panel(a.name, a.seeds, a.lineups, a.asc, rng_seed=a.rng_seed, exclude=panel_seeds())
        print(save_panel(panel))
        return 0
    if a.cmd == "compare":
        print(json.dumps(paired_compare(load_episodes(a.a), load_episodes(a.b)), indent=1))
        return 0
    if a.cmd == "sprt":
        eps = load_episodes(a.result)
        if a.config:
            eps = [e for e in eps if lineup_key(e["lineup"], e["ascension"]) == a.config]
        t = SPRT(a.p0, a.p1, a.alpha, a.beta)
        t.run(e["result"] == "win" for e in eps)
        print(json.dumps(t.as_dict(), indent=1))
        return 0

    panel = load_panel(a.panel)
    if a.uniform:
        vocab = Vocab(make_synthetic_vocab())
        tok = Tokenizer(vocab)
        policy: Policy = UniformPolicy(0)
        info: dict[str, Any] = {"kind": "uniform"}
        tag = a.tag or "uniform"
    else:
        if a.checkpoint is None:
            ap.error("--checkpoint or --uniform is required")
        net, tok, version, extra = load_policy_net(a.checkpoint, device=a.device, weights=a.weights)
        mode: Literal["sample", "greedy"] = "greedy" if a.temperature <= 0 else "sample"
        policy = TorchPolicy(net, tok, mode=mode, temperature=a.temperature if a.temperature > 0 else 1.0,
                             version=version)
        info = {"kind": "checkpoint", "checkpoint": str(a.checkpoint), "version": version, "weights": a.weights,
                "mode": mode, "temperature": a.temperature, "run": extra.get("run")}
        tag = a.tag or f"{extra.get('run')}-v{version}-{a.weights}-{mode}"
    fake = FakeHostConfig(learnable=a.fake_learnable, mean_decisions=a.fake_mean_decisions, hashes=False)
    if a.host_cmd:
        hosts = make_hosts(HostSection(command=tuple(a.host_cmd)), a.envs, seed=0,
                           log_dir=REPO / "runs" / "host-logs" / "eval", name="eval")
        hinfo = " ".join(a.host_cmd)
    elif a.fake_proc:
        hosts = make_hosts(HostSection(fake=fake), a.envs, seed=0, log_dir=REPO / "runs" / "host-logs" / "eval",
                           name="eval")
        hinfo = "fake_spirehost (subprocess)"
    else:
        hosts = _fake_hosts(a.envs, fake, 0)
        hinfo = "FakeHost (in-process)"
    try:
        res, path = evaluate(panel, policy, tok, hosts, tag=tag, policy_info=info, hosts_info=hinfo,
                             max_episode_decisions=a.max_episode_decisions, round_budget_s=a.round_budget_s)
    finally:
        for h in hosts:
            h.close()
    print(path)
    print(f"{res.completed_episodes}/{res.expected_episodes} episodes; "
          f"{'partial' if res.partial else 'complete'} ({res.stop_reason})")
    print(json.dumps(res.summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

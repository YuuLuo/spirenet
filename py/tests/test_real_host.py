"""Integration with the REAL SpireHost (skipped when its binary / vocab files are absent).

The host is launched from the launch spec (``py/configs/real_host.toml`` or ``$SPIRE_REAL_HOST_CONFIG``,
:func:`spire.host.real_host_spec`). Per transport (stdio, named pipe) a few episodes of uniformly random legal
play through :class:`spire.host.HostProcess`; every frame goes through ``parse_frame(verify_hashes=True)`` + the
header vocab / catalog hashes, the schema 14.5 deep checks, the tokenizer, a forward pass of a small SpireNet
built from the real vocab + catalog, and the reward derivation; the episode accounting must equal the host's
own TERMINAL counters (STEP_ID = GLOBAL[POLICY_DECISIONS] = decisions played, ELITES_WON / BOSSES_WON = the
kills inferred from the frames, TERMINAL_REASON consistent with RESULT) and the host's end-of-serve summary
line. The real vocab / catalog are checked against the tokenizer's limits. Also one short run of the actor
(:class:`spire.actor.Actor`) over named pipes.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.host import HostConfig, HostProcess, RealHostSpec, real_host_spec
from spire.metrics import EpisodeRecord
from spire.model import ModelConfig, SpireNet
from spire.pipes import PipeMux
from spire.proto import Frame, ResetCmd, frame_problems
from spire.rewards import Outcome, RoomIds, episode_targets, frame_facts, inferred_kills
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import G, Result, SeatCol, Table, TerminalReason
from spire.tokenize import Tokenizer
from spire.vocab import DEFAULT_FEATURES, Catalog, Vocab

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)


def _spec() -> RealHostSpec | None:
    try:
        spec = real_host_spec()
    except (OSError, ValueError, KeyError):
        return None
    if spec.missing() or shutil.which(spec.command[0]) is None and not Path(spec.command[0]).is_file():
        return None
    return spec


SPEC = _spec()
pytestmark = pytest.mark.skipif(SPEC is None, reason="real SpireHost binary / vocab not present")


@pytest.fixture(scope="module")
def spec() -> RealHostSpec:
    assert SPEC is not None
    return SPEC


@pytest.fixture(scope="module")
def real(spec: RealHostSpec) -> tuple[Vocab, Catalog]:
    vocab = Vocab.load(spec.vocab)
    return vocab, Catalog.load(spec.catalog, vocab)


def test_real_vocab_and_catalog_fit_the_tokenizer(real: tuple[Vocab, Catalog]) -> None:
    vocab, catalog = real
    tok = Tokenizer(vocab)  # TokenLayout raises if any derived row index would not fit int16
    assert tok.layout.n_rows - 1 <= 32767
    assert max(vocab.sizes().values()) < 32768  # errata E-19: every vocab id fits int16
    for cat in DEFAULT_FEATURES:
        sf = catalog.features(cat)
        assert sf.ids.shape[0] == vocab.size(cat) and np.isfinite(sf.nums).all()
    net = SpireNet(tok.layout, TINY, catalog)
    assert set(net.static.cats) == {"card", "relic", "potion", "power", "monster"}
    for cat in net.static.cats:  # real sentinels (999999999) are clipped like every other feature
        assert float(getattr(net.static, f"x_{cat}").abs().max()) <= 8.0


def _episode(host: HostProcess, rng: np.random.Generator, cmd: ResetCmd, tok: Tokenizer, net: SpireNet,
             rooms: RoomIds) -> tuple[list[Frame], int]:
    f = host.reset_with(cmd)
    frames = [f]
    while not f.is_terminal:
        f = host.step(int(rng.integers(f.n_actions)))
        frames.append(f)
    problems = [(fr.step_id, p) for fr in frames for p in frame_problems(fr)]
    assert not problems, problems[:5]
    obs = frames[:-1]
    samples = [tok.encode(fr) for fr in obs]
    samples.append(tok.encode(frames[-1]))  # TERMINAL frames tokenize too (value bootstrap use)
    with torch.no_grad():
        for i in range(0, len(obs), 32):
            chunk = samples[i : i + 32]
            out = net(tok.collate(chunk))
            n_act = torch.tensor([s.n_actions for s in chunk])
            finite = torch.isfinite(out["logits"]).sum(1)
            assert torch.equal(finite, n_act) and torch.isfinite(out["value"]).all()
    return frames, len(obs)


@pytest.mark.parametrize("transport", ["stdio", "pipe"])
def test_real_host_episodes(spec: RealHostSpec, real: tuple[Vocab, Catalog], transport: str, tmp_path: Path) -> None:
    vocab, catalog = real
    tok = Tokenizer(vocab)
    torch.manual_seed(0)
    net = SpireNet(tok.layout, TINY, catalog).eval()
    rooms = RoomIds.from_vocab(vocab)
    host = HostProcess(HostConfig(command=spec.command, env=spec.process_env(), transport=transport,
                                  name=f"it-{transport}", verify_hashes=True,
                                  expected_hashes=(vocab.hash, catalog.hash), start_timeout=120.0,
                                  step_timeout=60.0, log_path=str(tmp_path / "host.stderr.log")))
    rng = np.random.default_rng(1 if transport == "stdio" else 2)
    chars = [vocab.id("character", c) for c in ("IRONCLAD", "SILENT", "DEFECT")]
    played: list[tuple[int, int, int]] = []
    try:
        jobs = [ResetCmd((c,), asc, None, max_decisions=400) for c, asc in zip(chars, (0, 10, 0))]
        jobs.append(ResetCmd((chars[0],), 0, None, max_decisions=6))  # PY-6: a capped episode
        for cmd in jobs:
            frames, n = _episode(host, rng, cmd, tok, net, rooms)
            t = frames[-1]
            g = t.gvals
            played.append((t.result, g[G.TERMINAL_REASON], n))
            # the host's TERMINAL counters = our accounting (errata P-5 / PY-10, section 5)
            assert t.step_id == g[G.POLICY_DECISIONS] == n == host.last_episode.decisions  # type: ignore[union-attr]
            facts = [frame_facts(fr) for fr in frames]
            tgt = episode_targets(facts, rooms)
            if t.result != Result.ABORTED:
                assert inferred_kills(facts, rooms) == (g[G.ELITES_WON], g[G.BOSSES_WON])
                assert len(tgt.rewards) == n and np.isfinite(tgt.rewards).all()
            if g[G.TERMINAL_REASON] == TerminalReason.DEATH:
                assert tgt.outcome == Outcome.LOSS and g[G.IN_COMBAT] == 0 and t.rows[Table.CREATURES] == 0
                assert int(t.table(Table.SEATS)[0, SeatCol.HP]) == 0  # P-5: SEATS hp 0, no combat tables
            elif g[G.TERMINAL_REASON] == TerminalReason.DECISION_CAP:
                assert tgt.outcome == Outcome.LOSS and n == cmd.max_decisions
                assert tgt.win == 0.0 and tgt.floors_left is not None
                assert tgt.components["terminal"][-1] == pytest.approx(-0.1)
        assert played[-1][1] == TerminalReason.DECISION_CAP and played[-1][2] == 6
    finally:
        host.close()
    s = host.stats
    line = next((x for x in reversed(host.stderr_tail) if "serve done" in x), "")
    m = re.search(r"(\d+) episodes \((\d+) win, (\d+) loss, (\d+) truncated, (\d+) aborted, (\d+) abandoned\), "
                  r"(\d+) decisions, (\d+) errors", line)
    assert m, line
    assert tuple(int(x) for x in m.groups()) == (s.finished, s.wins, s.losses, s.truncated, s.aborted, s.abandoned,
                                               s.decisions, s.errors)
    assert s.finished == len(played) and s.lost == s.bad_frames == s.crashes == s.timeouts == 0


def test_actor_on_real_hosts_over_pipes(spec: RealHostSpec, real: tuple[Vocab, Catalog], tmp_path: Path) -> None:
    """The asynchronous actor path (routing, pipe mux, rollout recording) on two real hosts."""
    vocab, catalog = real
    tok = Tokenizer(vocab)
    mux = PipeMux()
    hosts = [HostProcess(HostConfig(command=spec.command, env=spec.process_env(), transport="pipe", name=f"ia{i}",
                                    expected_hashes=(vocab.hash, catalog.hash), start_timeout=120.0, step_timeout=60.0,
                                    log_path=str(tmp_path / f"a{i}.log")), mux=mux) for i in range(2)]
    recs: list[EpisodeRecord] = []
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
    actor = Actor(hosts, UniformPolicy(0), tok,
                  EpisodeSampler(EpisodeDist(lineups=(("IRONCLAD", 0.5), ("SILENT", 0.5))), vocab, seed=0),
                  ActorConfig(max_episode_decisions=150), store=store, on_episode=recs.append)
    try:
        actor.run(max_decisions=400, max_seconds=120)
    finally:
        for h in hosts:
            h.close()
    ended = [r for r in recs if r.result != "shutdown"]
    # "aborted": the obs build still runs temporary event handlers (a few % of natural runs abort, host-h1-obs)
    assert ended and all(r.result in ("win", "loss", "decision_cap", "aborted") for r in ended)
    assert all(r.terminal_reason in (1, 2, 3, 6) and r.decisions <= 150 for r in ended)
    assert sum(len(s) for s in segs) == store.stats.steps_released > 0
    assert actor.counters.protocol_errors == 0 and all(h.stats.lost == 0 for h in hosts)

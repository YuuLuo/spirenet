from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from ppo_stubs import CorridorEnv, StubNet, StubTokenizer, collect
from spire.learner import FileWeightPublisher, FileWeightSource, Learner, PPOConfig, lr_at
from spire.rewards import RewardConfig, RoomIds
from spire.rollout import GAEConfig, RolloutConfig, RolloutStore, Segment, SegmentRing
from spire.schema import Result
from spire.vocab import Vocab

GAMMA = 0.99
FAST = PPOConfig(epochs=4, minibatch=256, micro_batch=256, lr=3e-3, ent_coef=0.003,
                 gae=GAEConfig(gamma=GAMMA, lam=0.95), seed=0)


class Harness:
    """Corridor envs -> RolloutStore -> SegmentRing -> Learner, all in one thread."""

    def __init__(self, vocab: Vocab, seed: int, cfg: PPOConfig = FAST, n_envs: int = 32, **kw: object) -> None:
        torch.manual_seed(seed)
        self.tok = StubTokenizer()
        self.learner = Learner(StubNet(self.tok.n_features), self.tok, cfg, device="cpu",
                               size_fn=StubTokenizer.size, **kw)  # type: ignore[arg-type]
        self.ring = SegmentRing()
        self.store = RolloutStore(self.ring.put, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16),
                                  rewards=RewardConfig(gamma=cfg.gae.gamma))
        self.envs = [CorridorEnv(vocab, seed * 1000 + i) for i in range(n_envs)]
        self.frames: list = [None] * n_envs
        self.gen = torch.Generator().manual_seed(seed)
        self.results: list[int] = []

    def segments(self, n_samples: int) -> list[Segment]:
        while self.ring.n_samples < n_samples:
            self.results += collect(self.envs, self.frames, self.tok, self.learner.model, self.store,
                                    self.learner.version, 8, self.gen)
        got = self.ring.take(n_samples)
        assert got is not None
        return got

    def step(self, n_samples: int = 1024) -> dict:
        return self.learner.update(self.segments(n_samples))

    def win_rate(self, last: int) -> float:
        r = self.results[-last:]
        return sum(x == Result.WIN for x in r) / max(1, len(r))


def params(m: nn.Module) -> list[Tensor]:
    return [p.detach().clone() for p in m.state_dict().values()]


def strip_times(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("time_") and k not in ("samples_per_s", "lag_s")}


# =========================================================================== learning sanity


def test_ppo_learns_corridor(vocab: Vocab) -> None:
    """Random play wins ~6%; the optimum (always the right option) wins 100% with return ~1.0."""
    h = Harness(vocab, seed=1)
    h.step()
    start = h.win_rate(300)
    for n_updates in range(1, 41):
        m = h.step()
        if len(h.results) > 600 and h.win_rate(300) >= 0.95:
            break
    assert start < 0.15, start
    assert h.win_rate(300) >= 0.95, (n_updates, h.win_rate(300), m)
    for _ in range(3):  # the value head (behaviour values lag one iteration) catches up
        m = h.step()
    assert h.win_rate(300) >= 0.95
    assert m["return_mean"] > 0.9 and abs(m["value_mean"] - m["return_mean"]) < 0.1
    assert math.isfinite(m["loss_v"]) and m["nonfinite_skipped"] == 0 and m["illegal_chosen"] == 0
    assert set(m["entropy_by_kind"]) == {"EVENT", "COMBAT"}
    assert m["staleness_max"] >= 1  # episodes spanning collection rounds carry older versions


def test_determinism_under_seeds(vocab: Vocab) -> None:
    runs = []
    for _ in range(2):
        h = Harness(vocab, seed=3, n_envs=8)
        ms = [strip_times(h.step(256)) for _ in range(3)]
        runs.append((params(h.learner.model), params(h.learner.ema), ms, h.results))
    (p0, e0, m0, r0), (p1, e1, m1, r1) = runs
    assert all(torch.equal(a, b) for a, b in zip(p0, p1))
    assert all(torch.equal(a, b) for a, b in zip(e0, e1))
    assert m0 == m1 and r0 == r1


# =========================================================================== checkpoint / publication


def test_checkpoint_round_trip(vocab: Vocab, tmp_path: Path) -> None:
    a = Harness(vocab, seed=5, n_envs=8)
    a.step(256)
    a.step(256)
    path = tmp_path / "ck" / "learner.pt"
    a.learner.save(path)
    b = Harness(vocab, seed=99, n_envs=8)  # different init, rng, counters
    b.learner.load(path)
    assert (b.learner.version, b.learner.updates, b.learner.samples_seen) == \
        (a.learner.version, a.learner.updates, a.learner.samples_seen)
    assert all(torch.equal(x, y) for x, y in zip(params(a.learner.ema), params(b.learner.ema)))
    segs = a.segments(256)
    ma, mb = strip_times(a.learner.update(segs)), strip_times(b.learner.update(segs))
    assert ma == mb
    assert all(torch.equal(x, y) for x, y in zip(params(a.learner.model), params(b.learner.model)))
    assert all(torch.equal(x, y) for x, y in zip(params(a.learner.ema), params(b.learner.ema)))
    c = Harness(vocab, seed=5, cfg=replace(FAST, lr=1e-4), n_envs=2)
    with pytest.raises(ValueError):
        c.learner.load(path)
    c.learner.load(path, allow_config_change=True)


def test_file_weight_publication(vocab: Vocab, tmp_path: Path) -> None:
    pub = FileWeightPublisher(tmp_path / "w", keep=2)
    src = FileWeightSource(tmp_path / "w")
    assert src.poll(-1) is None and src.latest_version() == -1
    h = Harness(vocab, seed=2, n_envs=4, publisher=pub)
    h.learner.publish()  # version 0 = initial weights
    got = src.poll(-1)
    assert got is not None and got[0] == 0
    for _ in range(3):
        h.step(64)  # publishes versions 1..3
    v, sd = src.poll(0) or (None, None)
    assert v == 3 and src.poll(3) is None
    ref = h.learner.model.state_dict()
    assert sd is not None and all(torch.equal(sd[k], ref[k]) for k in ref)
    assert sorted(p.name for p in (tmp_path / "w").glob("acting-*.pt")) == \
        ["acting-0000000002.pt", "acting-0000000003.pt"]
    fresh = StubNet(h.tok.n_features)
    fresh.load_state_dict(sd)
    bf = FileWeightPublisher(tmp_path / "bf", dtype=torch.bfloat16)
    bf.publish(ref, 7)
    v, sd16 = FileWeightSource(tmp_path / "bf").poll(-1) or (None, None)
    assert v == 7 and sd16 is not None and all(t.dtype == torch.bfloat16 for t in sd16.values())


# =========================================================================== update mechanics


def test_micro_batch_accumulation_matches_full_batch(vocab: Vocab) -> None:
    cfg = replace(FAST, epochs=1, minibatch=100_000, micro_batch=100_000)
    a = Harness(vocab, seed=4, cfg=cfg, n_envs=8)
    segs = a.segments(200)
    b = Harness(vocab, seed=4, cfg=replace(cfg, micro_batch=7, micro_tokens=20), n_envs=8)
    ma, mb = a.learner.update(segs), b.learner.update(segs)
    assert mb["micro_batches"] > 20 and ma["micro_batches"] == 1
    for x, y in zip(params(a.learner.model), params(b.learner.model)):
        torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-6)
    assert ma["loss_pi"] == pytest.approx(mb["loss_pi"], rel=1e-4, abs=1e-7)


def test_kl_early_stop(vocab: Vocab) -> None:
    h = Harness(vocab, seed=6, cfg=replace(FAST, target_kl=1e-12, minibatch=64), n_envs=8)
    m = h.step(512)
    assert m["early_stop"] and m["epochs_done"] < 1.0 and m["minibatches"] == 8


def test_options_paths_run(vocab: Vocab) -> None:
    cfg = replace(FAST, value_clip=0.2, normalize_returns=True, recompute_values=True, ent_normalize=True,
                  lr_schedule="cosine", warmup_updates=1, total_updates=4, weight_decay=0.01,
                  ent_coef_by_kind=((2, 0.0),), gae=GAEConfig(gamma=GAMMA, macro_micro=True))
    h = Harness(vocab, seed=7, cfg=cfg, n_envs=8)
    for _ in range(3):
        m = h.step(256)
    assert all(math.isfinite(m[k]) for k in ("loss_pi", "loss_v", "loss_win", "loss_floors", "loss_hp"))
    assert m["return_std_running"] > 0 and m["lr"] == pytest.approx(lr_at(cfg, 2))


class _NaNValue(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        out = self.inner(batch)
        out["value"] = out["value"] * float("nan")
        return out


def test_nonfinite_minibatches_are_skipped(vocab: Vocab) -> None:
    h = Harness(vocab, seed=8, n_envs=8)
    segs = h.segments(512)
    torch.manual_seed(0)
    learner = Learner(_NaNValue(StubNet(h.tok.n_features)), h.tok, FAST, device="cpu")
    before = params(learner.model)
    m = learner.update(segs)
    assert m["nonfinite_skipped"] == m["minibatches"] * FAST.epochs
    assert all(torch.equal(x, y) for x, y in zip(before, params(learner.model)))


def test_lr_schedules() -> None:
    c = PPOConfig(lr=1.0, lr_schedule="linear", warmup_updates=2, total_updates=12, min_lr_frac=0.1)
    assert [round(lr_at(c, u), 4) for u in (0, 1, 2, 7, 12, 50)] == [0.5, 1.0, 1.0, 0.55, 0.1, 0.1]
    cos = replace(c, lr_schedule="cosine")
    assert lr_at(cos, 7) == pytest.approx(0.55) and lr_at(cos, 12) == pytest.approx(0.1)
    assert lr_at(PPOConfig(lr=0.3), 1000) == 0.3


def test_ema_tracks_online_weights(vocab: Vocab) -> None:
    """EMA with warm-up (py-review 5.1): decay_t = min(decay, (1 + t) / (10 + t)) at EMA step t."""
    h = Harness(vocab, seed=9, cfg=replace(FAST, ema_decay=0.5, epochs=1, minibatch=10_000), n_envs=8)
    start = params(h.learner.model)
    h.step(128)
    online = params(h.learner.model)
    for s, o, e in zip(start, online, params(h.learner.ema)):
        torch.testing.assert_close(e, 0.1 * s + 0.9 * o)  # step 0: decay min(0.5, 1/10) = 0.1
    assert h.learner.ema_steps == 1
    h.step(128)  # step 1: min(0.5, 2/11)
    ema_prev, online_prev, online = params(h.learner.ema), online, params(h.learner.model)
    del ema_prev
    for e1, o0, o1, s in zip(params(h.learner.ema), online_prev, online, start):
        torch.testing.assert_close(e1, (2 / 11) * (0.1 * s + 0.9 * o0) + (9 / 11) * o1)
    assert np.isfinite(h.learner.predict_values([h.tok.encode(h.envs[0].reset())])).all()


def test_ema_warmup_schedule_reaches_the_configured_decay() -> None:
    from spire.trainstep import ema_weight
    assert [round(1 - ema_weight(0.998, t), 4) for t in (0, 1, 10, 100)] == [0.1, round(2 / 11, 4), 0.55, 0.9182]
    assert 1 - ema_weight(0.998, 10_000) == 0.998  # capped at the configured decay
    # the share of the random initialisation left after N steps: ~0 with warm-up, 35% without (py-review 5.1)
    keep = 1.0
    for t in range(520):
        keep *= 1 - ema_weight(0.998, t)
    assert keep < 1e-3 and 0.998 ** 520 > 0.35

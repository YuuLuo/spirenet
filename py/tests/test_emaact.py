"""learner.acting_weights (notes/next/round13-1119.md): actors that act with the learner's EMA weights.

* the published state after K updates is ``ema.state_dict()`` tensor for tensor with "ema" and ``model.state_dict()``
  with "online"; the learner itself (model, EMA, optimizer, counters, metrics) is the same in both modes;
* the refusals of "ema" (one message each) and config.json without the key for "online";
* threads topology end to end on fake hosts, 5 updates: the segments' policy versions advance, the learner rows carry
  ``acting_weights``, the last published weights are the final checkpoint's EMA ("ema") / model ("online");
* the dual learner (two cards): the published state is A's EMA."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from spire.learner import FileWeightPublisher, FileWeightSource, Learner
from spire.metrics import read_jsonl
from spire.train import Supervisor, TrainConfig, build, check_acting, check_dual, config_dict, load_config
from spire.vocab import Vocab

from spire.model import SpireNet
from test_dual import TINY, _cfg, _raw, _segments, _tok, table_path, two_cards  # noqa: F401 (fixture)
from test_learner import Harness, params, strip_times
from test_train import small_config

K = 3


def _published(directory: Path) -> tuple[int, dict[str, torch.Tensor]]:
    got = FileWeightSource(directory).poll(-1)
    assert got is not None
    return got


def _same_state(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert list(a) == list(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].device.type == "cpu", k
        assert torch.equal(a[k], b[k].detach().cpu()), k


# =========================================================================== the learner


def test_published_state_after_k_updates(vocab: Vocab, tmp_path: Path) -> None:
    hs, ms = {}, {}
    for mode in ("online", "ema"):
        h = Harness(vocab, seed=7, n_envs=8, publisher=FileWeightPublisher(tmp_path / mode), acting_weights=mode)
        ms[mode] = [strip_times(h.step(256)) for _ in range(K)]
        hs[mode] = h
        version, sd = _published(tmp_path / mode)
        assert version == h.learner.version == K
        _same_state(sd, (h.learner.ema if mode == "ema" else h.learner.model).state_dict())
        other = (h.learner.model if mode == "ema" else h.learner.ema).state_dict()
        assert any(not torch.equal(sd[k], other[k].cpu()) for k in sd if sd[k].is_floating_point()), mode
    on, ema = hs["online"].learner, hs["ema"].learner
    assert ms["online"] == ms["ema"]  # the learner trains and steps the EMA as before; only the publication differs
    assert all(torch.equal(x, y) for x, y in zip(params(on.model), params(ema.model)))
    assert all(torch.equal(x, y) for x, y in zip(params(on.ema), params(ema.ema)))
    assert (on.version, on.updates, on.samples_seen, on.ema_steps) == \
        (ema.version, ema.updates, ema.samples_seen, ema.ema_steps)
    assert hs["online"].results == hs["ema"].results


def test_learner_refuses_an_unknown_mode(vocab: Vocab) -> None:
    with pytest.raises(AssertionError, match="acting_weights 'mean': 'online' or 'ema'"):
        Harness(vocab, seed=1, n_envs=2, acting_weights="mean")


# =========================================================================== config


REFUSED = [
    ({"ppo": {"kickstart": {"teacher": "t.pt"}}}, "an active kickstart"),
    ({"ppo": {"recompute_values": True}}, "ppo.recompute_values"),
    ({"ppo": {"decoupled_clip_anchor": True}}, "ppo.decoupled_clip_anchor"),
    ({"model": {"critic": {}}, "ppo": {"critic": {}}}, "the critic (ppo.critic / model.critic)"),
]


def _ema_raw(acting: str = "ema", **over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {"run_name": "x", "learner": {"acting_weights": acting}}
    for k, v in over.items():
        raw[k] = {**raw.get(k, {}), **v}
    return raw


@pytest.mark.parametrize("over,message", REFUSED, ids=[m for _, m in REFUSED])
def test_refusals(over: dict[str, Any], message: str) -> None:
    cfg = build(TrainConfig, _ema_raw(**over))
    with pytest.raises(AssertionError, match="learner.acting_weights 'ema' refuses " + message.replace(".", r"\.")
                       .replace("(", r"\(").replace(")", r"\)")):
        check_acting(cfg)
    check_acting(build(TrainConfig, _ema_raw("online", **over)))  # "online": nothing is refused


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(AssertionError, match="learner.acting_weights 'mean': 'online' or 'ema'"):
        check_acting(build(TrainConfig, _ema_raw("mean")))


def test_config_json_and_load_config(tmp_path: Path) -> None:
    off = build(TrainConfig, {"run_name": "x"})
    assert off.learner.acting_weights == "online"
    assert "acting_weights" not in config_dict(off)["learner"]  # "online": the config.json of tree p9
    on = build(TrainConfig, _ema_raw())
    d = config_dict(on)
    assert d["learner"]["acting_weights"] == "ema"
    p = tmp_path / "config.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert load_config(p).learner.acting_weights == "ema"
    p.write_text(json.dumps(_ema_raw(ppo={"recompute_values": True})), encoding="utf-8")
    with pytest.raises(AssertionError, match=r"refuses ppo\.recompute_values"):
        load_config(p)  # load_config runs check_acting


def test_production_dual_shape_is_accepted(tmp_path: Path) -> None:
    raw = _raw(learner={"cuda_graphs": True, "compile": True, "pipeline_prepare": True, "acting_weights": "ema"},
               actor={"device": "cuda", "devices": ["cuda:2", "cuda:3"]}, eval={"device": "cuda:3"})
    raw["learner"]["device"] = "cuda:0"
    cfg = build(TrainConfig, raw)
    check_dual(cfg)
    check_acting(cfg)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config_dict(cfg)), encoding="utf-8")
    got = load_config(p)
    assert got.learner.acting_weights == "ema" and got.learner.replica_device == "cuda:1"


# =========================================================================== threads topology end to end


@pytest.mark.parametrize("mode", ["ema", "online"])
def test_threads_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    seen: list[tuple[int, int, int]] = []  # (learner version, oldest, newest acting version of the segments)
    update = Learner.update

    def recording(self: Learner, segments: Any) -> dict[str, Any]:
        seen.append((self.version, min(int(s.policy_version.min()) for s in segments),
                     max(int(s.policy_version.max()) for s in segments)))
        assert self.acting_weights == mode
        return update(self, segments)

    monkeypatch.setattr(Learner, "update", recording)
    d = config_dict(small_config(tmp_path, "ema-thr", topology="threads", max_updates=5))
    d["learner"]["acting_weights"] = mode
    d["actor"]["weight_poll_interval"] = 0.05  # the actors load each published version promptly
    cfg = build(TrainConfig, d)
    assert Supervisor(cfg).run() == 0
    run = cfg.run_dir
    rows = [r for r in read_jsonl(run / "metrics.jsonl") if r["kind"] == "learner"]
    assert [r["version"] for r in rows] == [1, 2, 3, 4, 5]
    assert all(r["acting_weights"] == mode for r in rows)
    assert len(seen) == 5 and [v for v, _, _ in seen] == [0, 1, 2, 3, 4]
    assert all(hi <= v for v, _, hi in seen)
    assert seen[-1][2] >= 1, seen  # the actors picked up published versions: the segments' versions advance
    assert [hi for _, _, hi in seen] == sorted(hi for _, _, hi in seen), seen
    on_disk = json.loads((run / "config.json").read_text(encoding="utf-8"))["learner"]
    assert on_disk.get("acting_weights", "online") == mode and ("acting_weights" in on_disk) == (mode == "ema")
    version, sd = _published(run / "weights")
    ck = torch.load(run / "ckpt" / "latest.pt", map_location="cpu", weights_only=True)
    assert version == ck["counters"]["version"] == 5
    _same_state(sd, ck["ema" if mode == "ema" else "model"])
    other = ck["model" if mode == "ema" else "ema"]
    assert any(not torch.equal(sd[k], other[k]) for k in sd if sd[k].is_floating_point())


# =========================================================================== dual learner (two cards)


@two_cards
def test_dual_learner_publishes_the_ema_of_a(vocab: Vocab, table_path: str, tmp_path: Path) -> None:  # noqa: F811
    tok = _tok(vocab, "v2a-free", table_path)
    segs = _segments(vocab, "v2a-free", tok)
    for mode in ("ema", "online"):
        torch.manual_seed(0)
        lr = Learner(SpireNet(tok.layout, TINY), tok, _cfg(), device="cuda:0", fast=True, graphs=True,
                     replica_device="cuda:1", publisher=FileWeightPublisher(tmp_path / mode), acting_weights=mode)
        assert lr.replica is not None
        for _ in range(K):
            lr.update(segs)
        version, sd = _published(tmp_path / mode)
        assert version == K and next(lr.ema.parameters()).device == torch.device("cuda:0")
        _same_state(sd, (lr.ema if mode == "ema" else lr.model).state_dict())

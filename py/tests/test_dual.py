"""learner.replica_device (notes/next/dual-spec.md): two learner cards for one run (spire.replica).

* the refusals (one message each), the placement checks and config.json without the key when off;
* D2: every micro-batch on A (the replica idle, kept in sync) = the single-device learner bit for bit (deterministic
  kernels, CUDA graphs, compile off; compile on: within the fast path's tolerance); every micro-batch on B = the all-A
  run within 1e-6 relative on the update;
* D3: the real split = the single learner up to the summation order (per-minibatch total gradient, parameters after
  K = 5 updates against the noise floor of two accumulation orders on one card, every sum metric); deterministic;
* D4: the replica equals A after every update and after a save / load; the checkpoint's key set is the single
  learner's; checkpoints cross between this tree and the base tree (SPIRE_DUAL_BASE = tree p7b);
* D5: the target_kl early stop and the KL brake act at the same minibatch as the single learner;
* D6: a non-finite gradient on B only skips the step exactly as one on A does; the replica equals A afterwards.

The measured numbers of D2 / D3 are printed (``-s``) as one JSON line per check (``DUAL ...``)."""
from __future__ import annotations

import copy
import gc
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy
from spire.learner import Learner, PPOConfig
from spire.model import ModelConfig, SpireNet
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore, Segment
from spire.schema import reset_flags_for
from spire.synth import FakeHost, FakeHostConfig
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.train import TrainConfig, build, check_dual, config_dict, load_config
from spire.vocab import Vocab

from test_removal import synthetic_table

HERE = Path(__file__).resolve().parent
two_cards = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
TIMING = ("time_prep", "time_train", "samples_per_s", "lag_s")
BUCKETS = (20, 28, 40, 64, 80, 96, 128, 192, 256, 384)
PLAIN = dict(public_history=True, cycle_ban=True, commit_select=True)
V2A = dict(forecast=True, calc=True, cycle_ban=True, commit_select=True, map_tokens="no_combat", route_summary=True,
           second_boss_table=True)
VARIANTS = {"plain-mask": (PLAIN, False), "plain-free": (PLAIN, True), "v2a-mask": (V2A, False),
            "v2a-free": (V2A, True)}


# =========================================================================== refusals, placement, config


REFUSED = [
    ({"ppo": {"kickstart": {"teacher": "t.pt"}}}, "an active kickstart"),
    ({"ppo": {"taste": {"file": "t.json", "p_episode": 0.1, "eps": 0.2}}}, "ppo.taste.on"),
    ({"ppo": {"advice": [["REST", "SMITH", 0.5, 0.05]]}}, "ppo.advice"),
    ({"ppo": {"recompute_values": True}}, "ppo.recompute_values"),
    ({"ppo": {"decoupled_clip_anchor": True}}, "ppo.decoupled_clip_anchor"),
    ({"ppo": {"normalize_returns": True}}, "ppo.normalize_returns"),
    ({"learner": {"fast": False}}, "learner.fast false"),
    ({"learner": {"cuda_graphs": False}}, "learner.cuda_graphs false"),
    ({"topology": "threads"}, "topology 'threads'"),
    ({"learner": {"replica_device": "cuda"}}, "a CUDA device cuda:N"),
    ({"learner": {"replica_device": "cpu"}}, "a CUDA device cuda:N"),
    ({"learner": {"device": "cpu"}}, "not CUDA"),
    ({"learner": {"device": "cuda:1"}}, "is the learner's device"),
    ({"actor": {"device": "cuda:1"}}, "is an actor's device"),
    ({"actor": {"devices": ["cuda:2", "cuda:1"]}}, "is an actor's device"),
    ({"eval": {"device": "cuda:1"}}, "is the evaluator's device"),
]


def _raw(**over: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {"run_name": "x", "learner": {"replica_device": "cuda:1", "device": "cuda:0"},
                           "actor": {"device": "cuda:2"}, "eval": {"device": "cuda:3"}}
    for k, v in over.items():
        raw[k] = {**raw.get(k, {}), **v} if isinstance(v, dict) else v
    return raw


@pytest.mark.parametrize("over,message", REFUSED, ids=[f"{i}-{m}" for i, (_, m) in enumerate(REFUSED)])
def test_refusals(over: dict[str, Any], message: str) -> None:
    cfg = build(TrainConfig, _raw(**over))
    with pytest.raises(AssertionError, match=message.replace(".", r"\.").replace("(", r"\(")):
        check_dual(cfg)
    raw = _raw(**over)
    off = build(TrainConfig, {**raw, "learner": {**raw["learner"], "replica_device": ""}})
    check_dual(off)  # off: nothing is refused


def test_critic_is_refused() -> None:
    cfg = build(TrainConfig, _raw(model={"critic": {}}, ppo={"critic": {}}))
    with pytest.raises(AssertionError, match=r"refuses the critic \(ppo\.critic / model\.critic\)"):
        check_dual(cfg)


def test_production_shape_is_accepted_and_config_json(tmp_path: Path) -> None:
    raw = _raw(learner={"cuda_graphs": True, "compile": True, "pipeline_prepare": True},
               actor={"device": "cuda", "devices": ["cuda:2", "cuda:3"]}, eval={"device": "cuda:3"})
    raw["learner"]["device"] = "cuda:0"
    cfg = build(TrainConfig, raw)
    check_dual(cfg)
    d = config_dict(cfg)
    assert d["learner"]["replica_device"] == "cuda:1"
    p = tmp_path / "config.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert load_config(p).learner.replica_device == "cuda:1"  # load_config runs check_dual
    off = build(TrainConfig, {"run_name": "x"})
    assert "replica_device" not in config_dict(off)["learner"]  # off: the config.json of tree p7b
    assert build(TrainConfig, config_dict(off)).learner.replica_device == ""


# =========================================================================== fixtures of the card tests


@pytest.fixture(scope="module")
def table_path(tmp_path_factory: pytest.TempPathFactory, vocab: Vocab) -> str:
    p = tmp_path_factory.mktemp("dual") / "removal-ratings.json"
    p.write_text(json.dumps(synthetic_table(vocab)), encoding="utf-8")
    return str(p)


def _tok(vocab: Vocab, variant: str, table: str) -> Tokenizer:
    flags, free = VARIANTS[variant]
    extra = {"removal_rule": True, "removal_ratings": table, "rules_mode": "free"} if free else {}
    return Tokenizer(vocab, TokenizerConfig(token_buckets=BUCKETS, **flags, **extra))  # type: ignore[arg-type]


_SEGS: dict[str, list[Segment]] = {}


def _segments(vocab: Vocab, variant: str, tok: Tokenizer) -> list[Segment]:
    """Training segments of in-process fake hosts (SampleBlocks), one fixed set per variant."""
    if variant not in _SEGS:
        v2a = VARIANTS[variant][0] is V2A
        segs: list[Segment] = []
        store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=64))
        hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
                 for i in range(12)]
        acfg = ActorConfig(max_batch=32, reset_flags=reset_flags_for(True, calc=True, map_in_combat=True)) if v2a \
            else ActorConfig(max_batch=32)
        Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0),)), vocab, 0), acfg,
              store=store).run(max_decisions=3000)
        assert sum(len(s) for s in segs) > 1500
        _SEGS[variant] = segs
    return _SEGS[variant]


def _cfg(**kw: Any) -> PPOConfig:
    base = dict(epochs=2, minibatch=512, micro_batch=256, micro_tokens=4096, lr=1e-4, seed=3, kl_lr_target=0.01,
                weight_decay=0.01)
    return PPOConfig(**{**base, **kw})  # type: ignore[arg-type]


def _learners(tok: Tokenizer, cfg: PPOConfig, *, replica: bool, n: int = 1, compile_model: bool = False) \
        -> list[Learner]:
    torch.manual_seed(0)
    base = SpireNet(tok.layout, TINY)
    return [Learner(copy.deepcopy(base), tok, cfg, device="cuda:0", fast=True, graphs=True, compile_model=compile_model,
                    replica_device="cuda:1" if replica else None) for _ in range(n)]


@pytest.fixture(autouse=True)
def _collect() -> Any:
    """Free the test's learners before the next test: a CUDA graph destroyed by a garbage collection that runs during
    another graph's capture invalidates that capture."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@pytest.fixture()
def det(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Deterministic CUDA kernels (the base tree's own runs are not reproducible without them: atomic sums)."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def _metrics(m: dict[str, Any], drop: tuple[str, ...] = ()) -> dict[str, Any]:
    return json.loads(json.dumps({k: v for k, v in m.items() if k not in TIMING + ("dual",) + drop}))


def _state(lr: Learner) -> dict[str, Any]:
    return {"model": {k: v.detach().cpu().clone() for k, v in lr.model.state_dict().items()},
            "ema": {k: v.detach().cpu().clone() for k, v in lr.ema.state_dict().items()},
            "opt": {i: {f: (t.detach().cpu().clone() if torch.is_tensor(t) else t) for f, t in s.items()}
                    for i, s in lr.opt.state_dict()["state"].items()},
            "counters": (lr.version, lr.updates, lr.samples_seen, lr.ema_steps),
            "rng": json.dumps(lr.rng.bit_generator.state, sort_keys=True)}


def _bit_equal(a: dict[str, Any], b: dict[str, Any]) -> None:
    for k in ("model", "ema"):
        assert a[k].keys() == b[k].keys()
        for name in a[k]:
            assert torch.equal(a[k][name], b[k][name]), (k, name)
    assert a["opt"].keys() == b["opt"].keys()
    for i in a["opt"]:
        for f in a["opt"][i]:
            x, y = a["opt"][i][f], b["opt"][i][f]
            assert (torch.equal(x, y) if torch.is_tensor(x) else x == y), ("adam", i, f)
    assert a["counters"] == b["counters"] and a["rng"] == b["rng"]


def _replica_is_a(lr: Learner) -> None:
    rep = lr.replica
    assert rep is not None
    for (na, pa), (nb, pb) in zip(lr.model.named_parameters(), rep.model.named_parameters()):
        assert na == nb and torch.equal(pa.detach().cpu(), pb.detach().cpu()), na
    for (na, ba), (nb, bb) in zip(lr.model.named_buffers(), rep.model.named_buffers()):
        assert na == nb and torch.equal(ba.cpu(), bb.cpu()), na
    assert float(rep.gflat.abs().sum()) == 0.0  # B's gradients zeroed after every step
    assert all(p.grad is not None and float(p.grad.abs().sum()) == 0.0 for p in rep.params_a)


def _flat(lr: Learner) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu() for p in lr.model.parameters()])


def _report(**kw: Any) -> None:
    print("DUAL", json.dumps(kw), flush=True)


# =========================================================================== D2


@two_cards
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_d2_all_on_a_is_the_single_learner(vocab: Vocab, table_path: str, variant: str, det: Any) -> None:
    tok = _tok(vocab, variant, table_path)
    segs = _segments(vocab, variant, tok)
    single, = _learners(tok, _cfg(), replica=False)
    dual, = _learners(tok, _cfg(), replica=True)
    assert dual.replica is not None
    dual.replica.split = "a"
    for k in range(3):
        ms, md = single.update(segs), dual.update(segs)
        assert md["dual"]["micro_b"] == 0 and md["dual"]["share_a"] == 1.0
        assert _metrics(ms) == _metrics(md), k
        _replica_is_a(dual)
    _bit_equal(_state(single), _state(dual))
    if VARIANTS[variant][1]:
        assert md["rules"]["mode"] == "free"
    _report(gate="D2-a", variant=variant, bit_equal=True, updates=3, micro=md["micro_batches"])


@two_cards
@pytest.mark.parametrize("variant", ["plain-free", "v2a-free"])
def test_d2_all_on_b_is_the_all_on_a_run(vocab: Vocab, table_path: str, variant: str, det: Any) -> None:
    tok = _tok(vocab, variant, table_path)
    segs = _segments(vocab, variant, tok)
    on_a, on_b = _learners(tok, _cfg(), replica=True, n=2)
    assert on_a.replica is not None and on_b.replica is not None
    on_a.replica.split, on_b.replica.split = "a", "b"
    worst, equal = 0.0, True
    for k in range(3):
        p0a, p0b = _flat(on_a), _flat(on_b)
        ma, mb = on_a.update(segs), on_b.update(segs)
        assert mb["dual"]["micro_a"] == 0 and mb["dual"]["share_a"] == 0.0
        da, db = _flat(on_a) - p0a, _flat(on_b) - p0b
        rel = float((db - da).norm() / da.norm())
        worst = max(worst, rel)
        equal = equal and bool(torch.equal(_flat(on_a), _flat(on_b)))
        assert rel <= 1e-6, (k, rel)
        _replica_is_a(on_b)
    sa, sb = _state(on_a), _state(on_b)
    exact_state = all(torch.equal(sa["model"][n], sb["model"][n]) for n in sa["model"])
    _report(gate="D2-b", variant=variant, max_rel_update=worst, params_bit_equal=equal, state_bit_equal=exact_state,
            metrics_equal=_metrics(ma, ("stepper",)) == _metrics(mb, ("stepper",)))


@two_cards
def test_d2_all_on_a_compiled(vocab: Vocab, table_path: str) -> None:
    """compile on: the fast path's tolerance (test_learner_fast: 2.5 x lr x epochs x 2 per weight, 5 % of the
    displacement); the replica equals A after every update."""
    variant = "v2a-free"
    tok = _tok(vocab, variant, table_path)
    segs = _segments(vocab, variant, tok)
    cfg = _cfg()
    single, = _learners(tok, cfg, replica=False, compile_model=True)
    dual, = _learners(tok, cfg, replica=True, compile_model=True)
    assert dual.replica is not None
    dual.replica.split = "a"
    p0 = _flat(single)
    for _ in range(3):
        ms, md = single.update(segs), dual.update(segs)
        _replica_is_a(dual)
    ds, dd = _flat(single) - p0, _flat(dual) - p0
    tol = 2.5 * cfg.lr * cfg.epochs * 2 * 3
    assert float((dd - ds).abs().max()) <= tol
    rel = float((dd - ds).abs().sum() / ds.abs().sum())
    assert rel < 0.05, rel
    for key in ("loss_pi", "loss_v", "entropy", "approx_kl", "grad_norm", "kl_lr_scale_mean"):
        assert md[key] == pytest.approx(ms[key], rel=0.05, abs=2e-3), key
    _report(gate="D2-compile", variant=variant, max_abs=float((dd - ds).abs().max()), rel_l1=rel,
            bit_equal=bool(torch.equal(_flat(single), _flat(dual))), captures_b=md["dual"]["stepper"]["captures"])


# =========================================================================== D3 / D4


class _Grads:
    """The total gradient of every minibatch (A's ``.grad`` when the clip reads it: after B's were added)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.rec: dict[int, list[torch.Tensor]] = {}
        orig = torch.nn.utils.clip_grad_norm_

        def wrap(params: Any, *a: Any, **k: Any) -> Any:
            if id(params) in self.rec:
                self.rec[id(params)].append(torch.cat([p.grad.detach().reshape(-1) for p in params]).cpu())
            return orig(params, *a, **k)

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", wrap)

    def watch(self, lr: Learner) -> list[torch.Tensor]:
        self.rec[id(lr._params)] = []
        return self.rec[id(lr._params)]


def _rel(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    assert len(a) == len(b) and a
    return max(float((x - y).norm() / y.norm()) for x, y in zip(a, b))


SUMS = ("loss_pi", "loss_v", "loss_win", "loss_floors", "loss_hp", "loss_ent_bonus", "approx_kl", "clipfrac",
        "clipfrac_epoch0", "ratio_dev", "ratio_dev_epoch0", "entropy", "entropy_norm", "entropy_base",
        "first_minibatch_kl", "kl_lr_scale_mean", "kl_lr_scale_min", "grad_norm")
COUNTS = ("samples", "minibatches", "micro_batches", "nonfinite_skipped", "illegal_chosen", "epochs_done",
          "combat_policy_samples")


def _sum_rel(ms: dict[str, Any], md: dict[str, Any]) -> tuple[float, str]:
    """The largest relative difference of the sum metrics and its name (counts and kinds asserted equal)."""
    worst, name = 0.0, ""

    def see(key: str, a: float, b: float) -> None:
        nonlocal worst, name
        r = abs(a - b) / max(abs(a), 1e-12) if a != b else 0.0
        if r > worst:
            worst, name = r, key

    for key in SUMS:
        see(key, ms[key], md[key])
    for kind, e in ms["entropy_by_kind"].items():
        f = md["entropy_by_kind"][kind]
        assert e["samples"] == f["samples"], kind
        see(f"entropy[{kind}]", e["entropy"], f["entropy"])
    if "rules" in ms:
        for key, a in ms["rules"].items():
            b = md["rules"][key]
            if isinstance(a, float):
                see(f"rules.{key}", a, b)
            else:
                assert a == b, ("rules", key)
    return worst, name


@two_cards
@pytest.mark.parametrize("variant,bf16", [("plain-mask", False), ("v2a-free", False), ("plain-free", True),
                                          ("v2a-free", True)])
def test_d3_split_is_the_single_learner_up_to_summation_order(vocab: Vocab, table_path: str, variant: str,
                                                              bf16: bool, det: Any, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    import spire.learner as learner_mod
    tok = _tok(vocab, variant, table_path)
    segs = _segments(vocab, variant, tok)
    cfg = _cfg(bf16=bf16)
    grads = _Grads(monkeypatch)
    single, rev = _learners(tok, cfg, replica=False, n=2)
    dual, dual2 = _learners(tok, cfg, replica=True, n=2)
    gs, gr, gd, gd2 = grads.watch(single), grads.watch(rev), grads.watch(dual), grads.watch(dual2)
    p0 = _flat(single)
    orig_micros = learner_mod.bucket_micros
    worst_sum, worst_floor, first, shares, names = 0.0, 0.0, {}, [], set()
    for k in range(5):
        ms = single.update(segs)
        monkeypatch.setattr(learner_mod, "bucket_micros", lambda *a: orig_micros(*a)[::-1])  # the other order, one card
        mr = rev.update(segs)
        monkeypatch.setattr(learner_mod, "bucket_micros", orig_micros)
        md, md2 = dual.update(segs), dual2.update(segs)
        _replica_is_a(dual)
        assert md["dual"]["micro_a"] > 0 and md["dual"]["micro_b"] > 0
        shares.append(md["dual"]["share_a"])
        assert _metrics(md) == _metrics(md2)  # deterministic
        for key in COUNTS:
            assert ms[key] == md[key], (k, key)
        (w, wn), (f, fn) = _sum_rel(ms, md), _sum_rel(ms, mr)
        if w > 1e-6:
            names.add(wn)
        worst_sum, worst_floor = max(worst_sum, w), max(worst_floor, f)
        if k == 0:  # the first update starts from the same parameters (its later minibatches drift already)
            first = {"dual": w, "dual_key": wn, "floor": f, "floor_key": fn}
    _bit_equal(_state(dual), _state(dual2))
    assert all(torch.equal(a, b) for a, b in zip(gd, gd2))
    g_dual, g_floor = _rel(gd, gs), _rel(gr, gs)
    g_first = float((gd[0] - gs[0]).norm() / gs[0].norm())
    ds = _flat(single) - p0
    p_dual = float((_flat(dual) - _flat(single)).norm() / ds.norm())
    p_floor = float((_flat(rev) - _flat(single)).norm() / ds.norm())
    _report(gate="D3", variant=variant, bf16=bf16, minibatches=len(gs), grad_rel_max=g_dual, grad_rel_floor=g_floor,
            grad_rel_first=g_first, param_rel=p_dual, param_rel_floor=p_floor, sum_rel_max=worst_sum,
            sum_rel_floor=worst_floor, sum_first_update=first, sum_1e5_all_updates=worst_sum <= 1e-5,
            sum_worst=sorted(names), share_a=[round(s, 4) for s in shares])
    if not bf16:
        assert g_dual <= 1e-5, g_dual
    assert g_dual <= 10 * max(g_floor, 1e-7), (g_dual, g_floor)
    assert p_dual <= 10 * max(p_floor, 1e-7), (p_dual, p_floor)
    # the sums against the floor of two accumulation orders on one card (the literal 1e-5 is reported above: the floor
    # itself exceeds it for loss_pi / first_minibatch_kl, notes/next/dual.md D3)
    assert worst_sum <= 10 * max(worst_floor, 1e-7), (worst_sum, worst_floor)
    # D4: a save / load round trip; the checkpoint holds A's state with the single learner's key set
    dual.save(tmp_path / "dual.pt")
    single.save(tmp_path / "single.pt")
    cd = torch.load(tmp_path / "dual.pt", map_location="cpu", weights_only=True)
    cs = torch.load(tmp_path / "single.pt", map_location="cpu", weights_only=True)
    assert _keys(cd) == _keys(cs)
    fresh, = _learners(tok, cfg, replica=True)
    fresh.load(tmp_path / "dual.pt")
    _replica_is_a(fresh)
    _bit_equal(_state(fresh), _state(dual))
    # the next update of the loaded learner: the replica follows (its numbers equal the original's only up to the padding
    # history of a fresh stepper's shapes, in the base tree's single learner too)
    m1 = fresh.update(segs)
    assert m1["dual"]["micro_b"] > 0
    _replica_is_a(fresh)


def _keys(ck: Any, prefix: str = "") -> set[str]:
    """The nested key set of a checkpoint (integer keys, the optimizer's parameter ids, as "#")."""
    if not isinstance(ck, dict):
        return set()
    out: set[str] = set()
    for k, v in ck.items():
        name = f"{prefix}/{'#' if isinstance(k, int) else k}"
        out.add(name)
        out |= _keys(v, name)
    return out


@two_cards
def test_d4_checkpoints_cross_with_the_base_tree(vocab: Vocab, table_path: str, tmp_path: Path) -> None:
    """SPIRE_DUAL_BASE = the base tree (p7b): its single learner loads the dual learner's checkpoint and the dual
    learner loads the base tree's; tests/dual_ref.py prints a digest of the loaded state in both trees."""
    base = os.environ.get("SPIRE_DUAL_BASE")
    if not base:
        pytest.skip("SPIRE_DUAL_BASE (tree p7b) not set")
    variant = "v2a-free"
    tok = _tok(vocab, variant, table_path)
    segs = _segments(vocab, variant, tok)
    seg_file = tmp_path / "segs.pkl"
    seg_file.write_bytes(pickle.dumps(segs, protocol=4))
    dual, = _learners(tok, _cfg(), replica=True)
    dual.update(segs)
    dual.save(tmp_path / "from-dual.pt")

    def ref(tree: str, *argv: str) -> dict[str, Any]:
        env = {**os.environ, "PYTHONPATH": tree, "PYTHONDONTWRITEBYTECODE": "1", "DUAL_TABLE": table_path}
        out = subprocess.run([sys.executable, "-B", str(HERE / "dual_ref.py"), *argv], env=env, capture_output=True,
                             text=True, timeout=1800)
        assert out.returncode == 0, out.stderr[-3000:]
        return json.loads(out.stdout.strip().splitlines()[-1])

    mine = str(HERE.parent)
    a = ref(base, "load", str(tmp_path / "from-dual.pt"), str(seg_file), str(tmp_path / "base-next.pt"))
    b = ref(mine, "load", str(tmp_path / "from-dual.pt"), str(seg_file), str(tmp_path / "dual-next.pt"), "dual")
    assert a["tree_dual"] is False and b["tree_dual"] is True
    assert a["loaded"] == b["loaded"]  # the same state loaded in both trees
    assert b["replica_equal"] is True
    # the base tree's checkpoint (one more update) loads in the dual learner (and the dual's next one equals it)
    c = ref(mine, "load", str(tmp_path / "base-next.pt"), str(seg_file), str(tmp_path / "x.pt"), "dual")
    assert c["replica_equal"] is True and c["loaded"] == a["next"]
    _report(gate="D4-cross", base_loads_dual=True, dual_loads_base=True, next_equal=a["next"] == b["next"])


# =========================================================================== D5 / D6


def _record_lr(lr: Learner) -> list[tuple[float, float]]:
    """(minibatch KL, scale after it) of every minibatch (reads the device: tests only)."""
    rec: list[tuple[float, float]] = []
    orig = lr._minibatch_lr

    def wrap(kl: torch.Tensor, acc: Any) -> None:
        orig(kl, acc)
        rec.append((float(kl), float(acc.lr_scale_min)))

    lr._minibatch_lr = wrap  # type: ignore[method-assign]
    return rec


@two_cards
@pytest.mark.parametrize("bf16", [False, True])
def test_d5_early_stop_and_kl_brake_at_the_same_minibatch(vocab: Vocab, table_path: str, det: Any, bf16: bool) -> None:
    """The gate: the same minibatches brake / the same minibatch stops. fp32: the minibatch KLs and scales within
    1e-5 relative too; bf16: their difference is reported (the summation-order drift of bf16 grows over minibatches)."""
    variant = "v2a-free"
    tok = _tok(vocab, variant, table_path)
    segs = _segments(vocab, variant, tok)
    # the KL brake: the target = the median minibatch KL of an unbraked run (about half the minibatches brake)
    probe, = _learners(tok, _cfg(kl_lr_target=None, lr=1e-3, bf16=bf16), replica=False)
    rp = _record_lr(probe)
    for _ in range(2):
        probe.update(segs)
    brake = float(np.median([kl for kl, _ in rp]))
    single, = _learners(tok, _cfg(kl_lr_target=brake, lr=1e-3, bf16=bf16), replica=False)
    dual, = _learners(tok, _cfg(kl_lr_target=brake, lr=1e-3, bf16=bf16), replica=True)
    rs, rd = _record_lr(single), _record_lr(dual)
    for _ in range(2):
        ms, md = single.update(segs), dual.update(segs)
    assert len(rs) == len(rd) > 4
    braked_s = [kl > brake for kl, _ in rs]
    assert [kl > brake for kl, _ in rd] == braked_s and any(braked_s) and not all(braked_s), braked_s
    worst = max(abs(a[0] - b[0]) / max(a[0], 1e-12) for a, b in zip(rs, rd))
    worst_scale = max(abs(a[1] - b[1]) / a[1] for a, b in zip(rs, rd))
    assert md["kl_lr_scale_min"] < 1
    if not bf16:
        assert worst <= 1e-5 and worst_scale <= 1e-5, (worst, worst_scale)
        assert md["kl_lr_scale_min"] == pytest.approx(ms["kl_lr_scale_min"], rel=1e-5)
    # the early stop: target_kl between the first minibatch's KL and the later ones' (stops inside the first epoch)
    probe, = _learners(tok, _cfg(lr=3e-3, kl_lr_target=None, bf16=bf16), replica=False)
    rp = _record_lr(probe)
    probe.update(segs)
    kls = [kl for kl, _ in rp]
    target = 0.5 * (kls[0] + max(kls[1:]))
    assert any(kl > target for kl in kls[1:]) and kls[0] <= target, kls
    single, = _learners(tok, _cfg(lr=3e-3, kl_lr_target=None, target_kl=target, bf16=bf16), replica=False)
    dual, = _learners(tok, _cfg(lr=3e-3, kl_lr_target=None, target_kl=target, bf16=bf16), replica=True)
    ms, md = single.update(segs), dual.update(segs)
    assert ms["early_stop"] and md["early_stop"] and ms["epochs_done"] == md["epochs_done"] < 2.0
    _replica_is_a(dual)
    _report(gate="D5", bf16=bf16, kl_lr_target=brake, brake_minibatches=len(rs), braked=sum(braked_s), kl_rel_max=worst, scale_rel_max=worst_scale,
            target_kl=target, epochs_done=md["epochs_done"])


@two_cards
def test_d6_nonfinite_gradient_on_b_skips_the_step(vocab: Vocab, table_path: str, det: Any,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    variant = "v2a-free"
    tok = _tok(vocab, variant, table_path)
    segs = _segments(vocab, variant, tok)
    cfg = _cfg()
    single, = _learners(tok, cfg, replica=False)
    on_a, on_b = _learners(tok, cfg, replica=True, n=2)
    J = 2  # the poisoned minibatch (0-based, of the update)
    seen: dict[int, int] = {}
    orig = torch.nn.utils.clip_grad_norm_

    def poison_single(params: Any, *a: Any, **k: Any) -> Any:  # today's case: A's gradient non-finite
        if params is single._params:
            seen[0] = seen.get(0, -1) + 1
            if seen[0] == J:
                params[0].grad.view(-1)[0] = float("nan")
        return orig(params, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", poison_single)

    def poisoned(lr: Learner, side: str) -> None:
        rep = lr.replica
        assert rep is not None
        inner = rep.combine
        count = [-1]

        def combine(grads_a: list[torch.Tensor], kl: torch.Tensor) -> torch.Tensor:
            count[0] += 1
            if count[0] == J:
                (rep.gflat if side == "b" else grads_a[0].view(-1))[0] = float("nan")
            return inner(grads_a, kl)

        rep.combine = combine  # type: ignore[method-assign]

    poisoned(on_a, "a")
    poisoned(on_b, "b")
    ms, ma, mb = single.update(segs), on_a.update(segs), on_b.update(segs)
    assert ms["nonfinite_skipped"] == ma["nonfinite_skipped"] == mb["nonfinite_skipped"] == 1
    _bit_equal(_state(on_a), _state(on_b))  # B's non-finite gradient = A's: the same skipped step
    assert _metrics(ma) == _metrics(mb)
    _replica_is_a(on_b)
    _replica_is_a(on_a)
    d = float((_flat(on_b) - _flat(single)).abs().max())
    assert d <= 2.5 * cfg.lr * cfg.epochs * 2, d
    m2 = on_b.update(segs)  # the next update is finite again
    assert m2["nonfinite_skipped"] == 0
    _replica_is_a(on_b)
    _report(gate="D6", skipped=mb["nonfinite_skipped"], dual_b_vs_single_max_abs=d)

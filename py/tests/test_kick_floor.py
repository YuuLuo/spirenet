"""Tree p7: ``ppo.kickstart.floor`` (notes/next/round4-1118.md D.2). With ``fade_updates`` and a release the factor is 1
until the release, falls linearly to ``floor`` over ``fade_updates`` updates and stays there; the teacher stays
loaded while the factor is above 0. floor = 0 is tree p6's code path (bit for bit: the formula, the PPO config, the
checkpoint counters, the learner row). A resume that lowers the floor starts a second linear fade from the factor in
effect to the new floor over ``fade_updates`` updates from the resume's first update; at 0 the teacher is dropped."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from spire.kickstart import (KickstartConfig, coefficients, fade_factor, load_teacher, refade_after_resume,
                             teacher_active)
from spire.learner import Learner
from spire.model import SpireNet
from spire.rollout import Segment
from spire.tokenize import Tokenizer
from test_kickstart import CFG, P0, TINY, WIDE, _Counting, _size, save_teacher, segments, tok  # noqa: F401

F = 4
R = 7  # the release update


def _p6_factor(u: int, release: int | None) -> float:  # tree p6's formula (kickstart.py of alt/pipe/py)
    if release is None:
        return 1.0
    return max(0.0, 1.0 - (u - release + 1) / F)


def test_factor_at_0_half_1_and_long_after() -> None:
    k0 = KickstartConfig("x", 1.0, 0.5, 4000, phase=1, fade_updates=F)
    k1 = replace(k0, floor=0.1)
    # 0 of the fade (the update before the release's first fade update), 1/2, 1, long after
    us = (R - 1, R + F // 2 - 1, R + F - 1, R + 10 ** 6)
    assert [fade_factor(k0, u, None if u < R else R) for u in us] == [1.0, 0.5, 0.0, 0.0]
    assert [fade_factor(k1, u, None if u < R else R) for u in us] == pytest.approx([1.0, 0.55, 0.1, 0.1], abs=1e-15)
    assert fade_factor(k1, R + F - 1, R) == fade_factor(k1, R + 10 ** 6, R) == 1.0 - 0.9 * 1.0
    assert [fade_factor(k1, u, R) for u in range(R, R + F)] == pytest.approx([0.775, 0.55, 0.325, 0.1], abs=1e-15)
    # floor 0: tree p6's numbers, bit for bit, at every update
    assert all(fade_factor(k0, u, r) == _p6_factor(u, r) for r in (None, 0, 3, R) for u in range(max(r or 0, 0), 40))
    assert coefficients(k1, R + 100, R) == (pytest.approx(0.1), pytest.approx(0.05))
    assert teacher_active(k1, R + 10 ** 6, R) and not teacher_active(k0, R + F - 1, R)
    # floor in [0, 1), and only with phase 1 + fade_updates
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(AssertionError, match="floor must be"):
            replace(k0, floor=bad)
    with pytest.raises(AssertionError, match="needs phase 1 and fade_updates"):
        KickstartConfig("x", floor=0.1)
    with pytest.raises(AssertionError, match="needs phase 1 and fade_updates"):
        replace(k1, phase=0)


def test_refade_formula() -> None:
    k1 = KickstartConfig("x", 1.0, 0.5, 4000, phase=1, fade_updates=F, floor=0.1)
    k0 = replace(k1, floor=0.0)
    # a resume at u0 = 30 (long after the fade: factor 0.1) into floor 0: 0.1 -> 0 over F updates from u0
    ref = refade_after_resume(k1, k0, 30, R, None)
    assert ref == (30, fade_factor(k1, 29, R))
    assert [fade_factor(k0, u, R, ref) for u in (29, 30, 31, 32, 33, 34, 10 ** 6)] == \
        pytest.approx([0.1, 0.075, 0.05, 0.025, 0.0, 0.0, 0.0], abs=1e-15)
    assert fade_factor(k0, 33, R, ref) == 0.0 and not teacher_active(k0, 33, R, ref) and teacher_active(k0, 32, R, ref)
    # a resume in the middle of the first fade: from the factor in effect (u0 - 1 = 8: 0.55)
    ref2 = refade_after_resume(k1, k0, 9, R, None)
    assert ref2 == (9, pytest.approx(0.55)) and fade_factor(k0, 12, R, ref2) == 0.0
    # before the release: no second fade, the new floor is the fade's
    assert refade_after_resume(k1, k0, 3, None, None) is None
    # the same floor: the checkpoint's own second fade continues; a raised floor is refused
    assert refade_after_resume(k0, k0, 31, R, ref) == ref
    with pytest.raises(AssertionError, match="only lower the floor"):
        refade_after_resume(k0, k1, 40, R, ref)
    with pytest.raises(AssertionError):
        fade_factor(k0, 27, R, ref)  # before u0 - 1: not a factor of this schedule


def test_refade_a_floor_changed_before_the_release() -> None:  # tree p7b
    k_ph0 = KickstartConfig("x", 1.0, 0.5, 4000, phase=0, fade_updates=F)
    k1 = KickstartConfig("x", 1.0, 0.5, 4000, phase=1, fade_updates=F, floor=0.1)
    k0 = replace(k1, floor=0.0)
    assert refade_after_resume(k_ph0, k1, 50, None, None) is None  # the phase switch with a floor
    assert refade_after_resume(k0, k1, 50, None, None) is None  # raised before the release
    assert refade_after_resume(k1, k0, 50, None, None) is None  # lowered before the release (as in tree p7)
    with pytest.raises(AssertionError, match="after the release may only lower the floor"):
        refade_after_resume(k0, k1, 60, 51, None)


def _learner(tok: Tokenizer, path: Path, release: Path, floor: float) -> tuple[Learner, _Counting]:
    cfg = replace(CFG, epochs=1, kickstart=KickstartConfig(str(path), 1.0, 0.5, 3, phase=1, fade_updates=F,
                                                           floor=floor))
    teacher, sha = load_teacher(path, tok, None, TINY)
    counting = _Counting(teacher)
    return Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cpu", size_fn=_size(tok), teacher=counting,
                   teacher_sha256=sha, release_file=release), counting


def test_teacher_kept_at_the_floor_then_a_second_fade_drops_it(tok: Tokenizer, segments: list[Segment],  # noqa: F811
                                                               tmp_path: Path) -> None:
    segs = segments[:12]
    torch.manual_seed(2)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, WIDE), tok)
    release = tmp_path / "KICK_RELEASE"
    torch.manual_seed(0)
    lr, counting = _learner(tok, path, release, 0.1)
    assert json.loads(json.dumps(lr.cfg.to_dict()))["kickstart"]["floor"] == 0.1
    for u in range(2):
        r = lr.update(segs)["kickstart"]
        assert (r["u"], r["pi_coef"], r["teacher_evaluated"], r["release_update"]) == (u, 1.0, True, None)
    release.write_text("")
    rows = [lr.update(segs)["kickstart"] for _ in range(F + 6)]
    release.unlink()
    assert [r["pi_coef"] for r in rows] == pytest.approx([0.775, 0.55, 0.325] + [0.1] * 7, abs=1e-12)
    assert [r["v_coef"] for r in rows] == pytest.approx([0.3875, 0.275, 0.1625] + [0.05] * 7, abs=1e-12)
    assert all(r["teacher_evaluated"] and r["release_update"] == 2 and "refade" not in r for r in rows)
    calls = counting.calls
    lr.update(segs)
    assert counting.calls > calls and lr.teacher is not None  # kept at the floor
    lr.save(tmp_path / "floor.pt")
    ck = torch.load(tmp_path / "floor.pt", weights_only=True)
    assert ck["config"]["kickstart"]["floor"] == 0.1 and "refade" not in ck["counters"]["kickstart"]
    # the state survives a save and a resume (same floor): the same coefficients, the teacher still evaluated
    lr_b, cnt_b = _learner(tok, path, release, 0.1)
    lr_b.load(tmp_path / "floor.pt")
    assert (lr_b.kick_u, lr_b.kick_release, lr_b.kick_refade) == (lr.kick_u, 2, None)
    r = lr_b.update(segs)["kickstart"]
    assert (r["u"], r["pi_coef"], r["teacher_evaluated"]) == (13, pytest.approx(0.1), True) and cnt_b.calls > 0
    # floor 0.1 -> 0 on a resume: a second linear fade 0.1 -> 0 over F updates from the first update after the resume
    lr0, cnt0 = _learner(tok, path, release, 0.0)
    lr0.load(tmp_path / "floor.pt")
    assert lr0.kick_refade == (13, pytest.approx(0.1))
    rows = [lr0.update(segs)["kickstart"] for _ in range(F + 2)]
    assert [r["u"] for r in rows] == list(range(13, 13 + F + 2))
    assert [r["pi_coef"] for r in rows] == pytest.approx([0.075, 0.05, 0.025, 0.0, 0.0, 0.0], abs=1e-12)
    assert [r["teacher_evaluated"] for r in rows] == [True, True, True, False, False, False]
    assert all(r["refade"] == [13, pytest.approx(0.1)] for r in rows)
    assert lr0.teacher is None
    lr0.save(tmp_path / "zero.pt")
    ck0 = torch.load(tmp_path / "zero.pt", weights_only=True)
    assert "floor" not in ck0["config"]["kickstart"] and ck0["counters"]["kickstart"]["refade"] == [13, pytest.approx(0.1)]
    # the second fade's state survives a resume too
    lr0b, _ = _learner(tok, path, release, 0.0)
    lr0b.load(tmp_path / "zero.pt")
    assert lr0b.kick_refade == lr0.kick_refade
    r = lr0b.update(segs)["kickstart"]
    assert (r["pi_coef"], r["teacher_evaluated"]) == (0.0, False) and lr0b.teacher is None
    # raising the floor on a resume is refused
    up, _ = _learner(tok, path, release, 0.2)
    with pytest.raises(AssertionError, match="only lower the floor"):
        up.load(tmp_path / "floor.pt")
    # switching the kickstart off: refused while the floor keeps the teacher, allowed after the second fade
    off = Learner(SpireNet(tok.layout, TINY), tok, CFG, device="cpu")
    with pytest.raises(AssertionError, match="switched off"):
        off.load(tmp_path / "floor.pt", allow_config_change=True)
    off.load(tmp_path / "zero.pt", allow_config_change=True)


def test_floor_raised_on_a_resume_before_the_release(tok: Tokenizer, segments: list[Segment],  # noqa: F811
                                                     tmp_path: Path) -> None:
    """Tree p7b (1) and (3): floor 0.0 -> 0.1 on a resume before the release is accepted (no refade) and the fade after
    the release ends at 0.1 and stays; a raise after the release is refused."""
    segs = segments[:12]
    torch.manual_seed(2)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, WIDE), tok)
    release = tmp_path / "KICK_RELEASE"
    torch.manual_seed(0)
    lr, _ = _learner(tok, path, release, 0.0)
    for _ in range(2):
        lr.update(segs)
    lr.save(tmp_path / "f0.pt")
    up, _ = _learner(tok, path, release, 0.1)
    up.load(tmp_path / "f0.pt")
    assert (up.kick_u, up.kick_release, up.kick_refade) == (2, None, None)
    r = up.update(segs)["kickstart"]
    assert (r["u"], r["pi_coef"], r["v_coef"], r["release_update"]) == (2, 1.0, 0.5, None) and "refade" not in r
    release.write_text("")
    rows = [up.update(segs)["kickstart"] for _ in range(F + 3)]
    release.unlink()
    want = [1.0 - 0.9 * min(1.0, (u - 3 + 1) / F) for u in range(3, 3 + F + 3)]
    assert [r["u"] for r in rows] == list(range(3, 3 + F + 3))
    assert [r["pi_coef"] for r in rows] == pytest.approx(want, abs=1e-12)
    assert [r["v_coef"] for r in rows] == pytest.approx([0.5 * w for w in want], abs=1e-12)
    assert rows[-1]["pi_coef"] == rows[-2]["pi_coef"] == pytest.approx(0.1, abs=1e-12)
    assert all(r["release_update"] == 3 and r["teacher_evaluated"] and "refade" not in r for r in rows)
    assert up.teacher is not None
    # (3) a raise after the release: refused (from floor 0.1 to 0.2, and from floor 0 to 0.1)
    up.save(tmp_path / "f1.pt")
    up2, _ = _learner(tok, path, release, 0.2)
    with pytest.raises(AssertionError, match="after the release may only lower the floor"):
        up2.load(tmp_path / "f1.pt")
    release.write_text("")
    lr.update(segs)
    release.unlink()
    lr.save(tmp_path / "r0.pt")
    up3, _ = _learner(tok, path, release, 0.1)
    with pytest.raises(AssertionError, match="after the release may only lower the floor"):
        up3.load(tmp_path / "r0.pt")


def test_floor_set_with_the_phase_switch(tok: Tokenizer, segments: list[Segment],  # noqa: F811
                                         tmp_path: Path) -> None:
    """Tree p7b (2): phase 0 -> 1 and floor 0.0 -> 0.1 in one resume (allow_config_change, as the phase switch alone):
    the state is the checkpoint's, no refade, factor 1.0 until the release, then the fade to 0.1."""
    torch.manual_seed(2)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, WIDE), tok)
    release = tmp_path / "KICK_RELEASE"
    ks0 = KickstartConfig(str(path), 1.0, 0.5, 4000, phase=0, fade_updates=F, smooth=((2, 0.05),))
    cfg0 = replace(P0, kickstart=ks0)
    cfg1 = replace(cfg0, kickstart=replace(ks0, phase=1, floor=0.1))
    teacher, sha = load_teacher(path, tok, None, TINY)
    torch.manual_seed(0)
    lr0 = Learner(SpireNet(tok.layout, TINY), tok, cfg0, device="cpu", size_fn=_size(tok), teacher=teacher,
                  teacher_sha256=sha, release_file=release)
    for _ in range(2):
        assert lr0.update(segments)["kickstart"]["phase"] == 0
    lr0.save(tmp_path / "p0.pt")
    lr1 = Learner(SpireNet(tok.layout, TINY), tok, cfg1, device="cpu", size_fn=_size(tok), teacher=teacher,
                  teacher_sha256=sha, release_file=release)
    with pytest.raises(ValueError, match="config differs"):
        lr1.load(tmp_path / "p0.pt")  # the phase switch needs allow_config_change, as in tree p7
    lr1.load(tmp_path / "p0.pt", allow_config_change=True)
    assert (lr1.kick_u, lr1.kick_release, lr1.kick_refade, lr1.updates) == (2, None, None, 2)
    m = lr1.update(segments)
    k = m["kickstart"]
    assert (k["phase"], k["u"], k["pi_coef"], k["v_coef"], k["teacher_evaluated"]) == (1, 2, 1.0, 0.5, True)
    assert m["loss_pi"] != 0.0 and "refade" not in k
    release.write_text("")
    rows = [lr1.update(segments)["kickstart"] for _ in range(F + 2)]
    release.unlink()
    assert [r["pi_coef"] for r in rows] == pytest.approx([1.0 - 0.9 * min(1.0, j / F) for j in range(1, F + 3)],
                                                         abs=1e-12)
    assert all(r["release_update"] == 3 for r in rows)
    lr1.save(tmp_path / "p1.pt")
    ck = torch.load(tmp_path / "p1.pt", weights_only=True)
    assert ck["config"]["kickstart"]["floor"] == 0.1 and ck["counters"]["kickstart"]["phase"] == 1
    assert "refade" not in ck["counters"]["kickstart"]


def test_floor_zero_keeps_the_p6_config_and_counters(tok: Tokenizer, segments: list[Segment],  # noqa: F811
                                                     tmp_path: Path) -> None:
    torch.manual_seed(2)
    path = save_teacher(tmp_path / "t.pt", SpireNet(tok.layout, WIDE), tok)
    torch.manual_seed(0)
    lr, _ = _learner(tok, path, tmp_path / "KICK_RELEASE", 0.0)
    d = lr.cfg.to_dict()
    assert "floor" not in d["kickstart"]
    assert set(d["kickstart"]) == {"teacher", "pi_coef", "v_coef", "updates", "phase", "fade_updates", "smooth"}
    r = lr.update(segments[:12])["kickstart"]
    assert set(r) >= {"u", "phase", "pi_coef", "v_coef", "teacher_evaluated", "release_update"} and "refade" not in r
    lr.save(tmp_path / "a.pt")
    ck = torch.load(tmp_path / "a.pt", weights_only=True)
    assert set(ck["counters"]["kickstart"]) == {"u", "teacher", "teacher_sha256", "phase", "release_update"}


def test_train_config_json_without_floor(tmp_path: Path) -> None:
    from spire.train import TrainConfig, build, config_dict
    assert "floor" not in config_dict(build(TrainConfig, {"run_name": "a"}))["ppo"]["kickstart"]
    on = build(TrainConfig, {"run_name": "a", "ppo": {"kickstart": {"teacher": "x", "fade_updates": 10, "floor": 0.1}}})
    assert on.ppo.kickstart.floor == 0.1 and config_dict(on)["ppo"]["kickstart"]["floor"] == 0.1

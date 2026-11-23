"""Kickstarting a new network (the STUDENT) from a frozen existing one (the TEACHER): notes/solo/kickstart-spec.md v1
and notes/solo/kickstart-spec-v2.md.

For every minibatch the learner also runs the teacher (no gradient, evaluation mode, the student's precision) on the
same collated batch and adds to the loss

    kick = pi_coef * mean over policy steps of KL(pi_T' || pi_S)  +  v_coef * mean over valid steps of
           0.5 * (V_S - V_T)^2

with pi_T / pi_S the BASE policies (softmax of the action logits over the step's legal rows, the student's mask for
both networks, before any exploration mixture or taste term), pi_T' = (1 - eps) pi_T + eps uniform on the steps whose
decision kind is listed in ``smooth`` (else pi_T), the PPO policy-step mask / the valid mask and the minibatch
denominators of the policy and value losses. From the update whose coefficients are 0 on the teacher is not
evaluated at all. Config ``ppo.kickstart`` (:class:`KickstartConfig`; ``teacher`` empty = off). The teacher is a
checkpoint of this code family: its EMA weights and its own model config, asserted to be fed what the student is fed
(:func:`load_teacher`).

Phases (v2): ``phase = 1`` (default) = v1: the student acts, PPO plus the kick term. ``phase = 0``: the actors act with
the teacher (spire.train.acting_policy) and the learner trains the student by the kick term plus the value / auxiliary
losses only (no PPO surrogate, entropy bonus, uniform barrier, early stop or KL brake; one epoch), with constant
coefficients. Schedule in phase 1: with ``fade_updates`` unset v1's linear fade over ``updates`` student updates; with
it set both coefficients stay at their start values until the file ``runs/<run>/KICK_RELEASE`` exists, the first
update that sees it is ``release_update`` and the coefficients fall linearly to 0 over ``fade_updates`` updates from
there (:func:`fade_factor`). ``floor`` (tree p7; with ``fade_updates``): the fade ends at that share instead of 0 and
stays there, the teacher stays loaded; a resume with a LOWER floor starts a second linear fade from the factor in
effect to the new floor over ``fade_updates`` updates from the resume's first update (``refade`` = (that update, the
factor), checkpoint counter ``kickstart.refade``); a second fade to 0 drops the teacher at its end.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .distributions import kind_coefficients, legal_uniform_log_probs
from .model import ModelConfig, SpireNet

if TYPE_CHECKING:
    from .tokenize import Tokenizer
    from .vocab import Catalog

N_ACC = 10
"""The device sums of a kickstarted update: [KL, (V_S - V_T)^2, arg-max agreements, policy steps, valid steps, and
(phase 0) the returns, their squares, the student's value errors (return - V_S), their squares, the loss]."""
A_RET, A_RET2, A_ERR, A_ERR2, A_LOSS = range(5, N_ACC)


@dataclass(frozen=True)
class KickstartConfig:
    teacher: str = ""  # checkpoint path; "" = off
    pi_coef: float = 1.0
    v_coef: float = 0.5
    updates: int = 4000  # phase 1 without fade_updates: student updates (from the run's first update) of the fade
    phase: int = 1  # 0 = the teacher acts and the student only distils; 1 = the student acts (PPO + kick term)
    fade_updates: int | None = None  # phase 1: constant until runs/<run>/KICK_RELEASE, then this long a fade
    smooth: tuple[tuple[int, float], ...] = ()  # (decision kind, eps): pi_T' = (1 - eps) pi_T + eps uniform
    floor: float = 0.0  # phase 1 with fade_updates: the factor falls to this share and stays there (tree p7)

    def __post_init__(self) -> None:
        assert self.pi_coef >= 0 and self.v_coef >= 0, "kickstart coefficients must be nonnegative"
        assert self.updates > 0, "kickstart.updates must be positive"
        assert self.phase in (0, 1), f"kickstart.phase must be 0 or 1, not {self.phase}"
        assert self.fade_updates is None or self.fade_updates > 0, "kickstart.fade_updates must be positive"
        kind_coefficients(self.smooth)  # decision kinds, nonnegative, no duplicates
        assert all(eps <= 1.0 for _, eps in self.smooth), "kickstart.smooth eps must be <= 1"
        assert 0.0 <= self.floor < 1.0, f"kickstart.floor must be in [0, 1), not {self.floor}"
        assert self.floor == 0.0 or (self.phase == 1 and self.fade_updates is not None), \
            "kickstart.floor needs phase 1 and fade_updates"

    @property
    def on(self) -> bool:
        return bool(self.teacher)


def fade_factor(cfg: KickstartConfig, u: int, release: int | None,
                refade: tuple[int, float] | None = None) -> float:
    """The share of the start coefficients in effect at student update ``u`` (0-based); ``release`` = the update that
    first saw the release file (None: not yet). Phase 0: 1. Phase 1 without ``fade_updates``: v1's
    ``max(0, 1 - u / updates)``. With ``fade_updates``: 1 until the release, then ``max(0, 1 - (u - release + 1) /
    fade_updates)`` - the update that sees the file is the first one of the fade, the ``fade_updates``-th one has 0;
    with ``floor`` > 0 ``1 - (1 - floor) min(1, (u - release + 1) / fade_updates)``. ``refade`` = (u0, f0) (a resume
    that lowered the floor, tree p7): ``f0 - (f0 - floor) min(1, (u - u0 + 1) / fade_updates)`` from u0 - 1 on."""
    if cfg.phase == 0:
        return 1.0
    if cfg.fade_updates is None:
        return max(0.0, 1.0 - u / cfg.updates)
    if release is None:
        assert refade is None, refade
        return 1.0
    assert u >= release, (u, release)
    if refade is not None:
        u0, f0 = refade
        assert u0 > release and u >= u0 - 1 and cfg.floor < f0 <= 1.0, (u, release, refade, cfg.floor)
        return f0 - (f0 - cfg.floor) * min(1.0, (u - u0 + 1) / cfg.fade_updates)
    if cfg.floor == 0.0:
        return max(0.0, 1.0 - (u - release + 1) / cfg.fade_updates)
    return 1.0 - (1.0 - cfg.floor) * min(1.0, (u - release + 1) / cfg.fade_updates)


def coefficients(cfg: KickstartConfig, u: int, release: int | None = None,
                 refade: tuple[int, float] | None = None) -> tuple[float, float]:
    """``(pi_coef, v_coef)`` at student update ``u`` (0-based) since the kickstart began."""
    f = fade_factor(cfg, u, release, refade)
    return cfg.pi_coef * f, cfg.v_coef * f


def teacher_active(cfg: KickstartConfig, u: int, release: int | None = None,
                   refade: tuple[int, float] | None = None) -> bool:
    """The teacher is evaluated at update ``u`` (it is dropped from the first update whose coefficients are 0)."""
    return cfg.on and fade_factor(cfg, u, release, refade) > 0.0


def refade_after_resume(saved: KickstartConfig, cfg: KickstartConfig, u: int, release: int | None,
                        refade: tuple[int, float] | None) -> tuple[int, float] | None:
    """The second fade of a resume at student update ``u`` (the resume's first update) from a checkpoint of floor
    ``saved.floor`` into ``cfg.floor`` (tree p7): unchanged floor = the checkpoint's ``refade``; a changed floor before
    the release, raised or lowered (tree p7b), = None: no fade has begun, the new floor is the floor of the fade to come
    (e.g. a phase-0 checkpoint, floor 0, resumed in phase 1 with a floor); after the release a lower floor = (u, the
    factor of update u - 1) and a raised floor is refused."""
    if saved.floor == cfg.floor:
        return refade
    if release is None:
        assert refade is None, refade
        return None
    assert cfg.floor < saved.floor, \
        f"kickstart.floor {saved.floor} -> {cfg.floor}: a resume after the release may only lower the floor"
    return (u, fade_factor(saved, u - 1, release, refade))


def kick_terms(s_logits: Tensor, s_value: Tensor, t_logits: Tensor, t_value: Tensor,
               eps: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Per step: KL(pi_T' || pi_S) in nats over the student's legal rows, (V_S - V_T)^2 and whether the two arg-max
    rows agree. ``eps`` (per step, or None = 0 everywhere): pi_T' = (1 - eps) pi_T + eps uniform over the step's legal
    rows on steps with eps > 0 and more than one legal row, else pi_T. A step with one legal row has KL exactly 0;
    illegal rows contribute nothing."""
    legal = torch.isfinite(s_logits)
    ls = legal_uniform_log_probs(s_logits)  # illegal rows 0
    t_masked = t_logits.masked_fill(~legal, float("-inf"))
    lt = legal_uniform_log_probs(t_masked)
    if eps is not None:
        n = legal.sum(-1, keepdim=True)
        e = eps[:, None].to(lt.dtype)
        smoothed = torch.logaddexp(lt + torch.log1p(-e), torch.log(e) - torch.log(n.to(lt.dtype)))
        lt = torch.where(legal & (e > 0) & (n > 1), smoothed, lt)
    kl = (torch.where(legal, lt.exp(), 0.0) * (lt - ls)).sum(-1)
    dv2 = (s_value - t_value).square()
    agree = s_logits.argmax(-1) == t_masked.argmax(-1)
    return kl, dv2, agree


def smooth_table(cfg: KickstartConfig, device: torch.device | str) -> Tensor | None:
    """Per decision kind eps of ``cfg.smooth`` (device tensor), None when nothing is listed."""
    return torch.as_tensor(kind_coefficients(cfg.smooth), device=device) if cfg.smooth else None


def kick_loss(out: dict[str, Tensor], pmf: Tensor, valid: Tensor, den_pol: Tensor | float, den_all: Tensor | float,
              pi_coef: Tensor | float, v_coef: Tensor | float, acc: Tensor, eps: Tensor | None = None) -> Tensor:
    """The kick term of one micro-batch: ``out`` = the student's (float) outputs plus ``teacher_logits`` /
    ``teacher_value``; ``pmf`` / ``valid`` = the float policy-step / valid masks of the PPO policy / value losses,
    ``den_pol`` / ``den_all`` their minibatch denominators, ``eps`` the per-step smoothing (:func:`kick_terms`). Adds
    the micro-batch's sums [KL, (V_S - V_T)^2, arg-max agreements, policy steps, valid steps] to ``acc[:5]`` in place
    (capturable)."""
    kl, dv2, agree = kick_terms(out["logits"], out["value"], out["teacher_logits"], out["teacher_value"], eps)
    loss = pi_coef * (kl * pmf).sum() / den_pol + v_coef * 0.5 * (dv2 * valid).sum() / den_all
    with torch.no_grad():
        acc[:5].add_(torch.stack([(kl * pmf).sum(), (dv2 * valid).sum(), (agree.float() * pmf).sum(), pmf.sum(),
                                  valid.sum()]))
    return loss


@torch.no_grad()
def distil_sums(value: Tensor, ret: Tensor, valid: Tensor, loss: Tensor, acc: Tensor) -> None:
    """Phase 0: adds the sums of the student's held-out explained variance (returns, their squares, the errors
    return - V_S and their squares over the valid steps) and the micro-batch's loss to ``acc[5:]`` (capturable)."""
    err = ret - value
    acc[A_RET:].add_(torch.stack([(ret * valid).sum(), (ret.square() * valid).sum(), (err * valid).sum(),
                                  (err.square() * valid).sum(), loss.detach()]))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_config(tokenizer: dict[str, Any], vocab_hash: int, catalog_hash: int, seat_symmetry: bool) -> dict[str, Any]:
    """What decides what a network is fed: every tokenizer key (forecast, removal rule, calc inputs, ...) except the
    removal table's path (the learner feeds the teacher the student's samples, whose removal masks the student's table
    made), the vocabulary / catalog hashes and seat symmetry."""
    from .tokenize import TokenizerConfig
    tok = json.loads(json.dumps(asdict(TokenizerConfig(**tokenizer))))
    del tok["removal_ratings"]
    return {"tokenizer": tok, "vocab_hash": int(vocab_hash), "catalog_hash": int(catalog_hash),
            "seat_symmetry": bool(seat_symmetry)}


def load_teacher(path: str | Path, tok: "Tokenizer", catalog: "Catalog | None",
                 student: ModelConfig) -> tuple[SpireNet, str]:
    """The teacher network of checkpoint ``path`` (its EMA weights, its own model config; evaluation mode, no
    gradients, on the CPU) and the checkpoint's sha256. Asserts that the teacher was fed what the student is fed."""
    # spire.rules: the teacher reads the same ban slots; a kickstarted run is in "mask"
    assert tok.config.rules_mode in ("mask", "free"), f"a kickstart teacher needs tokenizer.rules_mode mask or free, not {tok.config.rules_mode}"
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    assert ck.get("format") == 1, f"{path}: not a learner checkpoint"
    extra = ck["extra"]
    from .inputs2_migration import KEYS
    from .tokenize import TokenizerConfig
    t_keys = [getattr(TokenizerConfig(**extra["tokenizer"]), k) for k in KEYS]
    s_keys = [getattr(tok.config, k) for k in KEYS]
    # notes/inputs2/v2a.md 1: no dual feed - an active teacher has the run's map_tokens / route_summary /
    # second_boss_table (a student is migrated only after its teacher was dropped)
    assert t_keys == s_keys, f"kickstart teacher {path}: tokenizer {KEYS} {t_keys} != the run's {s_keys}"
    mcfg = ModelConfig(**extra["model_config"])
    t_in = input_config(extra["tokenizer"], extra["vocab_hash"], extra["catalog_hash"], mcfg.seat_symmetry)
    s_in = input_config(asdict(tok.config), tok.layout.vocab_hash, 0 if catalog is None else catalog.hash,
                        student.seat_symmetry)
    assert t_in == s_in, f"kickstart teacher {path} is fed differently from the student: teacher {t_in} != student {s_in}"
    with torch.random.fork_rng(devices=[]):  # the teacher's (overwritten) initialisation draws nothing of the run's
        net = SpireNet(tok.layout, mcfg, catalog)
    net.load_state_dict(ck["ema"])
    return net.eval().requires_grad_(False), file_sha256(path)

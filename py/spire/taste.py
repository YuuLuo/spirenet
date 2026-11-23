"""Taste: per-episode card-family exploration at CARD_REWARD decisions (notes/solo/taste-spec.md v1 + v1.1).

Why: the trained policy never takes whole card families (draw, the multiplier half of every engine), so the fight
policy never plays decks built around them and the value head cannot price them; a single uniform pick (the
``exploration_by_kind`` mixture) does not assemble an engine that needs several pieces. Taste is a DATA mechanism:
whole games commit to one family, as pure behaviour exploration with exact PPO accounting (spec v1.1 A).

Taste of an (episode, seat) (:meth:`Taste.begin`): three 64-bit values of one keyed hash of (taste seed, the
episode id, seat) mapped to [0, 1): below the character's ``p_episode`` the seat has a taste, the family is chosen
uniformly from the character's families by the second value, and the taste is "thin" when the third is below
``p_thin``. The episode id = (run seed, run start count, actor slot, actor incarnation, env id, the env's episode
count): unique across actors and restarts, and the hash is independent of the sampler's character / ascension draw.
Characters missing from the table (or with p_episode 0) never get a taste.

Target rows F(s) at a CARD_REWARD step of a seat with a taste (:func:`target_rows`): the PICK_OFFER_CARD rows that the
sampler can take (its own legal mask: illegal and cycle-banned rows are out) whose card is in the family and whose
copies in the acting seat's deck (plain and upgraded together) are below the card's cap; in a thin taste, when that
set is empty and the seat's deck holds at least ``thin_deck`` cards, F(s) = {the card reward's Skip row}. Below total
floor ``start_floor`` (default 0 = off) taste is inactive (spec v1.2 J). With F(s)
not empty the behaviour is ``b = (1 - eps) q + eps U_F`` (q: the distribution the actor samples today, the policy with
the ``exploration_by_kind`` mixture), sampled exactly so: the policy's own draw from q is kept with probability
``1 - eps``, else replaced by a uniform row of F (an actor-owned generator: the policy's random stream is the one of a
run without taste). Every other step is unchanged.

Accounting (spec 3): per step ``t_keep`` = 1 - eps where taste was active, else 1, and ``t_add`` = eps [a in F] / |F|
(else 0). The stored behaviour log-prob is ``log b(a) = logaddexp(log q(a) + log t_keep, log t_add)`` and the learner
evaluates the same formula on its current q (:func:`taste_log_prob`, both learner paths); on steps without taste it
is the identity. ``decoupled_clip_anchor`` and ``recompute_values`` must be off with taste (PPOConfig asserts).

Table (spec 5): ``{"version": <int >= 1>, "characters": {CHAR: {"families": {name: {"cards": {CARD_ID: cap}}}}}}``.
Off by default: with ``PPOConfig.taste`` absent / every p_episode 0 nothing here runs (no generator, no records).
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .metrics import COUNTED
from .proto import Frame
from .schema import ActionCol, CardCol, G, Kind, OptionCol, OptionKind, SeatCol, Table, Verb, Zone
from .vocab import Vocab

_PICK = int(Verb.PICK_OFFER_CARD)
_CHOOSE = int(Verb.CHOOSE_OPTION)
_CARD_REWARD = int(Kind.CARD_REWARD)
_DECK = int(Zone.DECK)
_ALT = int(OptionKind.CARD_REWARD_ALT)
NONE = "none"
THIN = "+thin"


@dataclass(frozen=True)
class TasteConfig:
    file: str = ""  # taste table (JSON, module docstring)
    p_episode: float | Mapping[str, float] = 0.0  # per seat-episode; a number or character name -> number (absent: 0)
    eps: float = 0.0  # mass of the uniform target pick where taste is active
    seed: int = 0  # taste hash key (with the run seed)
    p_thin: float = 0.5  # share of taste episodes that are thin
    thin_deck: int = 24  # thin: Skip is the target from this deck size on
    start_floor: int = 0  # taste is inactive at decisions with total floor below this (0 = from the start)

    def __post_init__(self) -> None:
        ps = list(self.p_episode.values()) if isinstance(self.p_episode, Mapping) else [self.p_episode]
        assert all(isinstance(p, (int, float)) and 0.0 <= p <= 1.0 for p in ps), f"taste.p_episode {self.p_episode}"
        assert 0.0 <= self.eps < 1.0, f"taste.eps {self.eps} outside [0, 1)"
        assert 0.0 <= self.p_thin <= 1.0 and self.thin_deck >= 0 and self.start_floor >= 0, \
            (self.p_thin, self.thin_deck, self.start_floor)
        assert not self.on or (self.file and self.eps > 0.0), "taste on (a p_episode > 0) needs a table file and eps > 0"

    @property
    def on(self) -> bool:
        ps = self.p_episode.values() if isinstance(self.p_episode, Mapping) else [self.p_episode]
        return any(p > 0 for p in ps)

    def p_for(self, character: str) -> float:
        if isinstance(self.p_episode, Mapping):
            return float(self.p_episode.get(character, 0.0))
        return float(self.p_episode)


@dataclass(frozen=True)
class Family:
    name: str
    cards: np.ndarray  # card vocab ids, sorted
    caps: np.ndarray  # copy cap per card (same order)


def load_table(path: str, vocab: Vocab) -> tuple[int, dict[int, tuple[Family, ...]]]:
    """(table version, character vocab id -> its families in table order); every name must be in the vocabulary."""
    raw = json.loads(open(path, encoding="utf-8").read())
    v = raw["version"]
    assert set(raw) <= {"version", "characters", "note", "comment"} and type(v) is int and v >= 1, \
        f"taste table {path}: keys {sorted(raw)}, version {v}"
    chars, cards = vocab["character"].entries, vocab["card"].entries
    out: dict[int, tuple[Family, ...]] = {}
    for char, spec in raw["characters"].items():
        cid = vocab.id("character", char)
        assert chars[cid] == char, f"taste table: unknown character {char}"
        assert set(spec) == {"families"} and spec["families"], f"taste table {char}: {sorted(spec)}"
        fams = []
        for name, fam in spec["families"].items():
            assert set(fam) == {"cards"} and fam["cards"], f"taste table {char}/{name}: {sorted(fam)}"
            ids = [vocab.id("card", c) for c in fam["cards"]]
            bad = [c for c, i in zip(fam["cards"], ids) if cards[i] != c]
            assert not bad, f"taste table {char}/{name}: unknown cards {bad}"
            caps = list(fam["cards"].values())
            assert all(type(c) is int and c >= 1 for c in caps), f"taste table {char}/{name}: caps {caps}"
            assert name != NONE and "|" not in name and not name.endswith(THIN), name
            order = np.argsort(ids)
            fams.append(Family(name, np.asarray(ids, np.int64)[order], np.asarray(caps, np.int64)[order]))
        out[cid] = tuple(fams)
    return v, out


def _deck(f: Frame, seat: int) -> np.ndarray:
    cards = f.table(Table.CARDS)
    return cards[(cards[:, CardCol.ZONE] == _DECK) & (cards[:, CardCol.SEAT] == seat)]


def family_rows(f: Frame, seat: int, fam: Family, legal: np.ndarray) -> np.ndarray:
    """Legal PICK_OFFER_CARD rows offering a card of ``fam`` that ``seat``'s deck holds fewer copies of than its cap."""
    acts = f.actions
    picks = np.flatnonzero((acts[:, ActionCol.VERB] == _PICK) & legal[:len(acts)])
    if not len(picks):
        return picks
    assert (acts[picks, ActionCol.SRC_TABLE] == Table.CARDS).all()
    cid = f.table(Table.CARDS)[acts[picks, ActionCol.SRC_ROW], CardCol.CARD_ID].astype(np.int64)
    pos = np.searchsorted(fam.cards, cid).clip(0, len(fam.cards) - 1)
    inside = fam.cards[pos] == cid
    if not inside.any():
        return picks[inside]
    deck = _deck(f, seat)
    copies = np.asarray([int(deck[deck[:, CardCol.CARD_ID] == c, CardCol.COUNT].sum()) for c in cid[inside]])
    return picks[inside][copies < fam.caps[pos[inside]]]


def skip_rows(f: Frame, legal: np.ndarray, skip_key: int) -> np.ndarray:
    acts = f.actions
    rows = np.flatnonzero((acts[:, ActionCol.VERB] == _CHOOSE) & legal[:len(acts)])
    o = f.table(Table.OPTIONS)[acts[rows, ActionCol.SRC_ROW]]
    out = rows[(o[:, OptionCol.KIND] == _ALT) & (o[:, OptionCol.KEY_ID] == skip_key)]
    assert len(out) <= 1, "one Skip per card reward"
    return out


def target_rows(f: Frame, seat: int, fam: Family, thin: bool, thin_deck: int, legal: np.ndarray,
                skip_key: int) -> tuple[np.ndarray, bool]:
    """(F(s), whether it is the thin rule's Skip) at a CARD_REWARD frame of ``seat`` with taste ``fam``."""
    rows = family_rows(f, seat, fam, legal)
    if len(rows) or not thin or int(f.table(Table.SEATS)[seat, SeatCol.DECK_COUNT]) < thin_deck:
        return rows, False
    return skip_rows(f, legal, skip_key), True


def taste_log_prob(logq: Tensor, t_keep: Tensor, t_add: Tensor) -> Tensor:
    """log b(a) = logaddexp(log q(a) + log t_keep, log t_add) (spec 3)."""
    return torch.logaddexp(logq + t_keep.log(), t_add.log())


def taste_learner_metrics(kinds: np.ndarray, t_keep: np.ndarray, t_add: np.ndarray, adv: np.ndarray
                          ) -> dict[str, Any]:
    """Learner row (spec v1.1 E): CARD_REWARD steps and their mean raw GAE advantage in three groups: "target" (taste
    took a target row, t_add > 0), "own" (taste active, the policy's own pick: t_keep < 1, t_add = 0), "off"."""
    cr = kinds == _CARD_REWARD
    out: dict[str, Any] = {"steps": int((t_keep < 1).sum())}
    for name, m in (("target", cr & (t_add > 0)), ("own", cr & (t_keep < 1) & (t_add == 0)), ("off", cr & (t_keep == 1))):
        n = int(m.sum())
        out[name] = {"n": n, "adv_mean": round(float(adv[m].mean()), 5) if n else None}
    return out


def episode_values(key: Sequence[int], seat: int) -> tuple[float, float, float]:
    """Three independent uniform [0, 1) values of (taste key, seat)."""
    h = hashlib.blake2b(struct.pack(f"<{len(key) + 1}q", *key, seat), digest_size=24, person=b"spire-taste").digest()
    a, b, c = struct.unpack("<3Q", h)
    return a / 2.0 ** 64, b / 2.0 ** 64, c / 2.0 ** 64


@dataclass
class _Seat:
    character: int
    family: int  # index into the character's families, -1 = none
    thin: bool
    active: int = 0  # steps where taste was active (F not empty, floor >= start_floor)
    family_taken: int = 0  # taken actions in F at family steps
    skips: int = 0  # taken Skips of the thin rule


@dataclass
class Taste:
    """Actor side (module docstring): per env the seats' tastes, the steering and the per-report counts."""

    cfg: TasteConfig
    vocab: Vocab
    key: tuple[int, ...]  # (taste seed, run seed, run start, actor slot, actor incarnation): hash key of this actor
    table: dict[int, tuple[Family, ...]] = field(init=False)
    rng: np.random.Generator = field(init=False)
    seats: dict[int, list[_Seat]] = field(init=False, default_factory=dict)
    n_episodes: dict[int, int] = field(init=False, default_factory=dict)  # per env id
    episodes: list[list[Any]] = field(init=False, default_factory=list)
    base_pi: dict[str, list[float]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        assert self.cfg.on
        self.version, self.table = load_table(self.cfg.file, self.vocab)
        self.rng = np.random.default_rng((*self.key, 0x7A57E))
        self.log_keep = math.log1p(-self.cfg.eps)
        self.chars = self.vocab["character"].entries
        self.skip_key = self.vocab.id("card_reward_alt", "SKIP")
        assert self.vocab["card_reward_alt"].entries[self.skip_key] == "SKIP"

    def begin(self, env_id: int, characters: Sequence[int]) -> None:
        assert env_id not in self.seats, f"env {env_id}: taste of an episode that did not end"
        k = self.n_episodes.get(env_id, 0)
        self.n_episodes[env_id] = k + 1
        seats = []
        for s, c in enumerate(characters):
            u, v, w = episode_values((*self.key, env_id, k), s)
            fams = self.table.get(int(c), ())
            on = bool(fams) and u < self.cfg.p_for(self.chars[int(c)])
            seats.append(_Seat(int(c), int(v * len(fams)) if on else -1, on and w < self.cfg.p_thin))
        self.seats[env_id] = seats

    def steer(self, env_id: int, f: Frame, logits: np.ndarray, logq: np.ndarray, action: int
              ) -> tuple[int, float, float, float]:
        """(action, log b(action), t_keep, t_add) of one decision; ``logits``: the sampler's scores (-inf = not
        takeable), ``logq``: log q over the rows."""
        g = f.gvals
        if g[G.KIND] != _CARD_REWARD:
            return action, float(logq[action]), 1.0, 0.0
        seat = int(g[G.ACTING_SEAT])
        st = self.seats[env_id][seat]
        assert int(f.table(Table.SEATS)[seat, SeatCol.CHARACTER_ID]) == st.character, "seat / job character mismatch"
        legal = np.isfinite(logits)
        if st.family < 0:
            self._base_pi(f, seat, st.character, logits, legal)
            return action, float(logq[action]), 1.0, 0.0
        if int(g[G.TOTAL_FLOOR]) < self.cfg.start_floor:
            return action, float(logq[action]), 1.0, 0.0
        rows, thin = target_rows(f, seat, self.table[st.character][st.family], st.thin, self.cfg.thin_deck, legal,
                                 self.skip_key)
        if not len(rows):
            return action, float(logq[action]), 1.0, 0.0
        eps = self.cfg.eps
        if self.rng.random() < eps:
            action = int(rows[self.rng.integers(len(rows))])
        st.active += 1
        t_add = eps / len(rows) if bool((rows == action).any()) else 0.0
        if t_add:
            st.skips += thin
            st.family_taken += not thin
        log_add = math.log(t_add) if t_add else -math.inf
        return action, float(np.logaddexp(self.log_keep + float(logq[action]), log_add)), 1.0 - eps, t_add

    def _base_pi(self, f: Frame, seat: int, character: int, logits: np.ndarray, legal: np.ndarray) -> None:
        """A "none" seat at a CARD_REWARD: per family the base policy's probability (no mixing) of its offered rows."""
        fams = self.table.get(character, ())
        if not fams:
            return
        z = np.where(legal, logits, -np.inf)
        p = np.exp(z - z.max())
        p /= p.sum()
        for fam in fams:
            rows = family_rows(f, seat, fam, legal)
            if len(rows):
                a = self.base_pi.setdefault(f"{self.chars[character]}|{fam.name}", [0.0, 0])
                a[0] += float(p[rows].sum())
                a[1] += 1

    def end(self, env_id: int, result: str, floor: int, ascension: int, bosses_won: int) -> None:
        seats = self.seats.pop(env_id)
        if result not in COUNTED:
            return
        act1_death = int(result == "loss" and bosses_won == 0)  # died before the first act boss was passed
        for st in seats:
            group = NONE if st.family < 0 else self.table[st.character][st.family].name + (THIN if st.thin else "")
            self.episodes.append([self.chars[st.character], group, int(ascension), int(result == "win"), int(floor),
                                  st.active, st.family_taken, st.skips, act1_death])

    def take(self) -> dict[str, Any]:
        """Since the last call: finished seat-episodes [char, group, ascension, win, floor, active steps, family cards
        taken, thin Skips taken, act-1 death] and the base-pi sums of "none" seats per "CHAR|family" [sum, frames]."""
        out = {"episodes": self.episodes, "base_pi": self.base_pi}
        self.episodes, self.base_pi = [], {}
        return out


class TasteStats:
    """Supervisor side: the train-row table of spec v1.1 E from the actors' :meth:`Taste.take` reports (cumulative
    since this supervisor started, and a window of the last ``window`` finished seat-episodes per key)."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.total: dict[str, list[int]] = {}
        self.recent: dict[str, deque[tuple[int, int, int]]] = {}
        self.pi: dict[str, list[float]] = {}

    def add(self, rep: Mapping[str, Any]) -> None:
        for char, group, asc, win, floor, active, fam, skips, act1 in rep["episodes"]:
            k = f"{char}|{group}|A{asc}"
            t = self.total.setdefault(k, [0] * 7)
            for i, x in enumerate((1, win, floor, active, fam, skips, act1)):
                t[i] += x
            self.recent.setdefault(k, deque(maxlen=self.window)).append((win, floor, act1))
        for k, (s, n) in rep["base_pi"].items():
            a = self.pi.setdefault(k, [0.0, 0])
            a[0] += s
            a[1] += n

    def summary(self) -> dict[str, Any]:
        groups: dict[str, Any] = {}
        none: dict[str, Any] = {}
        for k, t in sorted(self.total.items()):
            w = self.recent[k]
            n = len(w)
            row = {"n": t[0], "win_rate": round(t[1] / t[0], 4), "mean_floor": round(t[2] / t[0], 2),
                   "window_n": n, "window_win_rate": round(sum(x[0] for x in w) / n, 4),
                   "window_mean_floor": round(sum(x[1] for x in w) / n, 2),
                   "act1_death_share": round(t[6] / t[0], 4),
                   "window_act1_death_share": round(sum(x[2] for x in w) / n, 4),
                   "active_steps": t[3], "family_taken": t[4], "thin_skips": t[5]}
            groups[k] = row
            char, group, asc = k.split("|")
            if group == NONE:  # the curriculum's input (solo_curriculum.py --taste): lineup|A<asc>, outcome-window form
                none[f"{char}|{asc}"] = {"n": n, "win_rate": row["window_win_rate"],
                                         "mean_floor": row["window_mean_floor"]}
        return {"groups": groups, "none_outcomes": none,
                "base_pi_none": {k: {"frames": int(n), "mean": round(s / n, 5)} for k, (s, n) in sorted(self.pi.items())}}

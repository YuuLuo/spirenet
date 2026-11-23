"""Test-only stand-ins for spire.tokenize / spire.model plus a frame-based toy MDP.

* :class:`StubTokenizer` / :class:`StubNet` honour the stage-2 contract:
  ``encode(frame) -> Sample``, ``collate(list[Sample]) -> dict of tensors``,
  ``forward(batch) -> {"logits" (B, A_max; illegal = -inf), "value", "pwin_logit",
  "floors_pred", "hp_pred"}``.
* :func:`obs_frame` / :func:`terminal_frame` hand-build minimal valid frames with
  ``spire.synth.FrameBuilder`` (GLOBAL kind/floor/room/in_combat, SEATS hp, OPTIONS+ACTIONS).
* :class:`CorridorEnv` - a contextual MDP spoken in frames: ``floors`` event floors then a
  boss combat; at each decision the correct option is ``context % n_options`` (context in
  GLOBAL CUR_COL). Correct -> next floor (boss: win); wrong -> lose ``hp_loss`` HP, death at 0.
* :func:`collect` - a synchronous actor loop feeding :class:`spire.rollout.RolloutStore`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from spire.proto import Frame, parse_frame
from spire.rewards import frame_facts
from spire.rollout import RolloutStore
from spire.schema import FrameFlag, G, Kind, MsgType, OptionKind, Result, SeatCol, Table, TerminalReason, Verb
from spire.synth import FrameBuilder
from spire.vocab import Vocab


# --------------------------------------------------------------------------- frames


def obs_frame(kind: int, floor: int, room: int, in_combat: bool, hp: Sequence[int], max_hp: Sequence[int] | None = None,
              *, n_actions: int = 2, context: int = 0, overflow: bool = False, elites: int = 0,
              bosses: int = 0) -> Frame:
    """``elites`` / ``bosses``: the host's win counters GLOBAL[ELITES_WON / BOSSES_WON] (errata-v1 section 5)."""
    b = FrameBuilder(MsgType.OBS)
    _common(b, kind, floor, room, in_combat, hp, max_hp, elites, bosses)
    b.glob[G.CUR_COL] = context
    for i in range(n_actions):
        b.add(Table.OPTIONS, kind=OptionKind.EVENT_OPTION, seat=0, pos=i)
        b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=i)
    if overflow:
        b.flags |= FrameFlag.OVERFLOW
    return parse_frame(b.build(hashes=False))


def terminal_frame(result: int, floor: int, room: int, hp: Sequence[int], max_hp: Sequence[int] | None = None,
                   *, elites: int = 0, bosses: int = 0, reason: int | None = None, in_combat: bool = False,
                   steps: int = 0) -> Frame:
    """TERMINAL as the real host writes it: ``reason`` = GLOBAL[TERMINAL_REASON] (default WIN / DEATH /
    ABORTED by ``result``), GLOBAL[POLICY_DECISIONS] = STEP_ID = ``steps``."""
    b = FrameBuilder(MsgType.TERMINAL)
    _common(b, Kind.TERMINAL, floor, room, in_combat, hp, max_hp, elites, bosses)
    b.flags |= FrameFlag.TERMINAL
    b.result = result
    b.glob[G.TERMINAL_REASON] = reason if reason is not None else {
        Result.WIN: TerminalReason.WIN, Result.LOSS: TerminalReason.DEATH}.get(Result(result), TerminalReason.ABORTED)
    b.glob[G.POLICY_DECISIONS] = b.step_id = steps
    return parse_frame(b.build(hashes=False))


def _common(b: FrameBuilder, kind: int, floor: int, room: int, in_combat: bool, hp: Sequence[int],
            max_hp: Sequence[int] | None, elites: int = 0, bosses: int = 0) -> None:
    mx = list(max_hp) if max_hp is not None else [100] * len(hp)
    b.glob[G.KIND], b.glob[G.N_SEATS] = kind, len(hp)
    b.glob[G.TOTAL_FLOOR], b.glob[G.ROOM_TYPE], b.glob[G.IN_COMBAT] = floor, room, int(in_combat)
    b.glob[G.ELITES_WON], b.glob[G.BOSSES_WON] = elites, bosses
    for s, (h, m) in enumerate(zip(hp, mx)):
        b.add(Table.SEATS, seat=s, hp=h, max_hp=m, alive=int(h > 0), creature_row=-1)


def room_ids(vocab: Vocab) -> dict[str, int]:
    return {n: vocab.id("enum:RoomType", n) for n in ("Monster", "Elite", "Boss", "Event", "Map")}


# --------------------------------------------------------------------------- tokenizer / model


@dataclass(frozen=True)
class StubSample:
    x: np.ndarray
    n_actions: int


class StubTokenizer:
    def __init__(self, n_ctx: int = 4, a_max: int = 4) -> None:
        self.n_ctx, self.a_max = n_ctx, a_max
        self.n_features = n_ctx + a_max + 1 + 3

    def encode(self, frame: Frame) -> StubSample:
        g = frame.glob
        x = np.zeros(self.n_features, np.float32)
        x[int(g[G.CUR_COL]) % self.n_ctx] = 1.0
        x[self.n_ctx + min(frame.n_actions, self.a_max)] = 1.0
        seats = frame.table(Table.SEATS)
        x[-3] = float(g[G.TOTAL_FLOOR]) / 10.0
        x[-2] = float(seats[0, SeatCol.HP]) / max(1, int(seats[0, SeatCol.MAX_HP]))
        x[-1] = float(g[G.IN_COMBAT])
        return StubSample(x, frame.n_actions)

    def collate(self, samples: Sequence[StubSample]) -> dict[str, Tensor]:
        return {"x": torch.from_numpy(np.stack([s.x for s in samples])),
                "n_actions": torch.tensor([s.n_actions for s in samples], dtype=torch.int64)}

    @staticmethod
    def size(sample: StubSample) -> int:
        return sample.n_actions


class StubNet(nn.Module):
    def __init__(self, n_features: int, a_max: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.a_max = a_max
        self.trunk = nn.Sequential(nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, a_max)
        self.heads = nn.Linear(hidden, 4)  # value, pwin_logit, floors_pred, hp_pred
        nn.init.zeros_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        h = self.trunk(batch["x"])
        legal = torch.arange(self.a_max, device=h.device)[None, :] < batch["n_actions"][:, None]
        logits = self.pi(h).masked_fill(~legal, float("-inf"))
        v, pw, fl, hp = self.heads(h).unbind(-1)
        return {"logits": logits, "value": v, "pwin_logit": pw, "floors_pred": fl, "hp_pred": hp}


# --------------------------------------------------------------------------- toy MDP


class CorridorEnv:
    def __init__(self, vocab: Vocab, seed: int, *, floors: int = 3, n_ctx: int = 4, hp_loss: int = 50) -> None:
        self.rng = np.random.default_rng(seed)
        self.rooms = room_ids(vocab)
        self.floors, self.n_ctx, self.hp_loss = floors, n_ctx, hp_loss
        self.floor = self.hp = self.ctx = self.n = 0

    @property
    def boss(self) -> bool:
        return self.floor > self.floors

    def reset(self) -> Frame:
        self.floor, self.hp = 1, 100
        return self._draw()

    def _draw(self) -> Frame:
        self.ctx = int(self.rng.integers(self.n_ctx))
        self.n = int(self.rng.integers(2, 5))
        if self.boss:
            return obs_frame(Kind.COMBAT, self.floor, self.rooms["Boss"], True, [self.hp], n_actions=self.n,
                             context=self.ctx)
        return obs_frame(Kind.EVENT, self.floor, self.rooms["Event"], False, [self.hp], n_actions=self.n,
                         context=self.ctx)

    def correct(self) -> int:
        return self.ctx % self.n

    def step(self, action: int) -> Frame:
        if action == self.correct():
            if self.boss:  # the boss fight is won: the host's counter goes up on the TERMINAL frame
                return terminal_frame(Result.WIN, self.floor, self.rooms["Boss"], [self.hp], bosses=1)
            self.floor += 1
        else:
            self.hp -= self.hp_loss
            if self.hp <= 0:
                room = self.rooms["Boss"] if self.boss else self.rooms["Event"]
                return terminal_frame(Result.LOSS, self.floor, room, [0])
        return self._draw()


def collect(envs: list[CorridorEnv], frames: list[Frame | None], tok: StubTokenizer, model: nn.Module,
            store: RolloutStore, version: int, n_steps: int, gen: torch.Generator) -> list[int]:
    """Step all envs in lockstep for ``n_steps`` rounds; returns the results of finished episodes."""
    results: list[int] = []
    for i, env in enumerate(envs):
        if frames[i] is None:
            frames[i] = env.reset()
            store.begin_episode(i, (2,), 0, seed_id=i)
    model.eval()
    for _ in range(n_steps):
        samples = [tok.encode(f) for f in frames if f is not None]
        with torch.no_grad():
            out = model(tok.collate(samples))
        dist = torch.distributions.Categorical(logits=out["logits"])
        acts = torch.multinomial(dist.probs, 1, generator=gen).squeeze(1)
        logps = dist.log_prob(acts)
        for i, env in enumerate(envs):
            f = frames[i]
            assert f is not None
            store.add_step(i, frame_facts(f), samples[i], int(acts[i]), float(logps[i]), float(out["value"][i]), version)
            nf = env.step(int(acts[i]))
            if nf.is_terminal:
                store.end_episode(i, frame_facts(nf))
                results.append(nf.result)
                nf = env.reset()
                store.begin_episode(i, (2,), 0, seed_id=i)
            frames[i] = nf
    model.train()
    return results

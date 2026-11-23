"""Reference run for tests/test_rules.py R1 / R2 (rules out 1.2, notes/next/p4.md); PYTHONPATH = the tree under test.

RULES_MODE: base (tree p3, no rules modes), mask (this tree, rules_mode left at its default), teach0 (rules_mode
"teach" with ppo.rules.coef_removal 0). A fixed-seed CPU actor (TorchPolicy, tokenizer flags cycle_ban +
public_history + commit_select + the removal rule with RULES_TABLE, exploration mixture; three fake hosts with shop /
event removal prompts (four), one fake menu host whose rewards are declined and re-opened, so cycles close, and one fake rest
site whose SMITH opens an upgrade prompt that CANCEL leaves) plays 3,600 decisions; one eager learner update follows. Prints one JSON line: hashes of the fixed-seed initialisation (keys and
values), of the samples (with and without the ban slots), of every recorded action / log-prob / value / kind / legal
count, of the updated parameters and the update's loss terms."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cycle_integration import _host  # noqa: E402  (the fake menu game: cycles close)

from spire.proto import Command, QuitCmd, ResetCmd, StepCmd  # noqa: E402
from spire.schema import (CardSelectSub, FrameFlag, G, MsgType, OptionKind, Result, Table,  # noqa: E402
                          TerminalReason, Verb, Zone)
from spire.synth import FakeGame, FrameBuilder  # noqa: E402
from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, TorchPolicy  # noqa: E402
from spire.learner import Learner, PPOConfig  # noqa: E402
from spire.model import ModelConfig, SpireNet  # noqa: E402
from spire.rewards import RoomIds  # noqa: E402
from spire.rollout import RolloutConfig, RolloutStore  # noqa: E402
from spire.schema import Kind  # noqa: E402
from spire.synth import FakeHost, FakeHostConfig  # noqa: E402
from spire.tokenize import CYCLE_BAN_SLOTS, Tokenizer, TokenizerConfig  # noqa: E402
from spire.vocab import Vocab, make_synthetic_vocab  # noqa: E402



class _SmithGame(FakeGame):
    """REST (SMITH, PROCEED) -> DECK_UPGRADE grid (two cards, CANCEL back to REST) -> CONFIRM step (CONFIRM, CANCEL) ->
    REST without SMITH; PROCEED ends the game (a death); 14 decisions at most."""

    def __init__(self, vocab: Vocab) -> None:
        super().__init__(vocab)
        self.step, self.screen, self.smith = 0, "rest", True

    def handle(self, cmd: Command) -> list[bytes]:
        if isinstance(cmd, QuitCmd):
            return []
        if isinstance(cmd, ResetCmd):
            self.episode_id += 1
            self.step, self.screen, self.smith = 0, "rest", True
        else:
            assert isinstance(cmd, StepCmd) and cmd.step_id == self.step
            self.step += 1
            a = cmd.action_index
            if self.screen == "rest":
                self.screen = "grid" if self.smith and a == 0 else "end"
            elif self.screen == "grid":
                self.screen = "rest" if a == 2 else "confirm"
            else:
                self.screen, self.smith = ("rest", True) if a == 1 else ("rest", False)
            if self.step == 14:
                self.screen = "end"
        b = FrameBuilder(MsgType.TERMINAL if self.screen == "end" else MsgType.OBS)
        b.episode_id, b.step_id = self.episode_id, self.step
        g = b.glob
        g[G.N_SEATS], g[G.TOTAL_FLOOR] = 1, 2
        g[G.ROOM_TYPE] = self.vocab.id("enum:RoomType", "RestSite")
        b.add(Table.SEATS, seat=0, hp=50, max_hp=80, alive=1, creature_row=-1,
              character_id=self.vocab.id("character", "IRONCLAD"))
        if self.screen == "rest":
            g[G.KIND] = Kind.REST
            if self.smith:
                b.add(Table.OPTIONS, kind=OptionKind.REST_OPTION, seat=0, pos=0,
                      key_id=self.vocab.id("rest_option", "SMITH"))
                b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=0)
            b.add(Table.ACTIONS, verb=Verb.PROCEED, seat=0)
        elif self.screen in ("grid", "confirm"):
            g[G.KIND], g[G.KIND_SUB], g[G.SEL_MIN], g[G.SEL_MAX] = Kind.CARD_SELECT, CardSelectSub.DECK_UPGRADE, 1, 1
            if self.screen == "grid":
                for pos in range(2):
                    row = b.add(Table.CARDS, zone=Zone.CHOICE, seat=0, pos=pos, count=1, card_id=7 + pos)
                    b.add(Table.ACTIONS, verb=Verb.SELECT_CARD, seat=0, src_table=Table.CARDS, src_row=row)
            else:
                g[G.SEL_COUNT] = 1
                b.add(Table.ACTIONS, verb=Verb.CONFIRM_SELECTION, seat=0)
            b.add(Table.ACTIONS, verb=Verb.CANCEL, seat=0)
        else:
            g[G.KIND] = Kind.TERMINAL
            b.flags, b.result = FrameFlag.TERMINAL, Result.LOSS
            g[G.TERMINAL_REASON] = TerminalReason.DECISION_CAP if self.step == 14 else TerminalReason.DEATH
            g[G.POLICY_DECISIONS] = self.step
        return [b.build(hashes=False)]


def _smith_host(vocab: Vocab) -> FakeHost:
    host = FakeHost(vocab)
    host.game = _SmithGame(vocab)
    return host


mode = os.environ["RULES_MODE"]
assert mode in ("base", "mask", "teach0"), mode
torch.manual_seed(0)
torch.set_num_threads(1)
vocab = Vocab(make_synthetic_vocab())
flags: dict = {"cycle_ban": True, "public_history": True, "commit_select": True, "removal_rule": True,
               "removal_ratings": os.environ["RULES_TABLE"]}
ppo: dict = {}
if mode == "teach0":
    from spire.rules import RulesConfig
    flags["rules_mode"] = "teach"
    ppo["rules"] = RulesConfig(coef_removal=0.0)
tok = Tokenizer(vocab, TokenizerConfig(**flags))
net = SpireNet(tok.layout, ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=16, d_cond=16, d_static=32))


def state_hash(sd: dict) -> str:
    h = hashlib.sha256()
    for k, v in sd.items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


init = state_hash(net.state_dict())
keys = sorted(net.state_dict())
expl = ((int(Kind.CARD_REWARD), 0.1), (int(Kind.CARD_SELECT), 0.1), (int(Kind.COMBAT), 0.05))
cfg = PPOConfig(minibatch=512, micro_batch=256, epochs=1, exploration_by_kind=expl, **ppo)
learner = Learner(net, tok, cfg, device="cpu", fast=False)
policy = TorchPolicy(learner.model, tok, seed=11, exploration_by_kind=expl)
segs: list = []
store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32))
hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(mean_decisions=60, hashes=False, abort_prob=0.0))
         for i in range(4)] + [_host(vocab), _smith_host(vocab)]
commit = []


def on_decision(env: int, f, a: int) -> None:
    g = f.gvals
    if g[G.KIND] == Kind.CARD_SELECT and g[G.KIND_SUB] == CardSelectSub.DECK_UPGRADE and not g[G.IN_COMBAT]:
        commit.append(a)



actor = Actor(hosts, policy, tok, EpisodeSampler(EpisodeDist(lineups=(("SILENT", 1.0), ("IRONCLAD", 1.0))),
                                                 vocab, 0), ActorConfig(), store=store, on_decision=on_decision)
actor.run(max_decisions=3600)
h, hs, hb = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
slots = {"removal": 0, "cancel_or_cycle": 0, "commit_frames": len(commit)}
for s in segs:
    for a in (s.actions, s.logp, s.values, s.kinds, s.n_legal):
        h.update(np.ascontiguousarray(a).tobytes())
    for i in range(len(s)):
        x = s.samples[i]
        for a in (x.types, x.raw, x.rel, x.edges, x.cond, x.history[:-CYCLE_BAN_SLOTS]):
            hs.update(np.ascontiguousarray(a).tobytes())
        b = x.history[-CYCLE_BAN_SLOTS:]
        hb.update(np.ascontiguousarray(b).tobytes())
        slots["removal"] += int(b[0] < 0)
        slots["cancel_or_cycle"] += int((b > 0).any() and b[0] >= 0)
m = learner.update(segs)
terms = {k: m[k] for k in ("loss_pi", "loss_v", "loss_win", "loss_floors", "loss_hp", "loss_ent_bonus", "approx_kl",
                           "clipfrac", "entropy", "grad_norm")}
print(json.dumps({"mode": mode, "decisions": int(sum(len(s) for s in segs)), "init": init, "keys": keys,
                  "samples": hs.hexdigest(), "slots_hash": hb.hexdigest(), "slots": slots, "steps": h.hexdigest(),
                  "params": state_hash(learner.model.state_dict()), "terms": terms, "rules": m.get("rules")}))

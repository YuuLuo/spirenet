"""Synthetic flag-on episodes for the critic gates (notes/critic/contract.md 6, PG1 / FP1 / FP2 on synthetic frames):
FakeHost episodes with RESET flag HIDDEN, the policy's actions recorded, and the tokenizer configurations the gates
run in (calc host = forecast + calc; inputs v2a; the rules modes)."""
from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spire.critic import training_reset_flags
from spire.cycle import CycleBan
from spire.proto import parse_frame
from spire.removal import load_rule
from spire.schema import G, reset_flags_for
from spire.synth import FakeHost, FakeHostConfig, strip_hidden
from spire.tokenize import Tokenizer, TokenizerConfig
from spire.vocab import Vocab, make_synthetic_vocab

from test_removal import synthetic_table

N_FRAMES = 3000
N_FIGHT = 1500
CONFIGS = ("calc", "v2a")  # the calc host's frames (793 | 1024) and inputs v2a (795 | 1024)


@dataclass
class Stream:
    """Episodes of flag-on payloads and the action taken after each OBS frame (-1 after a TERMINAL)."""

    episodes: list[list[bytes]]
    actions: list[list[int]]

    @property
    def frames(self) -> list[bytes]:
        return [p for e in self.episodes for p in e]


@functools.lru_cache(maxsize=None)
def vocab() -> Vocab:
    return Vocab(make_synthetic_vocab())


@functools.lru_cache(maxsize=None)
def table_path() -> str:
    p = Path(os.environ.get("PYTEST_TMP", "/tmp")) / f"critic-removal-{os.getpid()}.json"
    if not p.exists():
        p.write_text(json.dumps(synthetic_table(vocab())), encoding="utf-8")
    return str(p)


def flags_of(config: str) -> int:
    return training_reset_flags(reset_flags_for(True, calc=True, map_in_combat=config == "v2a"))


def tok_config(config: str, rules: str | None = None) -> TokenizerConfig:
    """The calc host's tokenizer (forecast + calc) or inputs v2a; ``rules``: cycle ban + removal rule in that mode."""
    kw: dict = dict(forecast=True, calc=True)
    if config == "v2a":
        kw.update(route_summary=True, map_tokens="no_combat", second_boss_table=True)
    if rules is not None:
        kw.update(cycle_ban=True, commit_select=True, removal_rule=True, removal_ratings=table_path(), rules_mode=rules,
                  public_history=True)
    return TokenizerConfig(**kw)


def tokenizers(config: str, rules: str | None = None, native: bool = True) -> tuple[Tokenizer, Tokenizer]:
    """(tokenizer of the training actor: allow_hidden, tokenizer of every other tool)."""
    on, off = Tokenizer(vocab(), tok_config(config, rules), native=native), Tokenizer(vocab(), tok_config(config, rules),
                                                                                      native=native)
    on.allow_hidden = True
    return on, off


def rule():
    return load_rule(vocab(), table_path())


@functools.lru_cache(maxsize=None)
def stream(config: str, seed: int = 7) -> Stream:
    """>= N_FRAMES frames, >= N_FIGHT of them in fights, from seeded FakeHost episodes and uniform random actions."""
    v = vocab()
    host = FakeHost(v, seed, FakeHostConfig(mean_decisions=300, length_sigma=0.3, abort_prob=0.0),
                    verify_hashes=True, training_hidden=True)
    chars = [v.id("character", v["character"].entries[2])]
    rng = np.random.default_rng(seed)
    tok = Tokenizer(v, tok_config(config, "teach"))
    eps: list[list[bytes]] = []
    acts: list[list[int]] = []
    n = fight = 0
    ep = 0
    while n < N_FRAMES or fight < N_FIGHT:
        f = host.reset(1000 * seed + ep, chars, int(rng.integers(0, 11)), flags=flags_of(config))
        ep += 1
        cyc = CycleBan(v, commit_select=True, removal=rule(), mode="teach")
        frames: list[bytes] = []
        a: list[int] = []
        while True:
            frames.append(bytes(f.payload))
            fight += int(f.gvals[G.IN_COMBAT] == 1)
            if f.is_terminal:
                a.append(-1)
                break
            # uniform over the rows the actor may take: the cycle-ban / removal-rule tracker of rules mode "teach"
            s = tok.encode(parse_frame(strip_hidden(bytes(f.payload))))
            cyc.observe(parse_frame(strip_hidden(bytes(f.payload))), s)
            legal = [i for i in range(f.n_actions) if i not in cyc.active_bans]
            k = int(legal[int(rng.integers(len(legal)))])
            cyc.record(parse_frame(strip_hidden(bytes(f.payload))), k)
            a.append(k)
            f = host.step(k)
        n += len(frames)
        eps.append(frames)
        acts.append(a)
    return Stream(eps, acts)


def in_fight(payload: bytes) -> bool:
    return parse_frame(payload).gvals[G.IN_COMBAT] == 1

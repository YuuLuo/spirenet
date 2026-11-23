"""An episode lost to a host timeout leaves a reproduction record (on_error, kind "lost"): the exact RESET and every
action index sent, the unanswered one last. Replaying it on a fresh deterministic host reproduces every frame the
lost episode saw, byte for byte (EPISODE_ID aside), up to the request that never got an answer."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy, seed_to_wire
from spire.host import HostConfig, HostProcess, HostTimeout
from spire.proto import ResetCmd, StepCmd
from spire.rewards import RoomIds
from spire.rollout import RolloutConfig, RolloutStore
from spire.schema import H
from spire.synth import FakeGame, FakeHostConfig
from spire.tokenize import Tokenizer
from spire.vocab import Vocab

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "fake_spirehost.py")
FAKE = ("--serve", "--no-hashes", "--mean-decisions", "30", "--seed", "1")


def _fake(tmp_path: Path, *extra: str, step_timeout: float = 1.0) -> HostProcess:
    return HostProcess(HostConfig(command=(sys.executable, "-B", SCRIPT, *FAKE, *extra), start_timeout=30.0,
                                  reset_timeout=5.0, step_timeout=step_timeout, name="t",
                                  log_path=str(tmp_path / "h.log"), max_restarts=20))


def _masked(payload: bytes) -> bytes:
    w = np.frombuffer(payload, dtype="<i4").copy()
    w[H.EPISODE_ID] = 0
    return w.tobytes()


def test_host_keeps_the_unanswered_step(tmp_path: Path) -> None:
    host = _fake(tmp_path, "--hang-after", "5")
    try:
        f = host.reset(seed_to_wire("ABCDEFGHJKLM"), [0])
        sent = []
        with pytest.raises(HostTimeout):
            while True:
                assert not f.is_terminal, "the fake episode must outlast the hang"
                a = f.n_actions - 1
                sent.append(a)
                f = host.step(a)
        ep = host.last_episode
        assert ep is not None and ep.actions == sent[:-1] and ep.unanswered == sent[-1]
        assert host.stats.timeouts == 1 and host.stats.lost == 1
    finally:
        host.close()


def test_lost_record_replays_the_lost_episode(vocab: Vocab, tmp_path: Path) -> None:
    host = _fake(tmp_path, "--hang-after", "40")
    seen: dict[int, list[tuple[bytes, int]]] = defaultdict(list)  # env -> (frame bytes, action) of its episode

    def on_decision(env_id: int, f, action: int) -> None:
        if f.step_id == 0:
            seen[env_id].clear()
        seen[env_id].append((bytes(f.payload), action))

    errs: list[dict] = []
    at_loss: list[list[tuple[bytes, int]]] = []  # the env's frames when its record arrived

    def on_error(rec: dict) -> None:
        errs.append(rec)
        at_loss.append(list(seen[rec["env"]]))

    actor = Actor([host], UniformPolicy(0), Tokenizer(vocab), EpisodeSampler(EpisodeDist(), vocab, 0),
                  ActorConfig(reset_flags=25, max_episode_decisions=500), on_error=on_error,
                  on_decision=on_decision,
                  store=RolloutStore(lambda seg: None, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=32)))
    try:
        actor.run(max_seconds=6.0)
    finally:
        host.close()
    lost = [e for e in errs if e.get("kind") == "lost"]
    assert len(lost) == len(errs) == actor.counters.lost_recorded >= 1
    rec = lost[0]
    assert "HostTimeout" in rec["detail"] and rec["unanswered"] >= 0
    assert rec["decisions"] == len(rec["actions"]) + 1
    assert rec["reset_flags"] == 25 and rec["max_decisions"] == 500 * len(rec["characters"])
    original = at_loss[0]
    assert [a for _, a in original] == rec["actions"] + [rec["unanswered"]]
    # replay on a fresh in-process game with the fake host's configuration
    game = FakeGame(None, seed=1, config=FakeHostConfig(mean_decisions=30, hashes=False))
    cmd = ResetCmd(tuple(rec["characters"]), rec["ascension"], seed_to_wire(rec["seed"]), flags=rec["reset_flags"],
                   max_decisions=rec["max_decisions"])
    (payload,) = game.handle(cmd)
    for step, (frame, action) in enumerate(original):
        assert _masked(payload) == _masked(frame), f"replay diverged at decision {step}"
        (payload,) = game.handle(StepCmd(step, action))

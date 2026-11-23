"""errata-v1 (notes/schema-errata-v1.md) on the Python side: PY-1 .. PY-10 and the two corrections.

GLOBAL 44..51 and TerminalReason (PY-1/2), terminal handling that tells deaths from truncations (PY-4),
kill bonuses from the host counters (PY-5, tests/test_rewards.py), RESET word 14 (PY-6), the real-host
launch spec as configuration (PY-7), ``p:pos`` width (PY-8), the fake host filling the new slots (PY-9), the
TERMINAL STEP_ID assertion (PY-10); event fights are Monster-room fights (E-22) and a death TERMINAL
carries no combat tables (P-5).
"""
from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spire.actor import Actor, ActorConfig, EpisodeDist, EpisodeSampler, UniformPolicy, terminal_result_name
from spire.host import REAL_HOST_CONFIG, HostConfig, real_host_spec
from spire.metrics import EpisodeRecord, OutcomeTracker
from spire.proto import ProtocolError, ResetCmd, StepCmd, decode_command, encode_reset, parse_frame
from spire.rewards import Outcome, RewardConfig, RoomIds, frame_facts
from spire.rollout import RolloutConfig, RolloutStore, Segment, SegmentEnd, gae, GAEConfig
from spire.schema import (
    G_TERMINAL_ONLY,
    GLOBAL_OFFSET,
    TERMINAL_RESULT,
    TRUNCATION_REASONS,
    WORD,
    G,
    Kind,
    Result,
    SeatCol,
    Table,
    TerminalReason,
)
from spire.synth import FAKE_ABORT_REASON, FakeGame, FakeHost, FakeHostConfig
from spire.tokenize import PROTOCOL_ROWS, SPECS, Ent, Tokenizer
from spire.vocab import Vocab

FAST: dict[str, Any] = dict(mean_decisions=60, hashes=False, abort_prob=0.0)


def play(host: FakeHost, n: int, seed0: int = 0, *, cap: int = 0) -> list[list]:
    """``n`` uniform-random episodes; per episode the list of frames (TERMINAL last)."""
    rng = np.random.default_rng(seed0)
    out = []
    for e in range(n):
        f = host.reset_with(ResetCmd((4,), 0, seed0 * 1000 + e, max_decisions=cap))
        frames = [f]
        while not f.is_terminal:
            f = host.step(int(rng.integers(f.n_actions)))
            frames.append(f)
        out.append(frames)
    return out


# --------------------------------------------------------------------------- PY-1 / PY-2 / PY-6


def test_terminal_reason_and_result_table() -> None:
    assert TERMINAL_RESULT[TerminalReason.WIN] == Result.WIN and TERMINAL_RESULT[TerminalReason.ABORTED] == Result.ABORTED
    assert all(TERMINAL_RESULT[r] == Result.LOSS for r in (TerminalReason.DEATH, *TRUNCATION_REASONS))
    assert TerminalReason.DEATH not in TRUNCATION_REASONS and len(TRUNCATION_REASONS) == 3
    assert [terminal_result_name(Result.LOSS, int(r)) for r in TerminalReason] == \
        ["loss", "loss", "loss", "decision_cap", "turn_cap", "truncated", "loss"]
    assert terminal_result_name(Result.WIN, 1) == "win" and terminal_result_name(Result.ABORTED, 6) == "aborted"


def test_reset_word14_round_trip() -> None:
    cmd = ResetCmd((4, 2), 10, 12345, flags=25, max_decisions=777)
    msg = encode_reset(cmd)
    assert struct.unpack_from("<i", msg, 4 + 14 * 4)[0] == 777 and struct.unpack_from("<i", msg, 4 + 15 * 4)[0] == 0
    assert decode_command(msg[4:]) == cmd
    default = decode_command(encode_reset(ResetCmd((4,)))[4:])
    assert isinstance(default, ResetCmd) and default.max_decisions == 0  # 0 = host default
    with pytest.raises(ValueError):
        ResetCmd((4,), max_decisions=-1)


# --------------------------------------------------------------------------- PY-9: the fake host speaks the slots


def test_fake_frames_carry_win_history_and_terminal_slots(vocab: Vocab) -> None:
    host = FakeHost(vocab, seed=3, config=FakeHostConfig(**FAST, win_prob=1.0))
    room = {vocab.id("enum:RoomType", n): n for n in ("Monster", "Elite", "Boss")}
    combat_rooms: Counter[str] = Counter()
    for frames in play(host, 6, 1):
        prev = None
        for f in frames:
            g = f.gvals
            cw, ew, bw, lr, lf = (g[G.COMBATS_WON], g[G.ELITES_WON], g[G.BOSSES_WON], g[G.LAST_WON_ROOM_TYPE],
                                  g[G.LAST_WON_TOTAL_FLOOR])
            assert ew + bw <= cw
            if cw == 0:
                assert (lr, lf) == (0, -1)  # no win yet (real host: G[48] = -1)
            else:
                assert room.get(lr) in ("Monster", "Elite", "Boss") and 0 <= lf <= g[G.TOTAL_FLOOR]
            if prev is not None:
                assert all(a >= b for a, b in zip((cw, ew, bw), prev)), "win counters are monotone"
            prev = (cw, ew, bw)
            if f.is_obs:
                assert all(g[i] == 0 for i in G_TERMINAL_ONLY)
                if g[G.IN_COMBAT]:
                    combat_rooms[room.get(g[G.ROOM_TYPE], "?")] += 1
        t = frames[-1]
        g = t.gvals
        assert t.result == Result.WIN and g[G.TERMINAL_REASON] == TerminalReason.WIN
        assert g[G.POLICY_DECISIONS] == t.step_id == frames[-2].step_id + 1 == len(frames) - 1
        assert t.rows[Table.CREATURES] == 0 and t.rows[Table.ACTIONS] == 0
    assert set(combat_rooms) <= {"Monster", "Elite", "Boss"} and combat_rooms["Monster"] > 0


@pytest.mark.parametrize("pool", [0, 4])
def test_fake_host_decision_cap_and_reasons(vocab: Vocab, pool: int) -> None:
    """RESET word 14 caps the episode: TERMINAL(LOSS, DECISION_CAP) at STEP_ID = cap; turn cap / abort / death."""
    host = FakeHost(vocab, seed=5, config=FakeHostConfig(**{**FAST, "mean_decisions": 200}, frame_pool=pool))
    eps = play(host, 4, 2, cap=7)
    for frames in eps:
        t = frames[-1]
        assert len(frames) == 8 and t.step_id == 7 and t.result == Result.LOSS
        assert t.gvals[G.TERMINAL_REASON] == TerminalReason.DECISION_CAP and t.gvals[G.POLICY_DECISIONS] == 7
        assert frame_facts(t).truncated and t.table(Table.SEATS)[0, SeatCol.HP] > 0  # alive: not a death
    assert host.stats.truncated == 4 and host.stats.losses == 0 and host.stats.finished == 4
    assert host.last_episode is not None and host.last_episode.terminal_reason == TerminalReason.DECISION_CAP
    host = FakeHost(vocab, seed=6, config=FakeHostConfig(**FAST, turn_cap_prob=1.0, frame_pool=pool))
    assert all(f[-1].gvals[G.TERMINAL_REASON] == TerminalReason.COMBAT_TURN_CAP for f in play(host, 3, 3))
    host = FakeHost(vocab, seed=7, config=FakeHostConfig(**{**FAST, "abort_prob": 1.0}, frame_pool=pool))
    for frames in play(host, 3, 4):
        g = frames[-1].gvals
        assert frames[-1].result == Result.ABORTED and g[G.TERMINAL_REASON] == TerminalReason.ABORTED
        assert g[G.ABORT_REASON] == FAKE_ABORT_REASON
    host = FakeHost(vocab, seed=8, config=FakeHostConfig(**FAST, win_prob=0.0, frame_pool=pool))
    for frames in play(host, 3, 5):
        t = frames[-1]
        assert t.result == Result.LOSS and t.gvals[G.TERMINAL_REASON] == TerminalReason.DEATH
        assert t.gvals[G.PLAYERS_ALIVE] == 0 and int(t.table(Table.SEATS)[0, SeatCol.HP]) == 0  # death: hp 0
        assert t.rows[Table.CREATURES] == 0 and t.gvals[G.IN_COMBAT] == 0  # P-5: no combat tables on death
    assert host.stats.losses == 3 and host.stats.truncated == 0


def test_event_fights_are_monster_room_fights(vocab: Vocab) -> None:
    """E-22: the fake's event-started fights show ROOM_TYPE Monster and earn no elite / boss counter."""
    game = FakeGame(vocab, seed=11, config=FakeHostConfig(**{**FAST, "mean_decisions": 400}, win_prob=1.0))
    ev, mon = vocab.id("enum:RoomType", "Event"), vocab.id("enum:RoomType", "Monster")
    seen = steps = 0
    rng = np.random.default_rng(0)
    for e in range(200):
        if seen >= 3 or steps > 30_000:
            break
        f = parse_frame(game.handle(ResetCmd((4,), 0, 500 + e))[0])
        prev_kind = 0
        while not f.is_terminal:
            steps += 1
            g = f.gvals
            if f.is_obs and g[G.KIND] == Kind.COMBAT and prev_kind == Kind.EVENT:  # a fight straight out of an event
                assert g[G.ROOM_TYPE] == mon and g[G.ROOM_TYPE] != ev
                seen += 1
            prev_kind = g[G.KIND]
            f = parse_frame(game.handle(StepCmd(f.step_id, int(rng.integers(f.n_actions))))[-1])
    assert seen > 0


# --------------------------------------------------------------------------- PY-10


def _patched(payload: bytes, *, step: int | None = None, g: dict[int, int] | None = None,
             result: int | None = None) -> bytes:
    buf = bytearray(payload)
    if step is not None:
        struct.pack_into("<i", buf, 6 * WORD, step)
    if result is not None:
        struct.pack_into("<i", buf, 11 * WORD, result)
    for i, v in (g or {}).items():
        struct.pack_into("<i", buf, GLOBAL_OFFSET + i * WORD, v)
    return bytes(buf)


@pytest.mark.parametrize("bad", ["step", "decisions", "result"])
def test_terminal_step_id_assertion(vocab: Vocab, bad: str) -> None:
    host = FakeHost(vocab, seed=1, config=FakeHostConfig(**FAST, win_prob=1.0))
    orig = host.game._terminal

    def broken(ep):  # type: ignore[no-untyped-def]
        p = orig(ep)
        if bad == "step":
            return _patched(p, step=ep.ctx.step_id + 1, g={G.POLICY_DECISIONS: ep.ctx.step_id + 1})
        if bad == "decisions":
            return _patched(p, g={G.POLICY_DECISIONS: ep.ctx.step_id - 1})
        return _patched(p, g={G.TERMINAL_REASON: int(TerminalReason.DEATH)})  # WIN result with a DEATH reason
    host.game._terminal = broken  # type: ignore[method-assign]
    f = host.reset(None, (4,), 0)
    with pytest.raises(ProtocolError):
        while not f.is_terminal:
            f = host.step(0)
    assert host.stats.lost == 1 and host.stats.finished == 0


# --------------------------------------------------------------------------- PY-4: rollout + actor + statistics


def test_legacy_cap_truncation_bootstraps_and_masks_outcome(vocab: Vocab) -> None:
    host = FakeHost(vocab, seed=2, config=FakeHostConfig(**{**FAST, "mean_decisions": 200}))
    segs: list[Segment] = []
    store = RolloutStore(segs.append, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=4),
                         rewards=RewardConfig(caps_as_failure=False))
    rng = np.random.default_rng(0)
    f = host.reset_with(ResetCmd((4,), 0, 9, max_decisions=10))
    store.begin_episode(0, (4,), 0)
    values = []
    while not f.is_terminal:
        v = float(rng.random())
        values.append(v)
        store.add_step(0, frame_facts(f), object(), 0, -0.5, v, 0)
        f = host.step(0)
    assert store.end_episode(0, frame_facts(f)) == Outcome.TRUNCATED
    assert store.stats.host_truncated == 1 and store.stats.losses == 0 and store.stats.steps_released == 10
    last = segs[-1]
    assert last.end == SegmentEnd.TRUNCATED and last.bootstrap_value == pytest.approx(values[-1])
    assert not any(s.win_mask.any() for s in segs)  # censored outcome: no win / floors targets
    assert all(s.rewards.size == len(s) for s in segs)
    adv, ret = gae(segs, GAEConfig(gamma=0.99, lam=1.0))
    want_last = last.rewards[-1] + 0.99 * values[-1]  # one-step return bootstraps V(s_{T-1}) for V(s_T)
    assert ret[-1] == pytest.approx(want_last, abs=1e-5)


def test_actor_records_truncations_as_counted_failures(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    hosts = [FakeHost(vocab, seed=i, config=FakeHostConfig(**{**FAST, "mean_decisions": 400})) for i in range(3)]
    recs: list[EpisodeRecord] = []
    store = RolloutStore(lambda s: None, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16))
    actor = Actor(hosts, UniformPolicy(0), tok, EpisodeSampler(EpisodeDist(lineups=(("IRONCLAD", 1.0),)), vocab, 0),
                  ActorConfig(max_episode_decisions=25), store=store, on_episode=recs.append)
    actor.run(max_decisions=600)
    done = [r for r in recs if r.result != "shutdown"]
    assert done and all(r.result == "decision_cap" and r.decisions == 25 for r in done)
    assert all(r.terminal_reason == TerminalReason.DECISION_CAP and r.combats_won >= 0 for r in done)
    assert store.stats.host_truncated == len(done)
    tr = OutcomeTracker()
    for r in done:
        tr.add(r)
    tr.add(EpisodeRecord(lineup="IRONCLAD", ascension=0, result="win"))
    tr.add(EpisodeRecord(lineup="IRONCLAD", ascension=0, result="loss", floor=3))
    tr.add(EpisodeRecord(lineup="IRONCLAD", ascension=0, result="aborted"))
    s = tr.summary()["IRONCLAD|A0"]
    assert s["n"] == len(done) + 2 and s["wins"] == 1  # truncations are failures inside the denominator
    assert s["deaths"] == 1 and s["truncated"] == len(done) and s["death_floor"] == {3: 1}
    assert s["failures"] == {"decision_cap": len(done), "aborted": 1}


# --------------------------------------------------------------------------- PY-8 + feature hygiene


def test_tokenizer_layout_errata(vocab: Vocab) -> None:
    assert PROTOCOL_ROWS["p:pos"] == 33
    for e in (Ent.GLOBAL, Ent.DECISION):  # the only entities reading GLOBAL slots
        spec = SPECS[e]
        src = {int(x) for x in [*(i.src for i in spec.ids), *(n.src for n in spec.nums)] if not isinstance(x, str)}
        assert not src & {int(x) for x in G_TERMINAL_ONLY}, "TERMINAL-only protocol slots are never features"
        if e == Ent.GLOBAL:  # the public win history is (errata-v1 section 5)
            assert {int(x) for x in (G.COMBATS_WON, G.ELITES_WON, G.BOSSES_WON, G.LAST_WON_ROOM_TYPE,
                                     G.LAST_WON_TOTAL_FLOOR)} <= src
    tok = Tokenizer(vocab)
    assert tok.layout.regions["p:pos"][1] == 33


# --------------------------------------------------------------------------- PY-7: launch spec as configuration


def test_real_host_spec_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # the HostConfig() default below must come from the default file, not from a launch spec chosen for the
    # real-host tests via $SPIRE_REAL_HOST_CONFIG (py-adopt-algo: the full suite runs with it set)
    monkeypatch.delenv("SPIRE_REAL_HOST_CONFIG", raising=False)
    spec = real_host_spec(REAL_HOST_CONFIG)
    assert spec.command[-1] == "--serve" and spec.vocab.name == "vocab.json" and spec.catalog.name == "catalog.json"
    assert "{repo}" not in " ".join(spec.command)
    p = tmp_path / "h.toml"
    p.write_text('command = ["x.exe", "--serve", "--hp", "{repo}"]\nvocab = "v/vocab.json"\ncatalog = "c.json"\n'
                 '[env]\nSPIREHOST_GAME_DIR = "{repo}/host/game"\n', encoding="utf-8")
    s = real_host_spec(p)
    assert s.vocab == (tmp_path / "v" / "vocab.json").resolve() and s.command[:2] == ("x.exe", "--serve")
    assert s.env is not None and s.env["SPIREHOST_GAME_DIR"].endswith("host/game") and "{repo}" not in s.command[3]
    assert s.process_env() is not None and "PATH" in {k.upper() for k in s.process_env() or {}}
    assert s.command_with("--pipe", "n")[-2:] == ("--pipe", "n")
    assert sorted(Path(m).name for m in s.missing()) == ["c.json", "vocab.json"]
    p.write_text('command = []\nvocab = "v"\ncatalog = "c"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        real_host_spec(p)
    assert HostConfig().command == spec.command  # the default host command comes from the file


def test_static_prior_clips_catalog_sentinels(vocab: Vocab) -> None:
    """The real catalog has 999999999 sentinels (unsold relics' merchant_cost, endless upgrades): the static
    prior's symlog inputs are clipped to +-8 like every frame feature."""
    from spire.model import _static_matrix
    from spire.vocab import Catalog, make_synthetic_catalog
    raw = make_synthetic_catalog(vocab, seed=0)
    first = next(iter(raw["categories"]["relic"]))
    raw["categories"]["relic"][first]["merchant_cost"] = 999_999_999
    x, valid = _static_matrix(Catalog(raw, vocab), "relic", vocab.embedding_rows("relic"), 1)
    assert float(np.abs(x).max()) == pytest.approx(8.0) and valid.sum() == len(raw["categories"]["relic"])


def test_eval_summary_counts_truncations_as_failures() -> None:
    """PY-4 in the evaluation statistics: a decision-cap / turn-cap episode is a non-win in the denominator."""
    from spire.evalpanel import summarize
    recs = [{"lineup": "IRONCLAD", "ascension": 0, "result": r, "floor": 5}
            for r in ("win", "win", "loss", "decision_cap", "turn_cap", "aborted")]
    s = summarize(recs)["IRONCLAD|A0"]
    assert s["n"] == 6 and s["wins"] == 2 and s["win_rate"] == pytest.approx(2 / 6)
    assert s["failures"] == {"decision_cap": 1, "turn_cap": 1, "aborted": 1}

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator, cast

import numpy as np
import pytest

from spire import synth
from spire.actor import TorchPolicy, UniformPolicy
from spire.evalpanel import evaluate, make_panel
from spire.h5bench import MIXED_BUCKETS, shape_report, workload
from spire.h5eval import MeasuredPolicy
from spire.proto import frame_problems, parse_frame
from spire.schema import ActionCol, G, Kind, MapNodeCol, OptionCol, Table, TreasurePhase, Verb, ViewMode
from spire.synth import FakeHost, FakeHostConfig, FrameGen, RunCtx
from spire.tokenize import BatchShape, Tokenizer, TokenizerConfig
from spire.trainstep import MicroStepper
from spire.vocab import Vocab


def _first_h5_room(gen: FrameGen, room: str, rng: np.random.Generator | None = None
                   ) -> tuple[RunCtx, Generator[tuple[int, int], int, None], tuple[int, int]]:
    ctx = gen.random_ctx(np.random.default_rng(91), 4, view=ViewMode.TEAM)
    assert ctx.map is not None
    ctx.path = []
    ctx.map.nodes[ctx.map.start, MapNodeCol.POINT_TYPE] = gen.point[room]
    schedule = synth._schedule_h5(np.random.default_rng(0) if rng is None else rng, ctx, gen, "typical")
    assert next(schedule) == (Kind.MAP, 0)
    ctx.path.append(ctx.map.start)
    ctx.total_floor = 1
    return ctx, schedule, schedule.send(Verb.TRAVEL)


@pytest.mark.parametrize(("room", "kind"), [("Monster", Kind.REWARD), ("Shop", Kind.SHOP),
                                           ("RestSite", Kind.REST)])
def test_h5_room_retains_owner_until_proceed(gen: FrameGen, monkeypatch: pytest.MonkeyPatch,
                                           room: str, kind: Kind) -> None:
    def won_fight(rng: np.random.Generator, ctx: RunCtx, gen: FrameGen,
                  room: str) -> Generator[tuple[int, int], int, None]:
        yield from ()

    monkeypatch.setattr(synth, "_fight_h5", won_fight)
    ctx, schedule, decision = _first_h5_room(gen, room)
    assert decision == (kind, 0)
    for verb in [Verb.USE_POTION, Verb.DISCARD_POTION] * 7:
        assert schedule.send(verb) == (kind, 0)
        assert ctx.acting_seat == 0 and not ctx.room_done
    if kind in (Kind.REWARD, Kind.REST):
        child = schedule.send(Verb.CHOOSE_OPTION)
        assert child[0] == (Kind.CARD_REWARD if kind == Kind.REWARD else Kind.CARD_SELECT)
        assert ctx.acting_seat == 0 and not ctx.room_done
        answer = Verb.PICK_OFFER_CARD if kind == Kind.REWARD else Verb.CONFIRM_SELECTION
        assert schedule.send(answer) == (kind, 0)
    assert schedule.send(Verb.PROCEED) == (kind, 0)
    assert ctx.acting_seat == 1 and ctx.room_done == {0}
    schedule.close()


@pytest.mark.parametrize("shared", [False, True])
def test_h5_event_potions_do_not_answer_page_or_team_vote(gen: FrameGen, shared: bool) -> None:
    # Events resolve from Unknown map nodes. Control that resolution and the shared
    # branch explicitly; no invented Event MapPointType or dependence on RNG seeds.
    rng = cast(np.random.Generator, SimpleNamespace(choice=lambda choices: "Event",
                                                   random=lambda: 0.2 if shared else 0.5,
                                                   integers=lambda lo, hi: 1))
    ctx, schedule, decision = _first_h5_room(gen, "Unknown", rng)
    assert decision == (Kind.EVENT, 0) and ctx.shared_vote == shared
    for verb in [Verb.USE_POTION, Verb.DISCARD_POTION] * 7:
        assert schedule.send(verb) == (Kind.EVENT, 0)
        assert ctx.acting_seat == 0 and ctx.shared_vote == shared
        assert ctx.room_done == (set(range(4)) if shared else set())
    if shared:
        assert schedule.send(Verb.CHOOSE_OPTION) == (Kind.MAP, 0)
        assert not ctx.shared_vote
    else:
        # Private events have one or two scripted pages; only option choices advance them.
        for _ in range(2):
            decision = schedule.send(Verb.CHOOSE_OPTION)
            if decision[0] == Kind.CARD_SELECT:
                decision = schedule.send(Verb.CONFIRM_SELECTION)
            if ctx.acting_seat == 1:
                break
        assert ctx.acting_seat == 1 and ctx.room_done == {0}
    schedule.close()


@pytest.mark.parametrize("frame_pool", [0, 2])
def test_fake_h5_union_turn_and_seat_major_rooms(vocab: Vocab, frame_pool: int) -> None:
    host = FakeHost(vocab, config=FakeHostConfig(mean_decisions=200, length_sigma=0, abort_prob=0,
                                               hashes=False, frame_pool=frame_pool))
    frame = host.reset(17, (2, 2, 4, 4), 10)
    ended: set[int] = set()
    combat_key: tuple[int, int] | None = None
    previous_room_seat: dict[int, int] = {}
    seen_combat = seen_reward = seen_prompt = False
    for _ in range(100):
        assert not frame.is_terminal
        assert not frame_problems(frame), frame_problems(frame)
        seats = set(int(s) for s in frame.actions[:, ActionCol.SEAT])
        assert host.game.ep is not None
        ctx = host.game.ep.ctx
        # Pooled benchmarks intentionally freeze floor/round/HP; scheduling and the
        # seat-status fields must still follow this episode's actual H5 arbitration.
        floor = ctx.total_floor
        if frame_pool == 0:
            assert int(frame.glob[G.TOTAL_FLOOR]) == floor
        if frame.kind == Kind.COMBAT:
            seen_combat = True
            key = (floor, ctx.combat_round)
            if key != combat_key:
                ended.clear()
                combat_key = key
            assert seats == set(range(4)) - ended
            assert not np.any(frame.actions[:, ActionCol.VERB] == Verb.UNDO_END_TURN)
            action = int(np.flatnonzero(frame.actions[:, ActionCol.VERB] == Verb.END_TURN)[0])
            ended.add(int(frame.actions[action, ActionCol.SEAT]))
        else:
            seat = int(frame.glob[G.ACTING_SEAT])
            assert seats == {seat}
            if frame.kind == Kind.MAP:
                assert seat == 0
                if floor:
                    assert all(int(frame.glob[56 + s]) & 1 for s in range(4))
            elif frame.kind == Kind.CARD_SELECT and frame.glob[G.IN_COMBAT]:
                seen_prompt = True
            else:
                assert seat >= previous_room_seat.get(floor, 0)
                previous_room_seat[floor] = seat
                seen_reward |= frame.kind == Kind.REWARD
            progress = {
                Kind.MAP: (Verb.TRAVEL,), Kind.EVENT: (Verb.CHOOSE_OPTION,), Kind.SHOP: (Verb.PROCEED,),
                Kind.REWARD: (Verb.PROCEED, Verb.CHOOSE_OPTION), Kind.REST: (Verb.PROCEED, Verb.CHOOSE_OPTION),
                Kind.CARD_REWARD: (Verb.PICK_OFFER_CARD,),
                Kind.CARD_SELECT: (Verb.CONFIRM_SELECTION, Verb.SELECT_CARD, Verb.CANCEL, Verb.SKIP),
                Kind.TREASURE: ((Verb.OPEN_CHEST,) if frame.glob[G.KIND_SUB] == TreasurePhase.CHEST
                                else (Verb.CHOOSE_OPTION,)),
            }
            matches = [int(index) for verb in progress[Kind(frame.kind)]
                       for index in np.flatnonzero(frame.actions[:, ActionCol.VERB] == verb)]
            assert matches, f"no progress action for {Kind(frame.kind).name}"
            action = matches[0]
        frame = host.step(action)
    assert seen_combat and seen_reward and seen_prompt
    host.close()


def test_h5_treasure_votes_shared_rows_and_no_skip(gen: FrameGen) -> None:
    rng = np.random.default_rng(41)
    ctx = gen.random_ctx(rng, 4, view=ViewMode.TEAM)
    ctx.acting_seat = 2
    ctx.treasure_votes = {0: 1, 1: 2}
    frame = parse_frame(gen.frame(rng, Kind.TREASURE, ctx, kind_sub=TreasurePhase.RELIC))
    assert not frame_problems(frame)
    options = frame.table(Table.OPTIONS)
    assert (options[:, OptionCol.SEAT] == 2).all()
    assert options[1, OptionCol.FLAGS] & 1 << 7
    assert options[2, OptionCol.FLAGS] & 1 << 8
    assert not np.any(frame.actions[:, ActionCol.VERB] == Verb.SKIP)
    sample = Tokenizer(gen.vocab).encode(frame)
    assert sample.n_actions == frame.n_actions


def test_h5_workload_and_capacity(vocab: Vocab) -> None:
    samples = workload(vocab, 16, 71)
    large = [s for s in samples if s.cond[1] == 4]
    assert all(150 <= s.n_tokens <= 300 for s in large)
    assert max(s.n_actions for s in large) >= 200
    tok = Tokenizer(vocab, TokenizerConfig(token_buckets=MIXED_BUCKETS))
    report = shape_report(samples, tok, max_batch=8, micro_batch=8, micro_tokens=2048)
    assert report["graph_keys"] <= 64
    assert all(s["batch"] <= 8 for s in report["graph_shapes"])
    assert all(m["samples"] * m["tokens"] <= 2048 for m in report["learner_micros"])
    assert report["max_graph_input_bytes_exact"] > 0


def test_actor_graph_shape_growth_is_monotone_and_replaces_old_shape() -> None:
    # Exercise actual cache-growth bookkeeping with a CPU stand-in for the CUDA runner.
    policy = cast(Any, object.__new__(TorchPolicy))
    policy.runner = SimpleNamespace(graphs={})
    policy._caps, policy._staging = {}, {}
    small = BatchShape(8, 256, 128, 256, 256, 256, 2048)
    large = BatchShape(8, 256, 512, 1024, 512, 256, 8192)
    assert policy._grow_shape(small) == small
    policy.runner.graphs[small] = object()
    assert policy._grow_shape(large) == large
    assert small not in policy.runner.graphs
    assert small not in policy._staging
    assert policy._grow_shape(small) == large
    assert len(policy._caps) == 1


def test_learner_graph_shape_growth_handles_h5_actions() -> None:
    stepper = cast(Any, object.__new__(MicroStepper))
    stepper._caps, stepper._res = {}, {}
    initial = BatchShape(51, 320, 200, 1200, 4000, 0, 20000)
    first = stepper.shape_for(initial, 51)
    stepper._res[first] = object()
    large = BatchShape(51, 320, 470, 1500, 4500, 0, 40000)
    grown = stepper.shape_for(large, 51)
    assert grown.fits(large)
    assert grown.batch == 52 and grown.actions >= 470
    assert first not in stepper._res
    assert stepper.shape_for(initial, 51) == grown
    assert len(stepper._caps) == 1


def test_zero_shot_measurements_use_all_four_seats(vocab: Vocab, tmp_path: Path) -> None:
    panel = make_panel("h5-tools", 2, ["IRONCLAD+DEFECT+IRONCLAD+DEFECT"], [10], rng_seed=19)
    tok = Tokenizer(vocab)
    policy = MeasuredPolicy(UniformPolicy(1), vocab)
    host = FakeHost(vocab, config=FakeHostConfig(mean_decisions=10, length_sigma=0, abort_prob=0, hashes=False))
    try:
        result, path = evaluate(panel, policy, tok, [host], tag="test", policy_info={}, out_dir=tmp_path)
    finally:
        host.close()
    assert not result.partial and result.completed_episodes == 2
    assert path.is_file()
    assert result.decisions == 20
    frames = policy.report()["DEFECT+DEFECT+IRONCLAD+IRONCLAD"]
    assert frames["tokens_per_frame"]["n"] == result.decisions
    assert frames["actions_per_frame"]["max"] > 0

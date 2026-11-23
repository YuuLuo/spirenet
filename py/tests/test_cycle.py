"""The environment rule recognizes only exact, observed deterministic-menu cycles."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from spire.cycle import MENU_KINDS, CycleBan, CycleStatus, state_key
from spire.proto import Frame, parse_frame
from spire.schema import ActionCol, CardSelectSub, G, ItemZone, Kind, OptionKind, PotionFlag, Table, Verb
from spire.synth import FrameBuilder
from spire.tokenize import CYCLE_BAN_SLOTS, Sample, Tokenizer
from spire.vocab import Vocab


def menu(kind: Kind = Kind.REWARD, *, step: int = 0, actions: int = 3,
         marker: int = 0, episode: int = 1, auto: int = 0,
         globals_: dict[G, int] | None = None) -> Frame:
    b = FrameBuilder()
    b.episode_id, b.step_id, b.auto_steps = episode, step, auto
    b.glob[G.KIND], b.glob[G.N_SEATS], b.glob[G.TOTAL_FLOOR] = kind, 1, 1
    b.glob[G.PROMPT_KEY] = marker
    if globals_ is not None:
        for key, value in globals_.items():
            b.glob[key] = value
    b.add(Table.SEATS, seat=0, hp=80, max_hp=80, alive=1, creature_row=-1)
    for action in range(actions):
        b.add(Table.ACTIONS, verb=Verb.PROCEED, seat=0, aux=action)
    return parse_frame(b.build(hashes=False))


def visit(cycle: CycleBan, tok: Tokenizer, frame: Frame, action: int = 0) -> tuple[Sample, CycleStatus]:
    sample = tok.encode(frame)
    status = cycle.observe(frame, sample)
    cycle.record(frame, action)
    return sample, status


def banned(sample: Sample) -> list[int]:
    return [int(value) - 1 for value in sample.history[-CYCLE_BAN_SLOTS:] if value]


@pytest.fixture
def architect_vocab(vocab: Vocab) -> Vocab:
    raw = deepcopy(dict(vocab.raw))
    raw["categories"]["event"]["entries"][2] = "THE_ARCHITECT"
    raw["categories"]["event_option_key"]["entries"][2] = "THE_ARCHITECT.dialogue"
    return Vocab(raw)


def event_choice(event_id: int, option_id: int, *, step: int = 0, auto: int = 0) -> Frame:
    """One dialogue option plus a real potion-discard alternative, so dialogue is not auto-only."""
    b = FrameBuilder()
    b.episode_id, b.step_id, b.auto_steps = 1, step, auto
    b.glob[G.KIND], b.glob[G.N_SEATS], b.glob[G.TOTAL_FLOOR] = Kind.EVENT, 1, 1
    b.glob[G.EVENT_ID] = event_id
    b.add(Table.SEATS, seat=0, hp=80, max_hp=80, alive=1, creature_row=-1)
    b.add(Table.POTIONS, seat=0, zone=ItemZone.OWNED, slot=0, potion_id=2,
          flags=PotionFlag.DISCARDABLE)
    b.add(Table.OPTIONS, kind=OptionKind.EVENT_OPTION, seat=0, pos=0, key_id=option_id)
    b.add(Table.ACTIONS, verb=Verb.DISCARD_POTION, seat=0, src_table=Table.POTIONS, src_row=0)
    b.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=0)
    frame = parse_frame(b.build(hashes=False))
    assert frame.n_actions == 2
    assert tuple(frame.actions[:, ActionCol.VERB]) == (Verb.DISCARD_POTION, Verb.CHOOSE_OPTION)
    return frame


def test_first_closed_cycle_bans_departing_action_and_preserves_other_history(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    initial, first = visit(cycle, tok, menu(), 0)
    visit(cycle, tok, menu(Kind.CARD_REWARD, step=1), 1)
    frame = menu(step=2)
    sample = tok.encode(frame)
    sample.history[:-CYCLE_BAN_SLOTS] = 7
    status = cycle.observe(frame, sample)
    assert first == CycleStatus() and not banned(initial)
    assert status == CycleStatus(banned=True) and banned(sample) == [0]
    assert np.all(sample.history[:-CYCLE_BAN_SLOTS] == 7)
    assert cycle.path == [] and cycle.positions == {}
    cycle.record(frame, 1)


@pytest.mark.parametrize("field", ["types", "raw", "rel", "edges", "cond", "counts"])
def test_identity_covers_every_legacy_sample_field(vocab: Vocab, field: str) -> None:
    sample = Tokenizer(vocab).encode(menu())
    before = state_key(sample)
    if field == "counts":
        a, b, c, d = sample.counts
        sample.counts = a + 1, b, c, d
    elif field == "edges":
        assert sample.edges.size == 0
        sample.edges = np.ones((1, 3), np.int16)
    else:
        array = getattr(sample, field)
        array.flat[0] += 1
    assert state_key(sample) != before


def test_identity_ignores_entire_history_and_protocol_step(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    first, second = tok.encode(menu()), tok.encode(menu(step=9, auto=3))
    second.history[:] = 19
    assert state_key(first) == state_key(second)


def team_menu(seat: int, step: int, kind: Kind = Kind.REWARD) -> Frame:
    b = FrameBuilder()
    b.episode_id, b.step_id = 1, step
    b.glob[G.KIND], b.glob[G.N_SEATS], b.glob[G.TOTAL_FLOOR] = kind, 4, 1
    b.glob[G.ACTING_SEAT] = seat
    for index in range(4):
        b.add(Table.SEATS, seat=index, hp=80, max_hp=80, alive=1, creature_row=-1)
        # Stage A deliberately ignores these erratum words; seat identity must already suffice.
        b.glob[52 + index] = Kind.REWARD
        b.glob[56 + index] = int(index < seat)
    for action in range(3):
        b.add(Table.ACTIONS, verb=Verb.PROCEED, seat=seat, aux=action)
    return parse_frame(b.build(hashes=False))


def test_seat_major_identical_reward_chains_are_distinct_and_real_cycles_still_ban(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    keys = []
    for seat in range(4):
        sample, status = visit(cycle, tok, team_menu(seat, seat * 3))
        keys.append(state_key(sample))
        assert status == CycleStatus() and not banned(sample)
        visit(cycle, tok, team_menu(seat, seat * 3 + 1, Kind.CARD_REWARD))
        returned, status = visit(cycle, tok, team_menu(seat, seat * 3 + 2), 1)
        assert status.banned and banned(returned) == [0]
    assert len(set(keys)) == 4  # neither acting-seat nor action-seat is erased by symmetric weights


def test_visible_change_has_its_own_legal_set(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    visit(cycle, tok, menu(step=0), 0)
    returned, _ = visit(cycle, tok, menu(step=1), 1)
    changed, status = visit(cycle, tok, menu(step=2, marker=2), 0)
    assert banned(returned) == [0]
    assert not banned(changed) and status == CycleStatus()
    # Exact old states retain their own bans, even after a different observation.
    original, _ = visit(cycle, tok, menu(step=3), 2)
    assert banned(original) == [0, 1]


def test_nested_subscreens_close_innermost_then_parent(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    visit(cycle, tok, menu(Kind.SHOP), 0)
    visit(cycle, tok, menu(Kind.CARD_SELECT, step=1), 0)
    visit(cycle, tok, menu(Kind.CONFIRM, step=2), 0)
    child, _ = visit(cycle, tok, menu(Kind.CARD_SELECT, step=3), 1)
    assert banned(child) == [0] and len(cycle.path) == 2
    parent, _ = visit(cycle, tok, menu(Kind.SHOP, step=4), 1)
    assert banned(parent) == [0] and len(cycle.path) == 1
    assert len(cycle.bans) == 2


def test_two_offers_ban_only_the_observed_cycle_actions(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    visit(cycle, tok, menu(), 0)
    visit(cycle, tok, menu(Kind.CARD_REWARD, step=1, marker=2), 0)
    first, _ = visit(cycle, tok, menu(step=2), 1)
    visit(cycle, tok, menu(Kind.CARD_REWARD, step=3, marker=3), 0)
    both, _ = visit(cycle, tok, menu(step=4), 2)
    assert banned(first) == [0] and banned(both) == [0, 1]
    assert len(cycle.bans) == 1


@pytest.mark.parametrize("kind", [Kind.COMBAT, Kind.CRYSTAL])
def test_nonmenu_never_banned_and_breaks_open_path(vocab: Vocab, kind: Kind) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    visit(cycle, tok, menu(), 0)
    for step in (1, 2):
        sample, status = visit(cycle, tok, menu(kind, step=step), 0)
        assert not banned(sample) and status == CycleStatus()
        assert not cycle.path
    returned, _ = visit(cycle, tok, menu(step=3), 0)
    assert not banned(returned) and not cycle.bans


def test_combat_selection_breaks_path_and_is_not_banned(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    visit(cycle, tok, menu(), 0)
    for step in (1, 2):
        sample, status = visit(cycle, tok, menu(Kind.CARD_SELECT, step=step, globals_={G.IN_COMBAT: 1}))
        assert not banned(sample) and status == CycleStatus() and not cycle.path
    sample, _ = visit(cycle, tok, menu(step=3))
    assert not banned(sample)


def test_path_crosses_auto_advance_gaps(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    visit(cycle, tok, menu(), 0)
    visit(cycle, tok, menu(Kind.CARD_REWARD, step=1, auto=1), 0)
    sample, _ = visit(cycle, tok, menu(step=2), 1)
    assert banned(sample) == [0]  # forced decisions on the way do not hide the cycle
    sample, _ = visit(cycle, tok, menu(step=3, auto=2), 2)
    assert banned(sample) == [0, 1]


def test_event_option_returning_through_an_auto_advanced_popup_is_banned(vocab: Vocab) -> None:
    # Trial REJECT page: DOUBLE_DOWN -> abandon popup with NO only (auto-advanced) -> the same page again
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    page, status = visit(cycle, tok, menu(Kind.EVENT, actions=2), 1)
    assert not banned(page) and status == CycleStatus()
    again, status = visit(cycle, tok, menu(Kind.EVENT, actions=2, step=1, auto=1), 0)
    assert banned(again) == [1] and status == CycleStatus(banned=True)
    changed, status = visit(cycle, tok, menu(Kind.EVENT, actions=2, step=2, marker=5), 1)
    assert not banned(changed) and status == CycleStatus()  # another page: its own legal set


def test_architect_dialogue_can_repeat_identical_input_with_potion_discard_legal(architect_vocab: Vocab) -> None:
    # The Architect's dialogue counter advances behind identical public keys/rows.
    # A discardable potion supplies the second legal action, making each dialogue observable.
    vocab = architect_vocab
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    event_id = vocab["event"].index["THE_ARCHITECT"]
    option_id = vocab["event_option_key"].index["THE_ARCHITECT.dialogue"]
    first_key = state_key(tok.encode(event_choice(event_id, option_id)))
    for step in range(6):
        frame = event_choice(event_id, option_id, step=step, auto=step % 2)
        assert state_key(tok.encode(frame)) == first_key
        sample, status = visit(cycle, tok, frame, 1)
        assert not banned(sample) and status == CycleStatus()
        assert cycle.path == [] and cycle.positions == {} and cycle.key is None
        assert not cycle.bans


def test_architect_dialogue_breaks_ancestor_cycle_path(architect_vocab: Vocab) -> None:
    vocab = architect_vocab
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    event_id = vocab["event"].index["THE_ARCHITECT"]
    option_id = vocab["event_option_key"].index["THE_ARCHITECT.dialogue"]
    visit(cycle, tok, menu(), 0)
    visit(cycle, tok, event_choice(event_id, option_id, step=1), 1)
    returned, status = visit(cycle, tok, menu(step=2), 0)
    assert not banned(returned) and status == CycleStatus() and not cycle.bans
    assert len(cycle.path) == 1  # the arriving parent starts a fresh path


@pytest.mark.parametrize("other_event,other_option,action", [
    (False, True, 1),  # another option in The Architect is still subject to the ban
    (True, False, 1),  # the matching option key in another event is not exempt
    (False, False, 0),  # selecting a different action on the same page is not exempt
])
def test_architect_exemption_requires_exact_selected_public_identity(
        architect_vocab: Vocab, other_event: bool, other_option: bool, action: int) -> None:
    vocab = architect_vocab
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    event_id = 3 if other_event else vocab["event"].index["THE_ARCHITECT"]
    option_id = 3 if other_option else vocab["event_option_key"].index["THE_ARCHITECT.dialogue"]
    visit(cycle, tok, event_choice(event_id, option_id), action)
    returned, status = visit(cycle, tok, event_choice(event_id, option_id, step=1), 1 - action)
    assert banned(returned) == [action] and status == CycleStatus(banned=True)


@pytest.mark.parametrize("field", [G.ACT_INDEX, G.TOTAL_FLOOR, G.CUR_ROW, G.CUR_COL, G.ROOM_TYPE])
def test_each_room_identity_component_clears_path_and_bans(vocab: Vocab, field: G) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    visit(cycle, tok, menu(), 0)
    visit(cycle, tok, menu(step=1), 1)
    sample, _ = visit(cycle, tok, menu(step=2, globals_={field: 3}), 0)
    assert not banned(sample) and not cycle.bans and len(cycle.path) == 1


def test_episode_and_explicit_reset_clear(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    visit(cycle, tok, menu(), 0)
    visit(cycle, tok, menu(step=1), 1)
    sample, _ = visit(cycle, tok, menu(episode=2), 0)
    assert not banned(sample) and not cycle.bans
    cycle.clear()
    assert not cycle.bans and not cycle.path and cycle.room is None
    sample, _ = visit(cycle, tok, menu(episode=2, step=1), 0)
    assert not banned(sample)


@pytest.mark.parametrize("actions", [1, 2])
def test_all_banned_decision_is_playable_and_counted(vocab: Vocab, actions: int) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    for action in range(actions):
        sample, _ = visit(cycle, tok, menu(actions=actions, step=action), action)
        assert banned(sample) == list(range(action))
    for step in (actions, actions + 1):
        sample, status = visit(cycle, tok, menu(actions=actions, step=step), 0)
        assert not banned(sample) and status == CycleStatus(all_banned=True)
        assert len(cycle.bans[state_key(sample)]) == actions


def test_seventh_cycle_action_is_not_banned_and_is_counted(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    for action in range(7):
        sample, status = visit(cycle, tok, menu(actions=8, step=action), action)
        assert banned(sample) == list(range(action))
        assert status.capacity_hits == 0
    for step, action in ((7, 7), (8, 6)):
        sample, status = visit(cycle, tok, menu(actions=8, step=step), action)
        assert banned(sample) == list(range(6))
        assert status == CycleStatus(banned=True, capacity_hits=1)
        assert len(cycle.bans[state_key(sample)]) == 6


def test_scope_contains_exactly_the_audited_menu_kinds() -> None:
    assert MENU_KINDS == {Kind.REWARD, Kind.CARD_REWARD, Kind.CARD_SELECT, Kind.CONFIRM,
                          Kind.BUNDLE, Kind.SHOP, Kind.REST, Kind.MAP, Kind.TREASURE, Kind.EVENT}


def test_record_enforces_alignment_and_ban(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab)
    visit(cycle, tok, menu(), 0)
    frame = menu(step=1)
    cycle.observe(frame, tok.encode(frame))
    with pytest.raises(AssertionError, match="cycle-banned"):
        cycle.record(frame, 0)
    with pytest.raises(AssertionError, match="record the selected"):
        cycle.observe(frame, tok.encode(frame))
    cycle.record(frame, 1)


# =========================================================================== commit_select


def select(sub: CardSelectSub, verbs: tuple[Verb, ...], *, step: int = 0, in_combat: int = 0) -> Frame:
    """A card prompt reduced to its action verbs (PROCEED rows stand in for the selectable cards)."""
    b = FrameBuilder()
    b.episode_id, b.step_id = 1, step
    b.glob[G.KIND], b.glob[G.KIND_SUB], b.glob[G.N_SEATS], b.glob[G.TOTAL_FLOOR] = Kind.CARD_SELECT, sub, 1, 1
    b.glob[G.IN_COMBAT] = in_combat
    b.add(Table.SEATS, seat=0, hp=80, max_hp=80, alive=1, creature_row=-1)
    for row, verb in enumerate(verbs):
        b.add(Table.ACTIONS, verb=verb, seat=0, aux=row)
    return parse_frame(b.build(hashes=False))


def rows(frame: Frame, verb: Verb) -> list[int]:
    """Action rows of a verb (the frame holds its actions in canonical order, not in the order they were added)."""
    return [int(row) for row in np.flatnonzero(frame.actions[:, ActionCol.VERB] == verb)]


@pytest.mark.parametrize("sub", [CardSelectSub.DECK_UPGRADE, CardSelectSub.DECK_ENCHANT])
def test_commit_select_masks_cancel_at_the_grid_and_at_the_confirm_step(vocab: Vocab, sub: CardSelectSub) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab, commit_select=True)
    frame = select(sub, (Verb.PROCEED, Verb.PROCEED, Verb.CANCEL))
    grid, status = visit(cycle, tok, frame, rows(frame, Verb.PROCEED)[0])
    assert banned(grid) == rows(frame, Verb.CANCEL) and status.banned and not status.all_banned
    frame = select(sub, (Verb.CONFIRM_SELECTION, Verb.CANCEL), step=1)
    confirm = tok.encode(frame)
    assert cycle.observe(frame, confirm).banned and banned(confirm) == rows(frame, Verb.CANCEL)
    with pytest.raises(AssertionError, match="banned"):
        cycle.record(frame, rows(frame, Verb.CANCEL)[0])
    cycle.record(frame, rows(frame, Verb.CONFIRM_SELECTION)[0])


def test_commit_select_is_off_by_default_and_leaves_other_prompts_alone(vocab: Vocab) -> None:
    tok = Tokenizer(vocab)
    upgrade = select(CardSelectSub.DECK_UPGRADE, (Verb.CONFIRM_SELECTION, Verb.CANCEL))
    sample, status = visit(CycleBan(vocab), tok, upgrade, 1)
    assert banned(sample) == [] and status == CycleStatus()
    cycle = CycleBan(vocab, commit_select=True)
    for step, frame in enumerate((
            select(CardSelectSub.DECK_REMOVAL, (Verb.CONFIRM_SELECTION, Verb.CANCEL), step=1),
            select(CardSelectSub.DECK_TRANSFORM, (Verb.PROCEED, Verb.CANCEL), step=2),
            select(CardSelectSub.DECK_UPGRADE, (Verb.CANCEL,), step=3),                      # nothing else to do
            select(CardSelectSub.DECK_UPGRADE, (Verb.PROCEED, Verb.CANCEL), step=4, in_combat=1))):
        sample, status = visit(cycle, tok, frame, frame.n_actions - 1)
        assert banned(sample) == [] and not status.banned, step


def test_commit_select_comes_first_and_keeps_the_cycle_bans(vocab: Vocab) -> None:
    tok, cycle = Tokenizer(vocab), CycleBan(vocab, commit_select=True)
    verbs = (Verb.PROCEED, Verb.PROCEED, Verb.CANCEL)
    frame = select(CardSelectSub.DECK_UPGRADE, verbs)
    first, second = rows(frame, Verb.PROCEED)
    visit(cycle, tok, frame, first)
    frame = select(CardSelectSub.DECK_UPGRADE, verbs, step=1)  # the same prompt again: ``first`` closed a cycle
    again, status = visit(cycle, tok, frame, second)
    assert banned(again) == rows(frame, Verb.CANCEL) + [first] and status.banned and not status.all_banned

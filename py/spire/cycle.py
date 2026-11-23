"""Room-local bans on actions observed to close a public, deterministic menu cycle."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from struct import Struct

import numpy as np

from .proto import Frame
from .removal import RemovalRule, is_removal_prompt
from .rules import RULES_MODES
from .schema import ActionCol, CardSelectSub, G, Kind, OptionCol, OptionKind, Table, Verb
from .tokenize import CYCLE_BAN_SLOTS, Sample
from .vocab import Vocab


# EVENT joined the scope on 2026-11-15: Trial's DOUBLE_DOWN opens the abandon-run popup, the host offers only
# NO there (auto-advanced) and the REJECT page returns unchanged with DOUBLE_DOWN still enabled. The policy
# repeated it to the 30,000-decision cap in 64 episodes within 42 minutes of run l1-third.
MENU_KINDS = frozenset((Kind.REWARD, Kind.CARD_REWARD, Kind.CARD_SELECT, Kind.CONFIRM,
                       Kind.BUNDLE, Kind.SHOP, Kind.REST, Kind.MAP, Kind.TREASURE, Kind.EVENT))
# Card prompts a seat opens on purpose and can only leave without progress through CANCEL. Measured on 2026-11-16
# (notes/audit/strategy-audit.md, 520 greedy 4-seat games): the policy chose SMITH, picked a card and then CANCELled
# at 2,530 of 2,530 confirm steps - 0 upgrades in 16,304 rest visits - and the returning rest screen got SMITH
# cycle-banned, so it healed. With ``commit_select`` the CANCEL rows of these prompts are masked: choosing SMITH is
# the decision. Removal and transform prompts are left out on purpose: there the policy still picks its best card.
COMMIT_SUBS = frozenset((CardSelectSub.DECK_UPGRADE, CardSelectSub.DECK_ENCHANT))
# ``removal_rule`` (notes/solo/removal-spec.md): an out-of-combat DECK_REMOVAL prompt keeps exactly ONE legal row,
# the rule's (spire.removal). Six ban slots cannot hold the complement, so the slots carry it as one negative entry:
# slot 0 = -(row + 1), the other slots 0 = "only this row is legal" (model.py and UniformPolicy decode it).
_COUNTS = Struct("<4I")


def state_key(sample: Sample) -> bytes:
    """Hash the complete tokenized policy input, excluding its history side record.

    Counts fix entity-array lengths; raw width and conditioning length are fixed
    by the tokenizer. Edges are the sole remaining variable-length byte region.
    Hash updates consume the arrays' buffers without allocating their byte copies.
    """
    key = sha256()
    key.update(_COUNTS.pack(*sample.counts))
    for array in (sample.types, sample.raw, sample.rel, sample.edges, sample.cond):
        key.update(array.data)
    return key.digest()


@dataclass(frozen=True)
class CycleStatus:
    banned: bool = False
    all_banned: bool = False
    capacity_hits: int = 0
    removal_rule: bool = False  # the removal rule fixed this decision (one legal row)


_CLEAR = CycleStatus()


class CycleBan:
    """Observe before inference; record exactly one selected action afterwards.

    The unresolved path has unique keys. Returning to a path key bans the action
    that left that position, then removes the closed suffix. Bans persist for
    their exact state until the room/episode changes. COMBAT, CRYSTAL and every
    in-combat decision break the path: their hidden progress need not be visible.
    The public Architect dialogue identity also breaks the path on selection.
    ``commit_select`` adds the CANCEL rows of the prompts in ``COMMIT_SUBS`` to the mask (they come first).
    ``removal`` (the removal rule) replaces the mask of a removal prompt by its one legal row: ``active_bans`` then
    holds every other row, ``rule_row`` / ``rule_class`` the rule's row and the class of its card (None = CONFIRM).
    ``mode`` (``TokenizerConfig.rules_mode``, spire.rules): "mask" and "teach" as above; in "free" a removal prompt
    bans only its cycle rows (slots 1.. after the rule's slot 0; ``active_bans`` = those rows), and a removal prompt
    whose rule row closed a loop of it is an ordinary prompt (no slot 0, ``rule_row`` None).
    """

    def __init__(self, vocab: Vocab, *, commit_select: bool = False, removal: RemovalRule | None = None,
                 mode: str = "mask") -> None:
        assert mode in RULES_MODES, mode
        self.vocab = vocab
        self.commit_select = commit_select
        self.removal = removal
        self.mode = mode
        self.clear()

    def clear(self) -> None:
        self.room: tuple[int, ...] | None = None
        self.observed: tuple[int, int] | None = None
        self.path: list[tuple[bytes, int]] = []
        self.positions: dict[bytes, int] = {}
        self.bans: dict[bytes, list[int]] = {}
        self.key: bytes | None = None
        self.active_bans: tuple[int, ...] = ()
        self.status = _CLEAR
        self.rule_row: int | None = None
        self.rule_class: str | None = None

    def fork(self) -> "CycleBan":
        """An independent tracker in this one's state between two decisions (a fork child of the game, practice
        fights). The removal rule is stateless (shared); its last row / class are copied."""
        assert self.observed is None, "fork between record() and observe()"
        other = CycleBan(self.vocab, commit_select=self.commit_select, removal=self.removal, mode=self.mode)
        other.room = self.room
        other.path = list(self.path)
        other.positions = dict(self.positions)
        other.bans = {key: list(banned) for key, banned in self.bans.items()}
        other.key = self.key
        other.active_bans = self.active_bans
        other.status = self.status
        other.rule_row = self.rule_row
        other.rule_class = self.rule_class
        assert set(vars(other)) == set(vars(self)), "CycleBan.fork misses a field"
        return other

    def observe(self, frame: Frame, sample: Sample) -> CycleStatus:
        assert self.observed is None, "record the selected action before observing again"
        g = frame.gvals
        room = (frame.episode_id, g[G.ACT_INDEX], g[G.TOTAL_FLOOR], g[G.CUR_ROW],
                g[G.CUR_COL], g[G.ROOM_TYPE])
        if room != self.room:
            self.clear()
            self.room = room
        self.observed = (frame.episode_id, frame.step_id)
        self.active_bans = ()
        self.status = _CLEAR
        self.rule_row = self.rule_class = None
        sample.history[-CYCLE_BAN_SLOTS:] = 0
        # The v0.113 driver audit permits auto gaps (including AutoGold); their kinds are not in the frame.
        # Equal inputs alone are insufficient: record() breaks paths at the known aliased dialogue below.
        if g[G.KIND] not in MENU_KINDS or g[G.IN_COMBAT]:
            self.path.clear()
            self.positions.clear()
            self.key = None
            return self.status

        self.key = key = state_key(sample)
        index = self.positions.get(key)
        capacity_hits = 0
        if index is not None:
            action = self.path[index][1]
            banned = self.bans.setdefault(key, [])
            if action not in banned:
                if len(banned) < CYCLE_BAN_SLOTS:
                    assert action < 32767, "cycle action index exceeds the int16 side record"
                    banned.append(action)
                else:
                    capacity_hits = 1
            for old_key, _ in self.path[index:]:
                del self.positions[old_key]
            del self.path[index:]

        banned = self.bans.get(key, [])
        assert all(0 <= action < sample.n_actions for action in banned)
        if self.removal is not None and is_removal_prompt(frame) and self.mode == "free":
            row, cls = self.removal.choose(frame)
            if row not in banned:  # else: the rule's row closed a loop of this prompt, an ordinary prompt below
                cyc = banned[:CYCLE_BAN_SLOTS - 1]
                self.rule_row, self.rule_class = row, cls
                self.active_bans = tuple(cyc)
                slots = sample.history[-CYCLE_BAN_SLOTS:]
                slots[0] = -(row + 1)
                slots[1:1 + len(cyc)] = [action + 1 for action in cyc]
                self.status = CycleStatus(bool(cyc), False, capacity_hits, removal_rule=True)
                return self.status
        elif self.removal is not None and is_removal_prompt(frame):
            row, cls = self.removal.choose(frame)
            assert row not in banned, "the removal rule's row is cycle-banned"
            self.rule_row, self.rule_class = row, cls
            self.active_bans = tuple(a for a in range(sample.n_actions) if a != row)
            sample.history[-CYCLE_BAN_SLOTS] = -(row + 1)
            self.status = CycleStatus(capacity_hits=capacity_hits, removal_rule=True)
            return self.status
        if self.commit_select:
            cancel = self._cancel_rows(frame)
            banned = (cancel + [action for action in banned if action not in cancel])[:CYCLE_BAN_SLOTS]
        all_banned = bool(banned) and len(banned) == sample.n_actions
        if banned and not all_banned:
            self.active_bans = tuple(banned)
            slots = sample.history[-CYCLE_BAN_SLOTS:]
            slots[:len(banned)] = [action + 1 for action in banned]
        if banned or capacity_hits:
            self.status = CycleStatus(bool(self.active_bans), all_banned, capacity_hits)
        return self.status

    def record(self, frame: Frame, action: int) -> None:
        assert self.observed == (frame.episode_id, frame.step_id)
        assert 0 <= action < frame.n_actions
        assert action not in self.active_bans, "policy selected a cycle-banned action"
        if self._advances_architect_dialogue(frame, action):
            # TheArchitect.cs:336,344-357 advances CurrentLineIndex while retaining PROCEED.description;
            # host/SpireHost.Obs/Vocab/VocabExport.cs:82-85 folds every line to THE_ARCHITECT.dialogue.
            # A held potion adds DISCARD (Driver.Rooms/Potions.cs:109), making equal-looking lines policy
            # decisions. This is finite progress toward WinRun, not a cycle. Break the whole open path
            # so an ancestor cannot be banned across that hidden progress either.
            self.path.clear()
            self.positions.clear()
            self.key = None
        if self.key is not None:
            assert self.key not in self.positions
            self.positions[self.key] = len(self.path)
            self.path.append((self.key, action))
        self.observed = None

    @staticmethod
    def _cancel_rows(frame: Frame) -> list[int]:
        """CANCEL rows of an upgrade / enchant prompt that offers anything else (observe() excludes combat)."""
        g = frame.gvals
        if g[G.KIND] != Kind.CARD_SELECT or g[G.KIND_SUB] not in COMMIT_SUBS:
            return []
        rows = [int(row) for row in np.flatnonzero(frame.actions[:, ActionCol.VERB] == Verb.CANCEL)]
        assert len(rows) <= 1, "a card prompt has one CANCEL"
        return rows if len(rows) < frame.n_actions else []

    def _advances_architect_dialogue(self, frame: Frame, action: int) -> bool:
        if (frame.kind != Kind.EVENT
                or self.vocab.name("event", frame.gvals[G.EVENT_ID]) != "THE_ARCHITECT"):
            return False
        row = frame.actions[action]
        if row[ActionCol.VERB] != Verb.CHOOSE_OPTION or row[ActionCol.SRC_TABLE] != Table.OPTIONS:
            return False
        option = frame.table(Table.OPTIONS)[row[ActionCol.SRC_ROW]]
        return (option[OptionCol.KIND] == OptionKind.EVENT_OPTION
                and self.vocab.name("event_option_key", int(option[OptionCol.KEY_ID])) == "THE_ARCHITECT.dialogue")

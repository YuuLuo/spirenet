"""Loaders for ``vocab.json`` / ``catalog.json`` (schema 12) plus synthetic stand-ins for tests.

* :class:`Vocab` - category sizes, id <-> name lookup, embedding table sizes, canonical hash.
* :class:`Catalog` - static per-id attributes, turned into dense feature matrices.
* :func:`check_frame_hashes` - frame header VOCAB_HASH / CATALOG_HASH against the loaded files.
* :func:`make_synthetic_vocab` / :func:`make_synthetic_catalog` - deterministic fakes with the
  same structure, until the host's ``--export-vocab`` exists.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .proto import Frame
from .schema import VOCAB_CATEGORIES, VOCAB_PAD, VOCAB_UNK, SchemaError

PAD_NAME = "<PAD>"
UNK_NAME = "<UNK>"
EMBED_SLACK = 128
EMBED_ALIGN = 64


class VocabMismatch(SchemaError):
    """Frame header hashes disagree with the loaded vocab / catalog."""


def canonical_json(obj: Any) -> bytes:
    """Canonical JSON (schema 12.2): sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_hash(obj: Any) -> int:
    """First 4 bytes of SHA-256 of the canonical JSON, read as little-endian int32."""
    return int.from_bytes(hashlib.sha256(canonical_json(obj)).digest()[:4], "little", signed=True)


def embedding_rows(n_entries: int) -> int:
    """Embedding table size for a category: ceil((len + 128) / 64) * 64 (schema 12.2)."""
    return math.ceil((n_entries + EMBED_SLACK) / EMBED_ALIGN) * EMBED_ALIGN


@dataclass(frozen=True)
class Category:
    name: str
    entries: tuple[str, ...]
    retired: frozenset[str]
    meta: Mapping[str, Mapping[str, Any]]
    index: Mapping[str, int] = field(repr=False)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def embedding_rows(self) -> int:
        return embedding_rows(len(self.entries))


class Vocab:
    """``vocab.json``: per category, index 0 = ``<PAD>``, 1 = ``<UNK>``, append-only."""

    def __init__(self, obj: Mapping[str, Any]) -> None:
        cats = obj.get("categories")
        if not isinstance(cats, Mapping):
            raise SchemaError("vocab.json: missing 'categories'")
        self.raw = obj
        self.schema = int(obj.get("schema", 0))
        self.game_version = str(obj.get("game_version", ""))
        self.vocab_version = int(obj.get("vocab_version", 0))
        self.hash = canonical_hash(obj)
        self.categories: dict[str, Category] = {}
        for name, c in cats.items():
            entries = tuple(c["entries"])
            if len(entries) < 2 or entries[VOCAB_PAD] != PAD_NAME or entries[VOCAB_UNK] != UNK_NAME:
                raise SchemaError(f"vocab category {name}: entries must start with <PAD>, <UNK>")
            index = {e: i for i, e in enumerate(entries)}
            if len(index) != len(entries):
                raise SchemaError(f"vocab category {name}: duplicate entries")
            self.categories[name] = Category(
                name, entries, frozenset(c.get("retired", ())), dict(c.get("meta", {})), index
            )
        missing = [c for c in VOCAB_CATEGORIES if c not in self.categories]
        if missing:
            raise SchemaError(f"vocab.json lacks categories: {missing}")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __getitem__(self, cat: str) -> Category:
        return self.categories[cat]

    def __contains__(self, cat: str) -> bool:
        return cat in self.categories

    def size(self, cat: str) -> int:
        return len(self.categories[cat].entries)

    def embedding_rows(self, cat: str) -> int:
        return self.categories[cat].embedding_rows

    def sizes(self) -> dict[str, int]:
        return {k: len(c) for k, c in self.categories.items()}

    def id(self, cat: str, name: str) -> int:
        """Vocab id of ``name``; ``<UNK>`` (1) when absent."""
        return self.categories[cat].index.get(name, VOCAB_UNK)

    def ids(self, cat: str, names: Sequence[str]) -> NDArray[np.int32]:
        index = self.categories[cat].index
        return np.fromiter((index.get(n, VOCAB_UNK) for n in names), dtype=np.int32, count=len(names))

    def name(self, cat: str, idx: int) -> str:
        entries = self.categories[cat].entries
        if not 0 <= idx < len(entries):
            raise IndexError(f"{cat}: id {idx} outside [0, {len(entries)})")
        return entries[idx]


# --------------------------------------------------------------------------- catalog


@dataclass(frozen=True)
class FieldSpec:
    """How one catalog field becomes features.

    ``kind``: ``"num"`` (number -> float column), ``"bool"`` (0/1 float column),
    ``"id"`` (string -> vocab id of ``cat``, categorical column) or ``"multi"``
    (list of strings -> multi-hot float columns over vocab ``cat``).
    """

    name: str
    kind: str
    cat: str | None = None


_B = "bool"
DEFAULT_FEATURES: Mapping[str, tuple[FieldSpec, ...]] = {
    "card": (
        FieldSpec("type", "id", "enum:CardType"), FieldSpec("rarity", "id", "enum:CardRarity"),
        FieldSpec("pool", "id", "card_pool"), FieldSpec("character", "id", "character"),
        FieldSpec("target_type", "id", "enum:TargetType"),
        FieldSpec("multiplayer_constraint", "id", "enum:CardMultiplayerConstraint"),
        FieldSpec("orb_evoke_type", "id", "enum:OrbEvokeType"),
        FieldSpec("energy_cost", "num"), FieldSpec("costs_x", _B), FieldSpec("star_cost", "num"),
        FieldSpec("star_x", _B), FieldSpec("max_upgrade_level", "num"), FieldSpec("gains_block", _B),
        FieldSpec("can_be_generated_in_combat", _B), FieldSpec("is_basic_strike_or_defend", _B),
        FieldSpec("has_turn_end_in_hand_effect", _B),
        FieldSpec("keywords", "multi", "enum:CardKeyword"), FieldSpec("tags", "multi", "enum:CardTag"),
    ),
    "relic": (
        FieldSpec("rarity", "id", "enum:RelicRarity"), FieldSpec("pool", "id", "relic_pool"),
        FieldSpec("character", "id", "character"), FieldSpec("merchant_cost", "num"),
        FieldSpec("is_stackable", _B), FieldSpec("adds_pet", _B), FieldSpec("spawns_pets", _B),
        FieldSpec("has_upon_pickup_effect", _B), FieldSpec("is_allowed_in_shops", _B),
    ),
    "potion": (
        FieldSpec("rarity", "id", "enum:PotionRarity"), FieldSpec("usage", "id", "enum:PotionUsage"),
        FieldSpec("target_type", "id", "enum:TargetType"), FieldSpec("pool", "id", "potion_pool"),
        FieldSpec("can_be_generated_in_combat", _B),
    ),
    "power": (
        FieldSpec("type", "id", "enum:PowerType"), FieldSpec("stack_type", "id", "enum:PowerStackType"),
        FieldSpec("instance_type", "id", "enum:PowerInstanceType"), FieldSpec("allow_negative", _B),
        FieldSpec("should_scale_in_multiplayer", _B),
    ),
    "monster": (
        FieldSpec("hp_min", "num"), FieldSpec("hp_max", "num"), FieldSpec("hp_min_a8", "num"),
        FieldSpec("hp_max_a8", "num"), FieldSpec("hp_sentinel", _B),
    ),
    "encounter": (FieldSpec("room_type", "id", "enum:RoomType"),),
    "event": (FieldSpec("is_shared", _B), FieldSpec("is_ancient", _B)),
    "enchantment": (FieldSpec("show_amount", _B), FieldSpec("is_stackable", _B)),
    "affliction": (FieldSpec("can_afflict_unplayable", _B), FieldSpec("is_stackable", _B)),
    "character": (
        FieldSpec("starting_hp", "num"), FieldSpec("starting_gold", "num"),
        FieldSpec("max_energy", "num"), FieldSpec("base_orb_slots", "num"),
    ),
    "var_name": (FieldSpec("power", "id", "power"),),
}
"""Scalar fields of schema 12.4 per category. List/text fields (vars, printed_vars,
plain_text, starting_deck ...) are not featurised here."""

LEVELED = frozenset({"card"})  # catalog entries are lists indexed by upgrade level


@dataclass(frozen=True)
class StaticFeatures:
    """Dense static features of one category, row = vocab id.

    ``ids``: int32 (V, Fc) categorical vocab ids, ``nums``: float32 (V, Fn). For leveled
    categories (cards) both carry an upgrade-level axis: (V, L, F); levels beyond a card's
    maximum repeat its last level, so ``a[card_id, min(level, L - 1)]`` is always defined.
    Ids with no catalog entry (PAD/UNK/retired) are all-zero rows.
    """

    ids: NDArray[np.int32]
    nums: NDArray[np.float32]
    id_fields: tuple[str, ...]
    id_cats: tuple[str, ...]
    num_fields: tuple[str, ...]


class Catalog:
    """``catalog.json``: ``{"categories": {cat: {"<vocab id>": entry}}}``; card entries are
    lists indexed by upgrade level (schema 12.4)."""

    def __init__(self, obj: Mapping[str, Any], vocab: Vocab) -> None:
        cats = obj.get("categories")
        if not isinstance(cats, Mapping):
            raise SchemaError("catalog.json: missing 'categories'")
        self.raw = obj
        self.vocab = vocab
        self.hash = canonical_hash(obj)
        self.entries: dict[str, dict[int, Any]] = {}
        for cat, ents in cats.items():
            if cat not in vocab:
                raise SchemaError(f"catalog category {cat} not in vocab")
            size = vocab.size(cat)
            table: dict[int, Any] = {}
            for k, e in ents.items():
                i = int(k)
                if not 0 <= i < size:
                    raise SchemaError(f"catalog {cat}: id {i} outside vocab size {size}")
                if cat in LEVELED and (not isinstance(e, list) or not e):
                    raise SchemaError(f"catalog {cat}[{i}]: expected a non-empty per-level list")
                table[i] = e
            self.entries[cat] = table

    @classmethod
    def load(cls, path: str | Path, vocab: Vocab) -> "Catalog":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), vocab)

    def entry(self, cat: str, idx: int, level: int = 0) -> Mapping[str, Any] | None:
        e = self.entries.get(cat, {}).get(idx)
        if e is None or cat not in LEVELED:
            return e
        return e[min(level, len(e) - 1)]

    def features(self, cat: str, spec: Sequence[FieldSpec] | None = None) -> StaticFeatures:
        spec = tuple(DEFAULT_FEATURES.get(cat, ()) if spec is None else spec)
        v = self.vocab
        id_specs = [s for s in spec if s.kind == "id"]
        num_cols: list[str] = []
        for s in spec:
            if s.kind in ("num", "bool"):
                num_cols.append(s.name)
            elif s.kind == "multi":
                assert s.cat is not None
                num_cols.extend(f"{s.name}:{e}" for e in v[s.cat].entries[2:])
            elif s.kind != "id":
                raise ValueError(f"unknown field kind {s.kind}")
        ents = self.entries.get(cat, {})
        V = v.size(cat)
        L = max((len(e) for e in ents.values()), default=1) if cat in LEVELED else 1
        ids = np.zeros((V, L, len(id_specs)), dtype=np.int32)
        nums = np.zeros((V, L, len(num_cols)), dtype=np.float32)
        for i, e in ents.items():
            levels = e if cat in LEVELED else [e]
            for lv in range(L):
                rec = levels[min(lv, len(levels) - 1)]
                ids[i, lv] = [v.id(s.cat, rec[s.name]) if rec.get(s.name) else 0 for s in id_specs]  # type: ignore[arg-type]
                nums[i, lv] = _numeric_row(rec, spec, v)
        if cat not in LEVELED:
            ids, nums = ids[:, 0], nums[:, 0]
        return StaticFeatures(ids, nums, tuple(s.name for s in id_specs),
                              tuple(s.cat or "" for s in id_specs), tuple(num_cols))


def _numeric_row(rec: Mapping[str, Any], spec: Sequence[FieldSpec], v: Vocab) -> list[float]:
    row: list[float] = []
    for s in spec:
        if s.kind in ("num", "bool"):
            x = rec.get(s.name)
            row.append(float(x) if isinstance(x, (int, float)) else 0.0)
        elif s.kind == "multi":
            assert s.cat is not None
            hot = np.zeros(v.size(s.cat) - 2, dtype=np.float32)
            for name in rec.get(s.name) or ():
                j = v.id(s.cat, name) - 2
                if j >= 0:
                    hot[j] = 1.0
            row.extend(hot.tolist())
    return row


def check_hashes(vocab_hash: int, catalog_hash: int, expected_vocab: int, expected_catalog: int | None) -> None:
    """Raise :class:`VocabMismatch` unless frame header hashes equal the expected ones."""
    if vocab_hash != expected_vocab:
        raise VocabMismatch(f"frame vocab hash {vocab_hash:#x} != vocab.json {expected_vocab:#x}")
    if expected_catalog is not None and catalog_hash != expected_catalog:
        raise VocabMismatch(f"frame catalog hash {catalog_hash:#x} != catalog.json {expected_catalog:#x}")


def check_frame_hashes(frame: Frame, vocab: Vocab, catalog: Catalog | None = None) -> None:
    """Frame header VOCAB_HASH / CATALOG_HASH against the loaded vocab (and catalog)."""
    check_hashes(frame.vocab_hash, frame.catalog_hash, vocab.hash, None if catalog is None else catalog.hash)


# --------------------------------------------------------------------------- synthetic files

CHARACTERS = ("DEFECT", "FOOL", "IRONCLAD", "NECROBINDER", "REGENT", "SILENT")
_NAMED: Mapping[str, tuple[str, ...]] = {
    "character": CHARACTERS,
    "act": ("GLORY", "HIVE", "OVERGROWTH", "UNDERDOCKS"),
    "orb": ("DARK", "FROST", "GLASS", "LIGHTNING", "PLASMA"),
    "shop_section": ("CARD_REMOVAL", "CHARACTER_CARD", "COLORLESS_CARD", "POTION", "RELIC"),
    "rest_option": ("CLONE", "COOK", "DIG", "EVOLVE", "HATCH", "HEAL", "KINDLE", "LIFT", "MEND", "RESTORE", "SMITH"),
    "card_reward_alt": ("REROLL", "SACRIFICE", "SKIP"),
    "confirm_key": ("CONFIRM_NO", "CONFIRM_YES", "TRIAL_DOUBLE_DOWN"),
    "enum:CardType": ("Attack", "Curse", "Power", "Quest", "Skill", "Status"),
    "enum:CardRarity": ("Basic", "Common", "Uncommon", "Rare", "Special", "Curse", "Status", "Event", "Token"),
    "enum:TargetType": ("None", "Self", "AnyEnemy", "AllEnemies", "RandomEnemy", "AnyPlayer", "AnyAlly", "AllAllies", "Osty"),
    "enum:CardKeyword": ("Exhaust", "Ethereal", "Innate", "Unplayable", "Retain", "Sly", "Eternal"),
    "enum:CardCostColor": ("Normal", "Cheaper", "Expensive", "Unplayable"),
    "enum:PileType": ("None", "Draw", "Hand", "Discard", "Exhaust", "Play", "Deck"),
    "enum:IntentType": ("Attack", "Buff", "Debuff", "StrongDebuff", "Defend", "Escape", "Heal", "Hidden",
                        "Sleep", "Stun", "Summon", "Status", "CardDebuff", "DeathBlow", "Unknown"),
    "enum:PowerType": ("Buff", "Debuff", "None"),
    "enum:PowerStackType": ("Counter", "Single", "None"),
    "enum:PowerInstanceType": ("Single", "Instanced"),
    "enum:RelicStatus": ("Normal", "Active", "Disabled"),
    "enum:EnchantmentStatus": ("Normal", "Disabled"),
    "enum:HpDisplay": ("Normal", "InfiniteWithNumbers", "InfiniteWithoutNumbers"),
    "enum:CombatSide": ("Player", "Enemy"),
    "enum:PlayerTurnPhase": ("None", "Start", "Main", "End"),
    "enum:MapPointType": ("Unassigned", "Unknown", "Shop", "Treasure", "RestSite", "Monster", "Elite", "Boss", "Ancient"),
    "enum:MapPointState": ("None", "Travelable", "Traveled", "Untravelable"),
    "enum:RoomType": ("Unassigned", "Monster", "Elite", "Boss", "Treasure", "Shop", "Event", "RestSite", "Map"),
    "enum:RewardType": ("Gold", "Potion", "Relic", "Card", "SpecialCard", "RemoveCard", "UpgradeCard", "Linked"),
    "enum:GameMode": ("Standard", "Daily", "Custom"),
    "enum:AscensionLevel": tuple(f"A{i}" for i in range(11)),
    "enum:ArrowStatus": ("Retrieved", "Away", "Destroyed"),
    "enum:CrystalSphereItemType": ("CardReward", "Curse", "Gold", "Potion", "Relic"),
    "enum:CrystalSphereToolType": ("Small", "Big"),
}
# Synthetic sizes (non-special entries) for the remaining categories, roughly v0.113 scale.
_SIZES: Mapping[str, int] = {
    "card": 640, "relic": 320, "potion": 70, "power": 400, "monster": 150, "encounter": 120,
    "event": 70, "enchantment": 30, "affliction": 12, "modifier": 24, "card_pool": 12,
    "relic_pool": 10, "potion_pool": 8, "var_name": 170, "event_option_key": 221, "event_page": 160,
    "prompt_key": 55, "slot_name": 32, "enum:CardTag": 12, "enum:UnplayableReason": 10,
    "enum:OrbEvokeType": 4, "enum:CardMultiplayerConstraint": 3, "enum:RelicRarity": 8,
    "enum:PotionRarity": 5, "enum:PotionUsage": 4, "enum:StaticHoverTip": 40,
}


def make_synthetic_vocab(scale: float = 1.0) -> dict[str, Any]:
    """A vocab.json-shaped dict with every schema category; deterministic.

    Categories with known names (characters, acts, protocol-relevant enums) use plausible
    names; the rest get ``<CAT>_0000``-style names sized like v0.113 times ``scale``.
    """
    cats: dict[str, Any] = {}
    for cat in VOCAB_CATEGORIES:
        if cat in _NAMED:
            names = list(_NAMED[cat])
        else:
            n = max(3, int(round(_SIZES.get(cat, 8) * scale)))
            stem = cat.split(":")[-1].upper()
            names = [f"{stem}_{i:04d}" for i in range(n)]
            if cat == "power":  # real names of the powers the debuff telemetry reads (spire.h5eval)
                names[:3] = ["VULNERABLE_POWER", "WEAK_POWER", "ARTIFACT_POWER"]
        cats[cat] = {"entries": [PAD_NAME, UNK_NAME, *names], "retired": []}
    return {"schema": 1, "game_version": "synthetic", "sts2_sha256": "", "vocab_version": 1, "categories": cats}


SYNTHETIC_CARD_TEXT: Final = (
    "Deal {Damage:diff()} damage. Apply {VulnerablePower:diff()} [gold]Vulnerable[/gold].",
    "Apply {WeakPower:diff()} [gold]Weak[/gold].", "Gain {Block:diff()} [gold]Block[/gold].",
    "Deal {Damage:diff()} damage.", "Draw {Cards:diff()} cards.")


def make_synthetic_catalog(vocab: Vocab, seed: int = 0) -> dict[str, Any]:
    """A catalog.json-shaped dict covering every featurised category of ``vocab``."""
    rng = np.random.default_rng(seed)

    def pick(cat: str) -> str:
        return vocab.name(cat, int(rng.integers(2, vocab.size(cat))))

    def subset(cat: str, p: float) -> list[str]:
        return [e for e in vocab[cat].entries[2:] if rng.random() < p]

    out: dict[str, dict[str, Any]] = {}
    for cat, spec in DEFAULT_FEATURES.items():
        ents: dict[str, Any] = {}
        for i in range(2, vocab.size(cat)):
            levels = int(rng.choice([1, 2, 2, 2, 3])) if cat in LEVELED else 1
            recs = []
            for lv in range(levels):
                rec: dict[str, Any] = {}
                for s in spec:
                    if s.kind == "id":
                        rec[s.name] = pick(s.cat)  # type: ignore[arg-type]
                    elif s.kind == "bool":
                        rec[s.name] = bool(rng.random() < 0.3)
                    elif s.kind == "num":
                        rec[s.name] = int(rng.integers(0, 4 if cat == "card" else 300)) - (lv if cat == "card" else 0)
                    else:
                        rec[s.name] = subset(s.cat, 0.15)  # type: ignore[arg-type]
                recs.append(rec)
            if cat == "card":  # printed text (level 0, as in catalog.json); a few apply Vulnerable / Weak
                recs[0]["text"] = SYNTHETIC_CARD_TEXT[i % len(SYNTHETIC_CARD_TEXT)]
            ents[str(i)] = recs if cat in LEVELED else recs[0]
        out[cat] = ents
    return {"schema": 1, "game_version": "synthetic", "vocab_hash": vocab.hash, "categories": out}


def synthetic_vocab_catalog(seed: int = 0, scale: float = 1.0) -> tuple[Vocab, Catalog]:
    vocab = Vocab(make_synthetic_vocab(scale))
    return vocab, Catalog(make_synthetic_catalog(vocab, seed), vocab)


def write_synthetic(out_dir: str | Path, seed: int = 0, scale: float = 1.0) -> tuple[Path, Path]:
    """Write synthetic ``vocab.json`` and ``catalog.json`` into ``out_dir``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    vocab = Vocab(make_synthetic_vocab(scale))
    vp, cp = d / "vocab.json", d / "catalog.json"
    vp.write_text(json.dumps(vocab.raw, ensure_ascii=False, indent=1), encoding="utf-8")
    cp.write_text(json.dumps(make_synthetic_catalog(vocab, seed), ensure_ascii=False, indent=1), encoding="utf-8")
    return vp, cp

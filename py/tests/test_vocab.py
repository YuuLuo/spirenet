from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from spire.proto import parse_frame
from spire.schema import VOCAB_CATEGORIES, Kind, SchemaError
from spire.synth import FrameGen, random_frame
from spire.vocab import (
    Catalog,
    FieldSpec,
    Vocab,
    VocabMismatch,
    canonical_hash,
    check_frame_hashes,
    embedding_rows,
    make_synthetic_catalog,
    make_synthetic_vocab,
    write_synthetic,
)


def test_synthetic_vocab_structure(vocab: Vocab) -> None:
    assert set(VOCAB_CATEGORIES) <= set(vocab.categories)
    for cat in VOCAB_CATEGORIES:
        c = vocab[cat]
        assert c.entries[:2] == ("<PAD>", "<UNK>") and len(c) == vocab.size(cat) >= 3
    assert vocab.size("character") == 8 and vocab.id("character", "IRONCLAD") == 4


def test_lookup_both_ways(vocab: Vocab) -> None:
    for cat in ("card", "enum:IntentType", "rest_option"):
        for i in (0, 1, 2, vocab.size(cat) - 1):
            assert vocab.id(cat, vocab.name(cat, i)) == i
    assert vocab.id("card", "NO_SUCH_CARD") == 1
    assert vocab.ids("character", ["SILENT", "nope", "<PAD>"]).tolist() == [7, 1, 0]
    with pytest.raises(IndexError):
        vocab.name("card", vocab.size("card"))


def test_embedding_rows_formula(vocab: Vocab) -> None:
    assert embedding_rows(0) == 128 and embedding_rows(1) == 192 and embedding_rows(64) == 192
    assert vocab.embedding_rows("card") == -(-(vocab.size("card") + 128) // 64) * 64


def test_canonical_hash_ignores_key_order_and_whitespace() -> None:
    obj = make_synthetic_vocab()
    shuffled = {k: obj[k] for k in reversed(list(obj))}
    shuffled["categories"] = {k: obj["categories"][k] for k in reversed(list(obj["categories"]))}
    assert canonical_hash(obj) == canonical_hash(shuffled) == Vocab(obj).hash
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert canonical_hash(obj) == int.from_bytes(hashlib.sha256(raw).digest()[:4], "little", signed=True)
    other = copy.deepcopy(obj)
    other["categories"]["card"]["retired"].append("CARD_0000")
    assert canonical_hash(other) != canonical_hash(obj)


def test_files_roundtrip(tmp_path, vocab: Vocab) -> None:  # type: ignore[no-untyped-def]
    vp, cp = write_synthetic(tmp_path, seed=1)
    v = Vocab.load(vp)
    c = Catalog.load(cp, v)
    assert v.hash == vocab.hash
    assert c.hash == Catalog(make_synthetic_catalog(vocab, seed=1), vocab).hash


def test_vocab_validation() -> None:
    obj = make_synthetic_vocab()
    bad = copy.deepcopy(obj)
    del bad["categories"]["card"]
    with pytest.raises(SchemaError, match="lacks"):
        Vocab(bad)
    bad = copy.deepcopy(obj)
    bad["categories"]["orb"]["entries"][0] = "X"
    with pytest.raises(SchemaError, match="PAD"):
        Vocab(bad)
    bad = copy.deepcopy(obj)
    bad["categories"]["orb"]["entries"].append("DARK")
    with pytest.raises(SchemaError, match="duplicate"):
        Vocab(bad)


def test_catalog_features(vocab: Vocab, catalog: Catalog) -> None:
    card = catalog.features("card")
    V = vocab.size("card")
    assert card.ids.shape[:2] == card.nums.shape[:2] and card.ids.shape[0] == V
    L = card.ids.shape[1]
    assert L == max(len(e) for e in catalog.entries["card"].values())
    assert (card.ids[:2] == 0).all() and (card.nums[:2] == 0).all()  # PAD / UNK rows
    for j, cat in enumerate(card.id_cats):
        assert card.ids[2:, :, j].min() >= 2 and card.ids[:, :, j].max() < vocab.size(cat)
    # levels beyond a card's max repeat its last level
    i = next(k for k, e in catalog.entries["card"].items() if len(e) < L)
    n = len(catalog.entries["card"][i])
    assert np.array_equal(card.nums[i, n - 1], card.nums[i, L - 1])
    kw = [k for k, name in enumerate(card.num_fields) if name.startswith("keywords:")]
    assert len(kw) == vocab.size("enum:CardKeyword") - 2
    rec = catalog.entry("card", i, level=0)
    assert rec is not None
    for name in rec["keywords"]:
        assert card.nums[i, 0, card.num_fields.index(f"keywords:{name}")] == 1.0
    relic = catalog.features("relic")
    assert relic.ids.ndim == 2 and relic.nums.shape == (vocab.size("relic"), 6)
    custom = catalog.features("monster", [FieldSpec("hp_max", "num")])
    assert custom.nums.shape == (vocab.size("monster"), 1) and custom.ids.shape == (vocab.size("monster"), 0)


def test_catalog_validation(vocab: Vocab) -> None:
    obj = make_synthetic_catalog(vocab)
    obj["categories"]["relic"][str(vocab.size("relic"))] = {}
    with pytest.raises(SchemaError, match="outside"):
        Catalog(obj, vocab)
    obj = make_synthetic_catalog(vocab)
    obj["categories"]["card"]["2"] = {"type": "Attack"}
    with pytest.raises(SchemaError, match="per-level"):
        Catalog(obj, vocab)


def test_frame_hash_check(vocab: Vocab, catalog: Catalog, gen: FrameGen) -> None:
    f = parse_frame(random_frame(gen, np.random.default_rng(0), Kind.MAP))
    check_frame_hashes(f, vocab, catalog)
    other = Vocab(make_synthetic_vocab(scale=0.5))
    with pytest.raises(VocabMismatch, match="vocab"):
        check_frame_hashes(f, other)
    with pytest.raises(VocabMismatch, match="catalog"):
        check_frame_hashes(f, vocab, Catalog(make_synthetic_catalog(vocab, seed=99), vocab))

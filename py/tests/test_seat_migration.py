"""Stage A input audit and explicit, bit-preserving seat symmetry warm start."""
from __future__ import annotations

import json
import runpy
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from spire.learner import Learner, PPOConfig
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.schema import (DECISION_KINDS, GLOBAL_OFFSET, ActionCol, G, IntentCol, Kind,
                          MapNodeCol, OptionCol, Table, ViewMode, WORD)
from spire.seat_migration import (migrate_seat_symmetry_optimizer, migrate_seat_symmetry_weights,
                                  seat_column_groups)
from spire.synth import FrameGen
from spire.tokenize import HISTORY_ROWS, HISTORY_STATS, SPECS, Ent, Mode, Tokenizer, TokenizerConfig
from spire.vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab


TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=8, d_cond=8, d_static=16)


class _LegacySum(SpireNet):
    def character_condition(self, chars: Tensor) -> Tensor:
        return (self.cond_char(chars) * (chars > 0).unsqueeze(-1)).sum(1)


def _equal(a: dict[str, Tensor], b: dict[str, Tensor]) -> None:
    for key in OUTPUT_KEYS:
        assert torch.equal(a[key].view(torch.int32), b[key].view(torch.int32)), key


def test_mean_is_bit_exact_for_one_and_four_identical_characters(vocab: Vocab) -> None:
    net = SpireNet(Tokenizer(vocab).layout, TINY)
    chars = torch.arange(1, net.cond_char.num_embeddings)
    one = torch.stack([chars, torch.zeros_like(chars), torch.zeros_like(chars), torch.zeros_like(chars)], 1)
    with torch.no_grad():
        expected = net.cond_char(chars)
        assert torch.equal(net.character_condition(one).view(torch.int32), expected.view(torch.int32))
        assert torch.equal(net.character_condition(chars[:, None].repeat(1, 4)).view(torch.int32),
                           expected.view(torch.int32))
        assert torch.isfinite(net.character_condition(torch.zeros((1, 4), dtype=torch.long))).all()


@pytest.mark.parametrize("history", [False, True])
def test_sp_outputs_bit_identical_code_and_migration(vocab: Vocab, catalog: Catalog,
                                                    gen: FrameGen, history: bool) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=history))
    old = _LegacySum(tok.layout, TINY, catalog).eval()
    if old.history_proj is not None:
        with torch.no_grad():
            old.history_proj.weight.normal_(0.0, 0.01)
    current = SpireNet(tok.layout, TINY, catalog).eval()
    current.load_state_dict(old.state_dict())
    migrated = SpireNet(tok.layout, replace(TINY, seat_symmetry=True), catalog).eval()
    migrated.load_state_dict(migrate_seat_symmetry_weights(old.state_dict(), migrated, tok.layout))
    rng = np.random.default_rng(74)
    samples = [tok.encode(parse_frame(gen.frame(rng, k, n_seats=1))) for k in DECISION_KINDS]
    for sample in samples:
        # Non-empty public history descriptors from a valid one-seat action; numeric seat stays zero.
        indices = [sum(sample.counts[:3]), 0, 1, 2, 2]
        sample.history[:HISTORY_ROWS] = sample.types[indices]
        sample.history[HISTORY_ROWS:-HISTORY_STATS] = sample.raw[indices].ravel()
        sample.history[-HISTORY_STATS:] = np.arange(HISTORY_STATS)
        sample.history[-HISTORY_STATS + 20] = 0
    batch = tok.collate(samples)
    with torch.no_grad():
        _equal(old(batch), current(batch))
        _equal(old(batch), migrated(batch))


def test_all_rows_columns_and_history_ordinal_are_migrated(vocab: Vocab) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True))
    old = SpireNet(tok.layout, TINY)
    new = SpireNet(tok.layout, replace(TINY, seat_symmetry=True))
    before = {k: v.clone() for k, v in old.state_dict().items()}
    after = migrate_seat_symmetry_weights(before, new, tok.layout, source_view_mode=ViewMode.SEAT)
    for region, source, targets in (("p:seat", 1, (2, 3, 4)), ("p:n_seats", 1, (2, 3, 4)),
                                   ("p:view_mode", 1, (0,))):
        off = tok.layout.regions[region][0]
        for target in targets:
            assert torch.equal(after["ids.weight"][off + target], before["ids.weight"][off + source])
    for n in (2, 3, 4):
        assert torch.equal(after["cond_seats.weight"][n], before["cond_seats.weight"][1])
    for name, cols in seat_column_groups(tok.layout, TINY.d_id).items():
        if name == "intent_missing":
            assert torch.equal(after["in_proj.weight"][:, cols], before["in_proj.weight"][:, cols])
            assert torch.equal(after["multiseat_intent_missing"], before["in_proj.weight"][:, cols[:1]].repeat(1, 4))
        else:
            assert torch.equal(after["in_proj.weight"][:, cols], before["in_proj.weight"][:, cols[:1]].repeat(1, 4))
    assert torch.count_nonzero(after["history_proj.weight"][:, -HISTORY_STATS + 20]) == 0
    assert all(torch.equal(v, old.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize("pooled", [False, True])
def test_mixed_frames_use_correct_missing_projection_and_learn(vocab: Vocab, gen: FrameGen,
                                                              pooled: bool) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(pool_children=pooled))
    old = SpireNet(tok.layout, TINY)
    net = SpireNet(tok.layout, replace(TINY, seat_symmetry=True))
    net.load_state_dict(migrate_seat_symmetry_weights(old.state_dict(), net, tok.layout))
    rng = np.random.default_rng(21)
    samples = [tok.encode(parse_frame(gen.frame(rng, Kind.COMBAT, n_seats=n))) for n in (1, 4, 2, 3)]
    for sample in samples[1:]:
        sample.raw[sample.types == Ent.INTENT, tok.layout.slot(Ent.INTENT, IntentCol.DMG_SEAT1)] = -1
    batch = tok.collate(samples)
    ids, wide = net.decoder(batch["types"], batch["raw"])
    features = torch.cat([torch.zeros((len(ids), TINY.d_id)), wide], 1)
    owners = (batch["rel"][:, 0] // batch["tok_mask"].shape[1]).clamp(0, len(samples) - 1)
    multi = batch["cond"][owners, 1] > 1
    got = net._project_entities(features, batch["types"], multi)
    cols = seat_column_groups(tok.layout, TINY.d_id)["intent_missing"]
    assert net.multiseat_intent_missing is not None
    reference = net.in_proj.weight.clone()
    reference[:, cols] = net.multiseat_intent_missing
    selected = (batch["types"] == Ent.INTENT) & multi
    assert selected.any()
    # Independent sample boundaries test the owner-pointer routing, rather than merely
    # repeating its formula in both actual and expected computations.
    row0 = 0
    for sample in samples:
        rows = sum(sample.counts)
        expected_mask = torch.from_numpy(sample.types == Ent.INTENT) & bool(sample.cond[1] > 1)
        assert torch.equal(selected[row0:row0 + rows], expected_mask)
        row0 += rows
    expected = torch.where(selected[:, None], torch.nn.functional.linear(features, reference, net.in_proj.bias),
                           net.in_proj(features))
    assert torch.equal(got, expected)
    out = net(batch)
    assert all(torch.isfinite(out[k]).all() for k in OUTPUT_KEYS if k != "logits")
    out["value"].sum().backward()
    grad = net.multiseat_intent_missing.grad
    assert grad is not None and torch.count_nonzero(grad) > 0


def _warm_adam(learner: Learner, *, amsgrad: bool = False) -> None:
    for group in learner.opt.param_groups:
        group["amsgrad"] = amsgrad
    for p in learner.model.parameters():
        p.grad = torch.ones_like(p)
    learner.opt.step()
    for i, p in enumerate(learner.model.parameters()):
        state = learner.opt.state[p]
        state["step"].fill_(100 + i)
        values = torch.arange(p.numel(), dtype=p.dtype).reshape_as(p) / 1000 + i + 1
        state["exp_avg"].copy_(values / 10)
        state["exp_avg_sq"].copy_(values)
        if amsgrad:
            state["max_exp_avg_sq"].copy_(values + 1)


def _named_adam(learner: Learner) -> dict[str, dict[str, Tensor]]:
    return {name: {k: value.clone() for k, value in learner.opt.state[p].items()}
            for name, p in learner.model.named_parameters() if p in learner.opt.state}


def _assert_adam_copies(before: dict[str, dict[str, Tensor]], learner: Learner,
                        tok: Tokenizer, *, source_view: int = ViewMode.TEAM) -> None:
    after = _named_adam(learner)
    assert set(after) == set(before) | {"multiseat_intent_missing"}
    for name, entry in before.items():
        assert set(after[name]) == set(entry)
        for key, original in entry.items():
            expected = original.clone()
            if key != "step":
                if name == "ids.weight":
                    for region, source, targets in (("p:seat", 1, (2, 3, 4)), ("p:n_seats", 1, (2, 3, 4)),
                                                   ("p:view_mode", source_view, (1 - source_view,))):
                        off = tok.layout.regions[region][0]
                        for target in targets:
                            expected[off + target] = original[off + source]
                elif name == "cond_seats.weight":
                    expected[2:5] = original[1]
                elif name == "in_proj.weight":
                    for group, cols in seat_column_groups(tok.layout, TINY.d_id).items():
                        if group != "intent_missing":
                            expected[:, cols[1:]] = original[:, cols[:1]]
            assert torch.equal(after[name][key].reshape(-1).view(torch.uint8),
                               expected.reshape(-1).view(torch.uint8)), (name, key)
    missing = seat_column_groups(tok.layout, TINY.d_id)["intent_missing"]
    for key, original in before["in_proj.weight"].items():
        expected = original if key == "step" else original[:, missing[:1]].repeat(1, 4)
        assert torch.equal(after["multiseat_intent_missing"][key].reshape(-1).view(torch.uint8),
                           expected.reshape(-1).view(torch.uint8)), key


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("amsgrad", [False, True])
def test_migration_preserves_every_adam_state_and_strict_resume(vocab: Vocab, tmp_path: Path,
                                                              history: bool, amsgrad: bool) -> None:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=history))
    old = Learner(SpireNet(tok.layout, TINY), tok, PPOConfig(), device="cpu", fast=False)
    _warm_adam(old, amsgrad=amsgrad)
    before = _named_adam(old)
    old.version, old.updates, old.samples_seen = 7, 8, 9
    source, target = tmp_path / "old.pt", tmp_path / "new.pt"
    old.save(source)
    new = Learner(SpireNet(tok.layout, replace(TINY, seat_symmetry=True)), tok, PPOConfig(), device="cpu", fast=False)
    with pytest.raises(RuntimeError, match="multiseat_intent_missing"):
        new.load(source)
    new.load(source, migrate_seat_symmetry=True)
    _assert_adam_copies(before, new, tok)
    source_after = _named_adam(old)
    assert all(torch.equal(value, source_after[name][key])
               for name, entry in before.items() for key, value in entry.items())
    for old_group, new_group in zip(old.opt.param_groups, new.opt.param_groups):
        assert {k: v for k, v in old_group.items() if k != "params"} == \
               {k: v for k, v in new_group.items() if k != "params"}
    assert (new.version, new.updates, new.samples_seen) == (7, 8, 9)
    new.save(target)
    resumed = Learner(SpireNet(tok.layout, replace(TINY, seat_symmetry=True)), tok, PPOConfig(),
                      device="cpu", fast=False)
    resumed.load(target)
    _assert_adam_copies(before, resumed, tok)
    for p in resumed.model.parameters():
        p.grad = torch.ones_like(p)
    resumed.opt.step()  # PyTorch also consumes every migrated shape on the ordinary resume path.
    repeated = Learner(SpireNet(tok.layout, replace(TINY, seat_symmetry=True)), tok, PPOConfig(),
                       device="cpu", fast=False)
    with pytest.raises(AssertionError, match="exactly the pre-seat-symmetry"):
        repeated.load(target, migrate_seat_symmetry=True)
    assert isinstance(new.model, SpireNet)
    bad = dict(old.model.state_dict())
    bad["ids.weight"] = torch.zeros(1)
    with pytest.raises(AssertionError, match="cannot change existing shapes"):
        migrate_seat_symmetry_weights(bad, new.model, tok.layout)


def test_full_reset_is_explicit_and_keeps_saved_group_settings(vocab: Vocab, tmp_path: Path) -> None:
    tok = Tokenizer(vocab)
    old = Learner(SpireNet(tok.layout, TINY), tok, PPOConfig(lr=0.002), device="cpu", fast=False)
    _warm_adam(old)
    source = tmp_path / "old.pt"
    old.save(source)
    new = Learner(SpireNet(tok.layout, replace(TINY, seat_symmetry=True)), tok, PPOConfig(), device="cpu", fast=False)
    with pytest.raises(AssertionError, match="optimizer reset requires"):
        new.load(source, allow_config_change=True, reset_seat_symmetry_optimizer=True)
    new.load(source, allow_config_change=True, migrate_seat_symmetry=True, reset_seat_symmetry_optimizer=True)
    assert not new.opt.state
    for group in new.opt.param_groups:
        assert group["lr"] == 0.002


@pytest.mark.parametrize("corruption", ["shape", "unknown_id", "group_size", "step"])
def test_optimizer_migration_rejects_invalid_checkpoint_state(vocab: Vocab, corruption: str) -> None:
    tok = Tokenizer(vocab)
    old = Learner(SpireNet(tok.layout, TINY), tok, PPOConfig(), device="cpu", fast=False)
    _warm_adam(old)
    new = Learner(SpireNet(tok.layout, replace(TINY, seat_symmetry=True)), tok, PPOConfig(), device="cpu", fast=False)
    saved = old.opt.state_dict()
    first = next(iter(saved["state"]))
    if corruption == "shape":
        saved["state"][first]["exp_avg"] = torch.zeros(1)
        error = "invalid Adam moment shape"
    elif corruption == "unknown_id":
        saved["state"][-1] = saved["state"][first]
        error = "unknown parameter"
    elif corruption == "group_size":
        saved["param_groups"][0]["params"].pop()
        error = "cannot change existing optimizer groups"
    else:
        saved["state"][first]["step"] = torch.zeros(2)
        error = "invalid Adam step"
    assert isinstance(new.model, SpireNet)
    with pytest.raises(AssertionError, match=error):
        migrate_seat_symmetry_optimizer(saved, new.model, tok.layout, new.opt)


@pytest.mark.parametrize("reset", [False, True])
def test_standalone_migration_preserves_histories_and_can_reset(vocab: Vocab, catalog: Catalog, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch, reset: bool) -> None:
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "migrate_seat_symmetry.py"))
    tok = Tokenizer(vocab, TokenizerConfig(public_history=True))
    old = Learner(SpireNet(tok.layout, TINY, catalog), tok, PPOConfig(), device="cpu", fast=False)
    _warm_adam(old)
    before = _named_adam(old)
    source, target = tmp_path / "old.pt", tmp_path / "new.pt"
    extra: dict[str, Any] = {"vocab_hash": vocab.hash, "catalog_hash": catalog.hash,
                             "tokenizer": asdict(tok.config), "model_config": asdict(TINY)}
    old.save(source, extra=extra)
    vocab_path, catalog_path = tmp_path / "vocab.json", tmp_path / "catalog.json"
    vocab_path.write_text(json.dumps(make_synthetic_vocab()), encoding="utf-8")
    catalog_path.write_text(json.dumps(make_synthetic_catalog(vocab, seed=1)), encoding="utf-8")
    argv = ["migrate_seat_symmetry.py", "--checkpoint", str(source), "--output", str(target),
            "--vocab", str(vocab_path), "--catalog", str(catalog_path), "--source-view-mode", "seat"]
    if reset:
        argv.append("--reset-optimizer")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert script["main"]() == 0
    resumed = Learner(SpireNet(tok.layout, replace(TINY, seat_symmetry=True), catalog), tok, PPOConfig(),
                      device="cpu", fast=False)
    info = resumed.load(target)
    assert info["model_config"]["seat_symmetry"]
    if reset:
        assert not resumed.opt.state
    else:
        _assert_adam_copies(before, resumed, tok, source_view=ViewMode.SEAT)
    for p in resumed.model.parameters():
        p.grad = torch.ones_like(p)
    resumed.opt.step()


def test_stage_a_reserved_globals_and_option_votes_are_ignored(vocab: Vocab, gen: FrameGen) -> None:
    tok = Tokenizer(vocab)
    net = SpireNet(tok.layout, TINY).eval()
    rng = np.random.default_rng(24)
    frame = parse_frame(gen.frame(rng, Kind.TREASURE, n_seats=4))
    assert frame.rows[Table.OPTIONS] > 0
    words = np.frombuffer(frame.payload, dtype="<i4").copy()
    words[GLOBAL_OFFSET // WORD + 52:GLOBAL_OFFSET // WORD + 60] = np.arange(1, 9)
    off = frame.offsets[Table.OPTIONS]
    options = frame.table(Table.OPTIONS)
    for row in range(len(options)):
        words[off + row * options.shape[1] + OptionCol.FLAGS] |= 15 << 7
    changed = parse_frame(words.tobytes())
    a, b = tok.collate([tok.encode(frame)]), tok.collate([tok.encode(changed)])
    for x, y in zip(net.decoder(a["types"], a["raw"]), net.decoder(b["types"], b["raw"])):
        assert torch.equal(x, y)
    with torch.no_grad():
        _equal(net(a), net(b))


def test_stage_a_map_votes_visible_and_mend_aux_uses_crystal_embedding(vocab: Vocab, gen: FrameGen) -> None:
    tok = Tokenizer(vocab)
    decoder = SpireNet(tok.layout, TINY).decoder
    rng = np.random.default_rng(25)
    sample = tok.encode(parse_frame(gen.frame(rng, Kind.MAP, n_seats=4)))
    row = int(np.flatnonzero(sample.types == Ent.MAP_NODE)[0])
    sample.raw[row, tok.layout.slot(Ent.MAP_NODE, MapNodeCol.FLAGS)] = 15 << 4
    batch = tok.collate([sample])
    _, wide = decoder(batch["types"], batch["raw"])
    cols = seat_column_groups(tok.layout, TINY.d_id)["map_votes"]
    assert torch.equal(wide[row, [c - TINY.d_id for c in cols]], torch.ones(4))
    action_slot = tok.layout.slot(Ent.ACTION, ActionCol.AUX)
    raw = torch.zeros((4, tok.width), dtype=torch.int16)
    raw[:, action_slot] = torch.arange(4, dtype=torch.int16)
    ids, _ = decoder(torch.full((4,), int(Ent.ACTION)), raw)
    aux = SPECS[Ent.ACTION].ids[-1]
    assert aux.src == ActionCol.AUX and aux.table == "enum:CrystalSphereToolType"
    assert ids[:, 2].tolist() == list(range(tok.layout.regions[aux.table][0], tok.layout.regions[aux.table][0] + 4))
    # No MEND target relation is constructed: aux never participates in ACTION relation pointers.
    assert len(SPECS[Ent.ACTION].nums) == 2

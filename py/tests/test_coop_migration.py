"""Stage B exact warm start, learnability and strict online/EMA/Adam migration."""
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

from spire.coop_migration import migrate_coop_features_optimizer, migrate_coop_features_weights
from spire.history import PublicHistory
from spire.learner import Learner, PPOConfig
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet
from spire.proto import parse_frame
from spire.schema import DECISION_KINDS, G, Kind, OptionKind, Table, Verb
from spire.synth import FrameBuilder, FrameGen
from spire.tokenize import Ent, Tokenizer, TokenizerConfig
from spire.vocab import Catalog, Vocab


TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=8, d_cond=8, d_static=16)


def _tokens(vocab: Vocab, history: bool = False, *, mend: bool = True, coop: bool = True) -> tuple[Tokenizer, Tokenizer]:
    old = Tokenizer(vocab, TokenizerConfig(public_history=history), native=False)
    new = Tokenizer(vocab, replace(old.config, coop_features=coop, mend_target=mend), native=False)
    return old, new


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("symmetry", [False, True])
@pytest.mark.parametrize("seats", [1, 4])
def test_outputs_bit_identical_at_migration(vocab: Vocab, catalog: Catalog, gen: FrameGen,
                                           history: bool, symmetry: bool, seats: int) -> None:
    old_tok, new_tok = _tokens(vocab, history)
    config = replace(TINY, seat_symmetry=symmetry)
    old = SpireNet(old_tok.layout, config, catalog).eval()
    new = SpireNet(new_tok.layout, config, catalog).eval()
    new.load_state_dict(migrate_coop_features_weights(old.state_dict(), new, new_tok.layout))
    rng = np.random.default_rng(741)
    for kind in DECISION_KINDS:
        frame = parse_frame(gen.frame(rng, kind, n_seats=seats))
        before, after = old_tok.encode(frame), new_tok.encode(frame)
        with torch.no_grad():
            outputs = old(old_tok.collate([before])), new(new_tok.collate([after]))
        for key in OUTPUT_KEYS:
            assert torch.equal(outputs[0][key].view(torch.int32), outputs[1][key].view(torch.int32)), (kind, key)


def _warm(learner: Learner, amsgrad: bool) -> dict[str, dict[str, Tensor]]:
    for group in learner.opt.param_groups:
        group["amsgrad"] = amsgrad
    for parameter in learner.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    learner.opt.step()
    return {name: {key: value.clone() for key, value in learner.opt.state[parameter].items()}
            for name, parameter in learner.model.named_parameters()}


def _adam_exact(before: dict[str, dict[str, Tensor]], learner: Learner, old_width: int) -> None:
    for name, parameter in learner.model.named_parameters():
        state = learner.opt.state[parameter]
        if name in {"coop_screen.weight", "mend_target_gate"}:
            assert all(torch.count_nonzero(value) == 0 for value in state.values()), name
        else:
            assert set(state) == set(before[name])
            for key, value in state.items():
                if name == "in_proj.weight" and key != "step":
                    assert torch.equal(value[:, :old_width], before[name][key]), key
                    assert torch.count_nonzero(value[:, old_width:]) == 0, key
                else:
                    assert torch.equal(value, before[name][key]), (name, key)


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("mend,coop", [(False, True), (True, False), (True, True)])
def test_strict_checkpoint_and_adam_preservation(vocab: Vocab, tmp_path: Path,
                                                amsgrad: bool, mend: bool, coop: bool) -> None:
    old_tok, new_tok = _tokens(vocab, mend=mend, coop=coop)
    old = Learner(SpireNet(old_tok.layout, TINY), old_tok, PPOConfig(), device="cpu", fast=False)
    before = _warm(old, amsgrad)
    source, target = tmp_path / "before.pt", tmp_path / "after.pt"
    old.version, old.updates, old.samples_seen = 11, 12, 13
    old.save(source)
    new = Learner(SpireNet(new_tok.layout, TINY), new_tok, PPOConfig(), device="cpu", fast=False)
    with pytest.raises(RuntimeError):
        new.load(source)
    new.load(source, migrate_coop_features=True)
    assert isinstance(new.model, SpireNet)
    _adam_exact(before, new, TINY.d_id + old_tok.layout.n_wide)
    for name in ("model", "ema"):
        expected = migrate_coop_features_weights(getattr(old, name).state_dict(), new.model, new_tok.layout)
        assert all(torch.equal(value, expected[key]) for key, value in getattr(new, name).state_dict().items())
    assert (new.version, new.updates, new.samples_seen) == (11, 12, 13)
    new.save(target)
    resumed = Learner(SpireNet(new_tok.layout, TINY), new_tok, PPOConfig(), device="cpu", fast=False)
    resumed.load(target)
    _adam_exact(before, resumed, TINY.d_id + old_tok.layout.n_wide)
    for parameter in resumed.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    resumed.opt.step()
    with pytest.raises(AssertionError, match="exactly the pre-coop-features"):
        new.load(target, migrate_coop_features=True)


def test_added_features_and_mend_gate_are_learnable(vocab: Vocab, gen: FrameGen) -> None:
    old_tok, tok = _tokens(vocab)
    old = SpireNet(old_tok.layout, TINY).eval()
    net = SpireNet(tok.layout, TINY).eval()
    net.load_state_dict(migrate_coop_features_weights(old.state_dict(), net, tok.layout))
    sample = tok.encode(parse_frame(gen.frame(np.random.default_rng(15), Kind.REST, n_seats=4)))
    seats = sample.types == Ent.SEAT
    options = sample.types == Ent.OPTION
    sample.raw[seats, tok.layout.derived_slot[Ent.SEAT]["acts"]] = (Kind.REST << 1) | (63 << 5) | 1
    sample.raw[options, tok.layout.derived_slot[Ent.OPTION]["vote_count"]] = 3
    # A target-SEATS action is exactly the model-side MEND gate selector.
    action = sum(sample.counts[:3])
    sample.rel[action, 1] = 3
    out = net(tok.collate([sample]))
    (out["logits"][0, :sample.n_actions].sum() + out["value"].sum()).backward()
    assert net.mend_target_gate is not None and net.mend_target_gate.grad is not None
    assert net.mend_target_gate.grad.abs() > 0
    assert net.coop_screen is not None and net.coop_screen.weight.grad is not None
    assert net.coop_screen.weight.grad[Kind.REST].abs().sum() > 0
    assert net.in_proj.weight.grad is not None
    assert (net.in_proj.weight.grad[:, TINY.d_id + tok.layout.legacy_n_wide:].abs().sum(0) > 0).all()


def test_gate_changes_only_target_seat_action_rows(vocab: Vocab, gen: FrameGen) -> None:
    _, tok = _tokens(vocab)
    net = SpireNet(tok.layout, TINY).eval()
    rng = np.random.default_rng(154)
    samples = [tok.encode(parse_frame(gen.frame(rng, Kind.REST, n_seats=seats))) for seats in (1, 4)]
    for sample in samples:
        sample.rel[sum(sample.counts[:3]):, 1] = -1
    samples[1].rel[sum(samples[1].counts[:3]), 1] = 3
    batch = tok.collate(samples)
    with torch.no_grad():
        before = net(batch)
        assert net.mend_target_gate is not None
        net.mend_target_gate.fill_(1)
        after = net(batch)
    for key in OUTPUT_KEYS:
        if key == "logits":
            assert before[key][1, 0] != after[key][1, 0]
            after[key][1, 0] = before[key][1, 0]
        assert torch.equal(before[key].view(torch.int32), after[key].view(torch.int32)), key


def test_recorded_mend_target_history_shares_zero_gate(vocab: Vocab) -> None:
    old_tok, tok = _tokens(vocab, history=True)
    old = SpireNet(old_tok.layout, TINY).eval()
    assert old.history_proj is not None
    with torch.no_grad():
        old.history_proj.weight.normal_(0, 0.03)
    net = SpireNet(tok.layout, TINY).eval()
    net.load_state_dict(migrate_coop_features_weights(old.state_dict(), net, tok.layout))
    builder = FrameBuilder()
    builder.glob[G.KIND], builder.glob[G.N_SEATS] = Kind.REST, 4
    for seat in range(4):
        builder.add(Table.SEATS, seat=seat, hp=80 - 10*seat, max_hp=80, alive=1, creature_row=-1,
                    character_id=vocab.id("character", "IRONCLAD"))
    for row, name in enumerate(("MEND", "HEAL")):
        builder.add(Table.OPTIONS, kind=OptionKind.REST_OPTION, seat=0, pos=row,
                    key_id=vocab.id("rest_option", name))
        builder.add(Table.ACTIONS, verb=Verb.CHOOSE_OPTION, seat=0, src_table=Table.OPTIONS, src_row=row,
                    aux=1 if row == 0 else 0)
    first = parse_frame(builder.build(hashes=False))
    builder.step_id = 1
    second = parse_frame(builder.build(hashes=False))
    histories = PublicHistory(old_tok.width), PublicHistory(tok.width)
    for history, tokenizer in zip(histories, (old_tok, tok)):
        history.observe(first)
        history.record(first, tokenizer.encode(first), 0)
    assert histories[0].types[3] == Ent.PAD and histories[1].types[3] == Ent.SEAT
    samples = old_tok.encode(second), tok.encode(second)
    for sample, history in zip(samples, histories):
        sample.history = history.observe(second)
    before = old(old_tok.collate([samples[0]]))
    after = net(tok.collate([samples[1]]))
    for key in OUTPUT_KEYS:
        assert torch.equal(before[key].view(torch.int32), after[key].view(torch.int32)), key
    after["value"].sum().backward()  # Value has no live action target path: gradient must come through history.
    assert net.mend_target_gate is not None and net.mend_target_gate.grad is not None
    assert net.mend_target_gate.grad.abs() > 0


@pytest.mark.parametrize("corruption", ["shape", "unknown_id", "group_size", "step"])
def test_bad_adam_state_fails(vocab: Vocab, corruption: str) -> None:
    old_tok, new_tok = _tokens(vocab)
    old = Learner(SpireNet(old_tok.layout, TINY), old_tok, PPOConfig(), device="cpu", fast=False)
    _warm(old, False)
    new = Learner(SpireNet(new_tok.layout, TINY), new_tok, PPOConfig(), device="cpu", fast=False)
    saved = old.opt.state_dict()
    first = next(iter(saved["state"]))
    if corruption == "shape":
        saved["state"][first]["exp_avg"] = torch.zeros(1)
    elif corruption == "unknown_id":
        saved["state"][-1] = saved["state"][first]
    elif corruption == "group_size":
        saved["param_groups"][0]["params"].pop()
    else:
        saved["state"][first]["step"] = torch.zeros(2)
    with pytest.raises(AssertionError):
        assert isinstance(new.model, SpireNet)
        migrate_coop_features_optimizer(saved, new.model, new_tok.layout, new.opt)


def test_standalone_migration(vocab: Vocab, catalog: Catalog, tmp_path: Path,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "migrate_coop_features.py"))
    old_tok, new_tok = _tokens(vocab)
    old = Learner(SpireNet(old_tok.layout, TINY, catalog), old_tok, PPOConfig(), device="cpu", fast=False)
    before = _warm(old, True)
    source, target = tmp_path / "before.pt", tmp_path / "after.pt"
    vp, cp = tmp_path / "vocab.json", tmp_path / "catalog.json"
    vp.write_text(json.dumps(vocab.raw), encoding="utf-8")
    cp.write_text(json.dumps(catalog.raw), encoding="utf-8")
    extra: dict[str, Any] = {"tokenizer": asdict(old_tok.config), "model_config": asdict(TINY),
                             "vocab_hash": vocab.hash, "catalog_hash": catalog.hash}
    old.save(source, extra=extra)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(sys, "argv", ["migration", "--checkpoint", str(source), "--output", str(target),
                                      "--vocab", str(vp), "--catalog", str(cp)])
    assert script["main"]() == 0
    new = Learner(SpireNet(new_tok.layout, TINY, catalog), new_tok, PPOConfig(), device="cpu", fast=False)
    result = new.load(target)
    assert result["tokenizer"]["coop_features"] and result["tokenizer"]["mend_target"]
    _adam_exact(before, new, TINY.d_id + old_tok.layout.n_wide)
    with pytest.raises(AssertionError, match="output already exists"):
        script["main"]()


def test_cli_requires_resume_and_enabled_architecture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from spire import train
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"run_name": "migration", "tokenizer": {"coop_features": True,
                                                                       "mend_target": True}}), encoding="utf-8")
    seen: list[train.TrainConfig] = []

    class Supervisor:
        def __init__(self, cfg: train.TrainConfig, *, resume: bool) -> None:
            assert resume
            seen.append(cfg)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(train, "Supervisor", Supervisor)
    assert train.main(["--config", str(path), "--resume", "--migrate-coop-features"]) == 0
    assert seen[0].migrate_coop_features
    with pytest.raises(SystemExit, match="2"):
        train.main(["--config", str(path), "--migrate-coop-features"])
    path.write_text(json.dumps({"run_name": "migration"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        train.main(["--config", str(path), "--resume", "--migrate-coop-features"])


def test_one_time_flag_cleared_in_persisted_resume_config(tmp_path: Path) -> None:
    from spire import train
    cfg = train.TrainConfig(run_root=str(tmp_path), run_name="migration", migrate_coop_features=True,
                            tokenizer=TokenizerConfig(coop_features=True, mend_target=True))
    supervisor = train.Supervisor(cfg, resume=True)
    for writer in (supervisor.metrics, supervisor.agg.episodes, supervisor.agg.errors):
        writer.close()
    saved = train.load_config(cfg.run_dir / "config.json")
    assert not saved.migrate_coop_features
    assert saved.tokenizer.coop_features and saved.tokenizer.mend_target

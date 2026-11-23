"""Public-history migration is explicit, output-preserving, and retains old Adam moments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from spire.learner import Learner, PPOConfig
from spire.model import OUTPUT_KEYS, ModelConfig, SpireNet, migrate_public_history_weights
from spire.proto import parse_frame
from spire.schema import DECISION_KINDS
from spire.synth import FrameGen, random_frame
from spire.tokenize import HISTORY_ROWS, HISTORY_STATS, Batch, Tokenizer, TokenizerConfig
from spire.vocab import Catalog, Vocab
from spire import train


TINY = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, d_id=8, d_cond=8, d_static=16)


def _learner(vocab: Vocab, catalog: Catalog, history: bool) -> Learner:
    tok = Tokenizer(vocab, TokenizerConfig(public_history=history))
    return Learner(SpireNet(tok.layout, TINY, catalog), tok, PPOConfig(), device="cpu", fast=False)


def _batch(tok: Tokenizer, gen: FrameGen) -> Batch:
    rng = np.random.default_rng(39)
    samples = [tok.encode(parse_frame(random_frame(gen, rng, k))) for k in DECISION_KINDS]
    batch = tok.collate(samples)
    history = batch["history"]
    b = len(samples)
    width = tok.layout.width
    # Valid, distinct public descriptors, and nonzero room counters for every decision kind.
    history[:, :HISTORY_ROWS] = batch["types"][:HISTORY_ROWS]
    history[:, HISTORY_ROWS:-HISTORY_STATS] = batch["raw"][:HISTORY_ROWS].reshape(1, -1)
    history[:, -HISTORY_STATS:] = torch.arange(HISTORY_STATS, dtype=torch.int16)
    history[:b, -HISTORY_STATS] = torch.arange(1, b + 1, dtype=torch.int16)
    assert history.shape[1] == HISTORY_ROWS * (1 + width) + HISTORY_STATS
    return batch


def _optimizer_by_name(learner: Learner) -> dict[str, dict[str, Any]]:
    return {name: learner.opt.state[p] for name, p in learner.model.named_parameters() if p in learner.opt.state}


def _assert_state_equal(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], torch.Tensor):
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


def test_history_off_ignores_history_and_keeps_legacy_parameter_set(vocab: Vocab, catalog: Catalog,
                                                                 gen: FrameGen) -> None:
    tok = Tokenizer(vocab)
    torch.manual_seed(31)
    default = SpireNet(tok.layout, TINY, catalog).eval()
    torch.manual_seed(31)
    explicit = SpireNet(Tokenizer(vocab, TokenizerConfig(public_history=False)).layout, TINY, catalog).eval()
    assert default.history_proj is explicit.history_proj is None
    _assert_state_equal(default.state_dict(), explicit.state_dict())
    assert "history_proj.weight" not in default.state_dict()
    batch = _batch(tok, gen)
    legacy = {key: value for key, value in batch.items() if key != "history"}
    with torch.no_grad():
        before, after = default(legacy), explicit(batch)
    for key in OUTPUT_KEYS:
        assert torch.equal(before[key].view(torch.int32), after[key].view(torch.int32)), key


def test_explicit_migration_preserves_outputs_adam_and_round_trip(vocab: Vocab, catalog: Catalog,
                                                                gen: FrameGen, tmp_path: Path) -> None:
    old = _learner(vocab, catalog, False)
    for i, p in enumerate(old.model.parameters()):
        p.grad = torch.full_like(p, (i + 1) * 0.001)
    old.opt.step()
    old.opt.zero_grad(set_to_none=True)
    old._ema_step()
    old.version, old.updates, old.samples_seen = 17, 9, 1234
    path = tmp_path / "before.pt"
    old.save(path)
    new = _learner(vocab, catalog, True)
    with pytest.raises(RuntimeError, match="history_proj.weight"):
        new.load(path)
    new.load(path, migrate_public_history=True)
    assert (new.version, new.updates, new.samples_seen, new.ema_steps) == (17, 9, 1234, old.ema_steps)
    assert isinstance(new.model, SpireNet) and new.model.history_proj is not None
    assert torch.count_nonzero(new.model.history_proj.weight) == 0
    assert new.model.history_proj.weight not in new.opt.state
    old_states, new_states = _optimizer_by_name(old), _optimizer_by_name(new)
    assert set(new_states) == set(old_states)
    for key in old_states:
        _assert_state_equal(old_states[key], new_states[key])
    for a, b in zip(old.opt.param_groups, new.opt.param_groups):
        assert {k: v for k, v in a.items() if k != "params"} == {k: v for k, v in b.items() if k != "params"}

    assert isinstance(new.tokenizer, Tokenizer)
    batch = _batch(new.tokenizer, gen)
    with torch.no_grad():
        for before, after in ((old.model, new.model), (old.ema, new.ema)):
            a, b = before.eval()(batch), after.eval()(batch)
            for key in OUTPUT_KEYS:
                assert torch.equal(a[key].view(torch.int32), b[key].view(torch.int32)), key

    migrated = tmp_path / "after.pt"
    new.save(migrated)
    restored = _learner(vocab, catalog, True)
    restored.load(migrated)
    _assert_state_equal(new.model.state_dict(), restored.model.state_dict())
    with pytest.raises(AssertionError, match="exactly the pre-history"):
        restored.load(migrated, migrate_public_history=True)

    # Zero initialization must be learnable on the very first backward pass.
    out = new.model(batch)
    loss = out["value"].sum() + out["logits"][torch.isfinite(out["logits"])].sum()
    loss.backward()
    grad = new.model.history_proj.weight.grad
    assert grad is not None and torch.count_nonzero(grad) > 0
    new.opt.step()
    assert new.opt.state[new.model.history_proj.weight]["step"] == 1
    assert torch.count_nonzero(new.model.history_proj.weight) > 0


def test_migration_rejects_other_missing_extra_and_changed_shapes(vocab: Vocab, catalog: Catalog) -> None:
    old = _learner(vocab, catalog, False)
    new = _learner(vocab, catalog, True)
    assert isinstance(old.model, SpireNet) and isinstance(new.model, SpireNet)
    state = old.model.state_dict()
    name = next(iter(state))
    with pytest.raises(AssertionError, match="exactly the pre-history"):
        migrate_public_history_weights({k: v for k, v in state.items() if k != name}, new.model)
    with pytest.raises(AssertionError, match="exactly the pre-history"):
        migrate_public_history_weights({**state, "unexpected": torch.zeros(1)}, new.model)
    with pytest.raises(AssertionError, match="cannot change existing shapes"):
        migrate_public_history_weights({**state, name: torch.zeros(1)}, new.model)
    with pytest.raises(AssertionError, match="public_history=true"):
        migrate_public_history_weights(state, old.model)


def test_migration_cli_requires_resume_and_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"run_name": "migration", "tokenizer": {"public_history": True}}), encoding="utf-8")
    got: list[train.TrainConfig] = []

    class Supervisor:
        def __init__(self, cfg: train.TrainConfig, *, resume: bool) -> None:
            assert resume
            got.append(cfg)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(train, "Supervisor", Supervisor)
    assert train.main(["--config", str(path), "--resume", "--migrate-public-history"]) == 0
    assert got[0].migrate_public_history
    with pytest.raises(SystemExit, match="2"):
        train.main(["--config", str(path), "--migrate-public-history"])
    path.write_text(json.dumps({"run_name": "migration"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        train.main(["--config", str(path), "--resume", "--migrate-public-history"])

"""Stage A launch configuration and the explicit one-time training migration switch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spire import train
from spire.metrics import canonical_lineup


def test_migration_cli_requires_resume_and_architecture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"run_name": "migration", "model": {"seat_symmetry": True}}), encoding="utf-8")
    got: list[train.TrainConfig] = []

    class Supervisor:
        def __init__(self, cfg: train.TrainConfig, *, resume: bool) -> None:
            assert resume
            got.append(cfg)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(train, "Supervisor", Supervisor)
    assert train.main(["--config", str(path), "--resume", "--migrate-seat-symmetry"]) == 0
    assert got[0].migrate_seat_symmetry
    assert not got[0].reset_seat_symmetry_optimizer
    assert train.main(["--config", str(path), "--resume", "--migrate-seat-symmetry",
                       "--reset-seat-symmetry-optimizer"]) == 0
    assert got[1].migrate_seat_symmetry and got[1].reset_seat_symmetry_optimizer
    with pytest.raises(SystemExit, match="2"):
        train.main(["--config", str(path), "--resume", "--reset-seat-symmetry-optimizer"])
    with pytest.raises(SystemExit, match="2"):
        train.main(["--config", str(path), "--migrate-seat-symmetry"])
    path.write_text(json.dumps({"run_name": "migration"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        train.main(["--config", str(path), "--resume", "--migrate-seat-symmetry"])


def test_migration_flags_are_not_persisted_for_ordinary_resume(tmp_path: Path) -> None:
    cfg = train.TrainConfig(run_root=str(tmp_path), run_name="migration",
                            model=train.ModelSection(seat_symmetry=True),
                            migrate_seat_symmetry=True, reset_seat_symmetry_optimizer=True)
    supervisor = train.Supervisor(cfg, resume=True)
    for writer in (supervisor.metrics, supervisor.agg.episodes, supervisor.agg.errors):
        writer.close()
    saved = train.load_config(cfg.run_dir / "config.json")
    assert not saved.migrate_seat_symmetry and not saved.reset_seat_symmetry_optimizer
    assert saved.model.config().seat_symmetry


def test_config_canonicalizes_panel_and_sampling_names() -> None:
    cfg = train.build(train.TrainConfig, {
        "run_name": "canonical", "episodes": {"lineups": [["IRONCLAD+DEFECT+IRONCLAD+DEFECT", 1.0]]},
        "eval": {"lineups": ["IRONCLAD+DEFECT+IRONCLAD+DEFECT"]},
    })
    key = "DEFECT+DEFECT+IRONCLAD+IRONCLAD"
    assert train.config_dict(cfg)["episodes"]["lineups"] == [[key, 1.0]]
    assert train.config_dict(cfg)["eval"]["lineups"] == [key]

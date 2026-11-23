"""Ascension weights of single lineups (`lineup_ascensions`): the curriculum of the lineups that are ready for it."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spire.actor import EpisodeDist, EpisodeSampler, random_seed, validate_distribution
from spire.metrics import MetricsWriter
from spire.train import SamplingWeights, TrainConfig, build
from spire.vocab import Vocab

TEAM = "IRONCLAD+IRONCLAD+SILENT+SILENT"


def test_validate_reads_canonicalises_and_refuses(vocab: Vocab) -> None:
    base = EpisodeDist(lineups=(("IRONCLAD", 1.0),))
    d = validate_distribution({"lineups": [["IRONCLAD", 3], [TEAM, 1]], "ascensions": [[0, 1]],
                               "lineup_ascensions": [["SILENT+IRONCLAD+SILENT+IRONCLAD", [[0, 2], [3, 1]]]]}, vocab, base)
    assert d.lineup_ascensions == ((TEAM, ((0, 2.0), (3, 1.0))),)
    assert validate_distribution({}, vocab, d) == d  # missing keys keep the base, overrides included
    assert validate_distribution({"lineup_ascensions": []}, vocab, d).lineup_ascensions == ()
    bad: list[tuple[dict[str, Any], str]] = [
           ({"lineup_ascensions": [["SILENT", [[1, 1]]]]}, "not in lineups"),
           ({"lineups": [["IRONCLAD", 1]]}, "not in lineups"),  # the base's override would lose its lineup
           ({"lineup_ascensions": [[TEAM, []]]}, "non-empty"),
           ({"lineup_ascensions": [[TEAM, [[25, 1]]]]}, "0..20"),
           ({"lineup_ascensions": [[TEAM, [[1, 0]]]]}, "all weights are zero"),
           ({"lineup_ascensions": [[TEAM, [[1, 1]]], [TEAM, [[2, 1]]]]}, "duplicate"),
           ({"lineup_ascensions": [[TEAM]]}, "expected a list"),
           ({"lineup_ascensions": [["NOBODY", [[1, 1]]]]}, "unknown")]
    for data, msg in bad:
        with pytest.raises(ValueError, match=msg):
            validate_distribution(data, vocab, d)


def test_sampler_draws_each_lineup_from_its_own_ascensions(vocab: Vocab) -> None:
    dist = EpisodeDist(lineups=(("IRONCLAD", 1.0), (TEAM, 1.0)), ascensions=((0, 1.0),),
                       lineup_ascensions=((TEAM, ((2, 3.0), (5, 1.0))),))
    sampler = EpisodeSampler(dist, vocab, seed=7)
    seen = Counter((job.lineup, job.ascension) for job in (sampler.next_job(0) for _ in range(4000)))
    assert set(seen) == {("IRONCLAD", 0), (TEAM, 2), (TEAM, 5)}
    assert 0.70 < seen[(TEAM, 2)] / (seen[(TEAM, 2)] + seen[(TEAM, 5)]) < 0.80


def test_without_overrides_the_job_stream_is_the_old_one(vocab: Vocab) -> None:
    dist = EpisodeDist(lineups=(("IRONCLAD", 3.0), ("SILENT+SILENT", 1.0)), ascensions=((0, 1.0), (10, 1.0)))
    sampler = EpisodeSampler(dist, vocab, seed=11)
    rng = np.random.default_rng(11)
    names, p_lineup, ascs, p_asc = ["IRONCLAD", "SILENT+SILENT"], [0.75, 0.25], [0, 10], [0.5, 0.5]
    for _ in range(300):  # the two draws and the seed draw of the implementation before lineup_ascensions
        name = names[int(rng.choice(2, p=p_lineup))]
        asc = ascs[int(rng.choice(2, p=p_asc))]
        job = sampler.next_job(0)
        assert (job.lineup, job.ascension, job.seed) == (name, asc, random_seed(rng))


def test_weights_file_carries_the_overrides_to_the_actor(vocab: Vocab, tmp_path: Path) -> None:
    cfg = build(TrainConfig, {"run_name": "s", "run_root": str(tmp_path), "sampling_poll_s": 0.0,
                              "episodes": {"lineups": [["IRONCLAD", 1.0], [TEAM, 1.0]]}})
    assert cfg.episodes.lineup_ascensions == ()
    run = cfg.run_dir
    run.mkdir(parents=True)
    sw = SamplingWeights(cfg, run, MetricsWriter(run / "metrics.jsonl"))
    assert sw.active["lineup_ascensions"] == []
    sampler = EpisodeSampler(cfg.episodes, vocab, 0, weights_file=run / "sampling.active.json", poll_s=0.0)
    time.sleep(0.05)
    (run / "sampling.json").write_text(json.dumps({"lineups": [["IRONCLAD", 1.0], [TEAM, 1.0]],
                                                   "lineup_ascensions": [[TEAM, [[4, 1.0]]]]}), encoding="utf-8")
    assert sw.poll() and sw.active["lineup_ascensions"] == [[TEAM, [[4, 1.0]]]]
    assert sampler.reload() and sampler.weights_version == 2
    assert {(j.lineup, j.ascension) for j in (sampler.next_job(0) for _ in range(200))} == {("IRONCLAD", 0), (TEAM, 4)}
    time.sleep(0.05)
    (run / "sampling.json").write_text(json.dumps({"lineups": [["IRONCLAD", 1.0], [TEAM, 1.0]]}), encoding="utf-8")
    assert sw.poll() and sw.active["lineup_ascensions"] == []  # removing the key removes the curriculum
    assert sampler.reload()
    assert {j.ascension for j in (sampler.next_job(0) for _ in range(200))} == {0}

from __future__ import annotations

import pytest

from spire.synth import FrameGen
from spire.vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab


@pytest.fixture(scope="session")
def vocab() -> Vocab:
    return Vocab(make_synthetic_vocab())


@pytest.fixture(scope="session")
def catalog(vocab: Vocab) -> Catalog:
    return Catalog(make_synthetic_catalog(vocab, seed=1), vocab)


@pytest.fixture(scope="session")
def gen(vocab: Vocab, catalog: Catalog) -> FrameGen:
    return FrameGen(vocab, catalog.hash)

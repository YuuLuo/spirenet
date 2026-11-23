from __future__ import annotations

import numpy as np
import pytest
import torch

from ppo_stubs import CorridorEnv, collect, obs_frame
from spire.learner import Learner, PPOConfig
from spire.model import (
    OUTPUT_KEYS,
    PRESETS,
    GraphRunner,
    ModelConfig,
    SpireNet,
    checkpoint,
    export_weights,
    from_checkpoint,
    import_weights,
)
from spire.proto import parse_frame
from spire.rewards import RewardConfig, RoomIds
from spire.rollout import RolloutConfig, RolloutStore, SegmentRing
from spire.schema import (
    DECISION_KINDS,
    DIR_STRIDE,
    DYNAMIC_POINTERS,
    N_COLS,
    N_TABLES,
    POINTERS,
    CardCol,
    Kind,
    Table,
    Zone,
)
from spire.synth import FrameGen, random_frame
from spire.tokenize import BatchShape, Sample, Tokenizer, TokenizerConfig, batch_to
from spire.vocab import Catalog, Vocab

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)


@pytest.fixture(scope="module")
def tok(vocab: Vocab) -> Tokenizer:
    return Tokenizer(vocab)


@pytest.fixture(scope="module")
def frames(gen: FrameGen) -> list[bytes]:
    rng = np.random.default_rng(0)
    out = [random_frame(gen, rng, k, n_seats=s) for k in DECISION_KINDS for s in (1, 2, 4)]
    out += [random_frame(gen, rng, k, n_seats=4, size="extreme") for k in (Kind.COMBAT, Kind.CRYSTAL)]
    return out


def make_net(tok: Tokenizer, catalog: Catalog | None, seed: int = 0, config: ModelConfig = TINY) -> SpireNet:
    torch.manual_seed(seed)
    return SpireNet(tok.layout, config, catalog).eval()


def run(net: SpireNet, tok: Tokenizer, samples: list[Sample], shape: BatchShape | None = None) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return net(tok.collate(samples, shape=shape))


def encode(tok: Tokenizer, payloads: list[bytes]) -> list[Sample]:
    return [tok.encode(parse_frame(p)) for p in payloads]


# --------------------------------------------------------------------------- shapes and masks


@pytest.mark.parametrize("pooled", [True, False])
def test_outputs_and_masks_every_kind(tok: Tokenizer, vocab: Vocab, catalog: Catalog, frames: list[bytes],
                                      pooled: bool) -> None:
    t = tok if pooled else Tokenizer(vocab, TokenizerConfig(pool_children=False))
    net = make_net(t, catalog)
    samples = encode(t, frames)
    out = run(net, t, samples)
    assert set(out) == set(OUTPUT_KEYS)
    B = len(samples)
    A = max(s.n_actions for s in samples)
    assert out["logits"].shape[0] == B and out["logits"].shape[1] >= A
    for k in OUTPUT_KEYS[1:]:
        assert out[k].shape == (B,) and out[k].dtype == torch.float32 and torch.isfinite(out[k]).all()
    for i, s in enumerate(samples):
        row = out["logits"][i]
        assert torch.isfinite(row[: s.n_actions]).all()
        assert torch.isneginf(row[s.n_actions :]).all()
    assert torch.isfinite(torch.log_softmax(out["logits"], -1)[out["logits"] > -np.inf]).all()


def test_padding_does_not_change_outputs(tok: Tokenizer, catalog: Catalog, frames: list[bytes]) -> None:
    net = make_net(tok, catalog)
    samples = encode(tok, frames)
    batch = run(net, tok, samples)
    nat = tok.shape_of(samples)
    padded = run(net, tok, samples, BatchShape(nat.batch + 5, nat.tokens + 100, nat.actions + 24, nat.kids + 128,
                                               nat.vars + 128, nat.edges + 128, nat.rows + 333))
    for i, s in enumerate(samples):
        alone = run(net, tok, [s])
        a = s.n_actions
        for o in (batch, padded):
            torch.testing.assert_close(o["logits"][i, :a], alone["logits"][0, :a], atol=2e-5, rtol=1e-4)
            for k in OUTPUT_KEYS[1:]:
                torch.testing.assert_close(o[k][i], alone[k][0], atol=2e-5, rtol=1e-4)
    for k in OUTPUT_KEYS:
        assert torch.isfinite(padded[k][len(samples):]).all() or k == "logits"


# --------------------------------------------------------------------------- permutation invariance


def permute_rows(payload: bytes, table: Table, perm: np.ndarray) -> bytes:
    """Reorder the rows of ``table`` (new row i = old row perm[i]) and remap every reference to it.

    CSR offset columns of the owners become stale (the tokenizer uses owner pointers, not offsets).
    """
    words = np.frombuffer(payload, dtype="<i4").copy()
    d = words[16 + 64 : 16 + 64 + N_TABLES * DIR_STRIDE].reshape(N_TABLES, DIR_STRIDE)
    tabs = [words[d[t, 3] // 4 : d[t, 3] // 4 + d[t, 1] * N_COLS[t]].reshape(d[t, 1], N_COLS[t])
            for t in range(N_TABLES)]
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    tabs[table][:] = tabs[table][perm].copy()

    def remap(a: np.ndarray, col: int, mask: np.ndarray | None = None) -> None:
        v = a[:, col]
        m = v >= 0 if mask is None else mask & (v >= 0)
        v[m] = inv[v[m]]

    for (t, col), tgt in POINTERS.items():
        if tgt == table:
            remap(tabs[t], col)
    for t, tcol, rcol in DYNAMIC_POINTERS:
        remap(tabs[t], rcol, tabs[t][:, tcol] == table)
    return words.tobytes()


@pytest.mark.parametrize("table", [Table.CARDS, Table.CREATURES, Table.RELICS, Table.POTIONS, Table.MAP_NODES,
                                   Table.OPTIONS, Table.POWERS])
def test_row_order_carries_no_information(tok: Tokenizer, catalog: Catalog, gen: FrameGen, table: Table) -> None:
    net = make_net(tok, catalog)
    rng = np.random.default_rng(int(table))
    kinds = {Table.MAP_NODES: Kind.MAP, Table.OPTIONS: Kind.SHOP, Table.RELICS: Kind.EVENT}.get(table, Kind.COMBAT)
    checked = 0
    for _ in range(6):
        p = random_frame(gen, rng, kinds, n_seats=2)
        n = parse_frame(p).table(table).shape[0]
        if n < 2:
            continue
        perm = rng.permutation(n)
        a = run(net, tok, encode(tok, [p]))
        b = run(net, tok, encode(tok, [permute_rows(p, table, perm)]))
        for k in OUTPUT_KEYS:
            torch.testing.assert_close(a[k], b[k], atol=1e-5, rtol=1e-4)
        checked += 1
    assert checked >= 3


def test_draw_pile_multiset_order_is_irrelevant(tok: Tokenizer, catalog: Catalog, gen: FrameGen) -> None:
    """Permuting only the DRAW multiset rows (schema 5.2.3: order carries no information)."""
    net = make_net(tok, catalog)
    rng = np.random.default_rng(11)
    done = 0
    while done < 3:
        p = random_frame(gen, rng, Kind.COMBAT)
        cards = parse_frame(p).table(Table.CARDS)
        draw = np.flatnonzero(cards[:, CardCol.ZONE] == Zone.DRAW)
        if len(draw) < 3:
            continue
        perm = np.arange(len(cards))
        perm[draw] = draw[rng.permutation(len(draw))]
        a, b = run(net, tok, encode(tok, [p])), run(net, tok, encode(tok, [permute_rows(p, Table.CARDS, perm)]))
        for k in OUTPUT_KEYS:
            torch.testing.assert_close(a[k], b[k], atol=1e-5, rtol=1e-4)
        done += 1


def test_action_order_permutes_logits(tok: Tokenizer, catalog: Catalog, gen: FrameGen) -> None:
    net = make_net(tok, catalog)
    rng = np.random.default_rng(12)
    p = random_frame(gen, rng, Kind.COMBAT, n_seats=2)
    n = parse_frame(p).n_actions
    perm = rng.permutation(n)
    words = np.frombuffer(p, dtype="<i4").copy()
    d = words[16 + 64 : 16 + 64 + N_TABLES * DIR_STRIDE].reshape(N_TABLES, DIR_STRIDE)
    o = d[Table.ACTIONS, 3] // 4
    acts = words[o : o + n * N_COLS[Table.ACTIONS]].reshape(n, -1)
    acts[:] = acts[perm].copy()
    a, b = run(net, tok, encode(tok, [p])), run(net, tok, encode(tok, [words.tobytes()]))
    torch.testing.assert_close(b["logits"][0, :n], a["logits"][0, perm], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(a["value"], b["value"], atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- training properties


def test_gradient_reaches_every_parameter(tok: Tokenizer, vocab: Vocab, catalog: Catalog, frames: list[bytes]) -> None:
    net = make_net(tok, catalog).train()
    tok_mode = Tokenizer(vocab, TokenizerConfig(pool_children=False))  # child tokens exercise the owner link
    loss = torch.zeros(())
    for t in (tok, tok_mode):
        out = net(t.collate(encode(t, frames)))
        legal = torch.isfinite(out["logits"])
        lp = torch.log_softmax(out["logits"], -1)
        entropy = -(lp.exp() * lp.masked_fill(~legal, 0.0)).sum()  # mask before the product: no 0 * -inf
        loss = loss - entropy + lp[:, 0].sum() + sum(out[k].pow(2).sum() for k in OUTPUT_KEYS[1:])
    loss.backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())  # type: ignore[arg-type]


def test_determinism_under_fixed_seed(tok: Tokenizer, catalog: Catalog, frames: list[bytes]) -> None:
    samples = encode(tok, frames)
    a, b = make_net(tok, catalog, seed=3), make_net(tok, catalog, seed=3)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb)
    oa, ob, oa2 = run(a, tok, samples), run(b, tok, samples), run(a, tok, samples)
    for k in OUTPUT_KEYS:
        assert torch.equal(oa[k], ob[k]) and torch.equal(oa[k], oa2[k])
    c = make_net(tok, catalog, seed=4)
    assert not torch.equal(run(c, tok, samples)["value"], oa["value"])


def test_state_dict_round_trip(tok: Tokenizer, catalog: Catalog, frames: list[bytes], tmp_path) -> None:  # type: ignore[no-untyped-def]
    samples = encode(tok, frames)
    a = make_net(tok, catalog, seed=1)
    state = export_weights(a)
    assert all(v.device.type == "cpu" for v in state.values())
    assert set(state) == {n for n, _ in a.named_parameters()}  # buffers are rebuilt, not stored
    b = make_net(tok, catalog, seed=2)
    import_weights(b, state)
    oa, ob = run(a, tok, samples), run(b, tok, samples)
    for k in OUTPUT_KEYS:
        assert torch.equal(oa[k], ob[k])
    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint(a), path)
    c = from_checkpoint(torch.load(path, weights_only=True), tok.layout, catalog).eval()
    for k in OUTPUT_KEYS:
        assert torch.equal(run(c, tok, samples)[k], oa[k])


def test_catalog_is_optional_and_changes_outputs(tok: Tokenizer, catalog: Catalog, frames: list[bytes]) -> None:
    samples = encode(tok, frames)
    with_cat, without = make_net(tok, catalog), make_net(tok, None)
    assert with_cat.n_params > without.n_params
    a, b = run(with_cat, tok, samples), run(without, tok, samples)
    assert torch.isfinite(b["value"]).all() and not torch.allclose(a["value"], b["value"])


def test_learns_context_dependent_pointer(tok: Tokenizer) -> None:
    """The correct option is ``context % n_options`` (context in GLOBAL CUR_COL): needs global x option.

    Gradients are clipped to norm 1 as in the PPO learner: unclipped, this 12-sample toy hits gradient spikes
    (norm ~40) and collapses to uniform logits for most initialisations (measured: 1-2 of 6 seeds learn it in
    250-500 steps, with or without the errata-v1 GLOBAL features); clipped, 6 of 6 seeds reach loss < 0.05."""
    data = [(tok.encode(obs_frame(Kind.EVENT, 1, 0, False, [100], n_actions=n, context=ctx)), ctx % n)
            for ctx in range(4) for n in range(2, 5)]
    batch = tok.collate([s for s, _ in data])
    target = torch.tensor([a for _, a in data])
    net = make_net(tok, None).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(400):
        logits = net(batch)["logits"]
        loss = torch.nn.functional.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    assert loss.item() < 0.05 and torch.equal(logits.argmax(1), target)


def test_learner_contract(tok: Tokenizer, vocab: Vocab) -> None:
    """Real Tokenizer + SpireNet through spire.learner.Learner (two PPO iterations on CorridorEnv)."""
    torch.manual_seed(0)
    cfg = PPOConfig(epochs=2, minibatch=128, micro_batch=64, micro_tokens=2048, seed=0)
    learner = Learner(SpireNet(tok.layout, TINY), tok, cfg, device="cpu",
                      size_fn=lambda s: tok.config.bucket(s.n_tokens))
    ring = SegmentRing()
    store = RolloutStore(ring.put, RoomIds.from_vocab(vocab), config=RolloutConfig(segment_len=16),
                         rewards=RewardConfig(gamma=cfg.gae.gamma))
    envs = [CorridorEnv(vocab, i) for i in range(8)]
    frames: list = [None] * len(envs)
    gen = torch.Generator().manual_seed(0)
    before = [p.detach().clone() for p in learner.model.parameters()]
    samples: list[Sample] = []
    for _ in range(2):
        while ring.n_samples < 256:
            collect(envs, frames, tok, learner.model, store, learner.version, 8, gen)  # type: ignore[arg-type]
        segs = ring.take(256)
        assert segs is not None
        m = learner.update(segs)
        assert m["illegal_chosen"] == 0 and m["nonfinite_skipped"] == 0
        samples = [s for seg in segs for s in seg.samples]
    assert learner.version == 2
    assert any(not torch.equal(a, p) for a, p in zip(before, learner.model.parameters()))
    v = learner.predict_values(samples)
    assert v.shape == (len(samples),) and np.isfinite(v).all()


def test_id_dropout_hides_learned_ids_of_static_categories(tok: Tokenizer, catalog: Catalog,
                                                           frames: list[bytes]) -> None:
    batch = tok.collate(encode(tok, frames))
    net = make_net(tok, catalog, config=ModelConfig(**{**TINY.__dict__, "id_dropout": 1.0}))
    off, rows = tok.layout.regions["card"]

    def outputs(train: bool) -> torch.Tensor:
        net.train(train)
        with torch.no_grad():
            return net(batch)["value"]

    base_train, base_eval = outputs(True), outputs(False)
    with torch.no_grad():
        net.ids.weight[off + 2 : off + rows].normal_()
    assert torch.equal(outputs(True), base_train)  # learned card rows are always dropped
    assert not torch.allclose(outputs(False), base_eval)  # ... and used at evaluation


def test_preset_sizes(tok: Tokenizer, catalog: Catalog) -> None:
    sizes = {k: SpireNet(tok.layout, c, catalog).n_params for k, c in PRESETS.items()}
    assert 2.3e6 < sizes["2.5m"] < 2.7e6 and 3.8e6 < sizes["4m"] < 4.4e6 and 6.2e6 < sizes["6.5m"] < 6.9e6


# --------------------------------------------------------------------------- CUDA


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_bf16_autocast_and_inference_mode_close_to_fp32(tok: Tokenizer, catalog: Catalog, frames: list[bytes]) -> None:
    samples = encode(tok, frames)
    net = make_net(tok, catalog, config=PRESETS["2.5m"])
    ref = run(net, tok, samples)
    net = net.cuda()
    b = batch_to(tok.collate(samples), "cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = net(b)
    for k in OUTPUT_KEYS:
        x, y = out[k].cpu(), ref[k]
        fin = torch.isfinite(y)
        assert torch.equal(fin, torch.isfinite(x))
        assert (x[fin] - y[fin]).abs().max() < 5e-3


@cuda
def test_graph_runner_matches_eager_and_tracks_weight_updates(tok: Tokenizer, catalog: Catalog,
                                                              frames: list[bytes]) -> None:
    samples = encode(tok, frames)
    net = make_net(tok, catalog, config=PRESETS["2.5m"]).cuda()
    runner = GraphRunner(net, tok)
    for part in (samples[:5], samples[5:17], samples):
        g = runner(part)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            e = net(batch_to(tok.collate(part), "cuda"))
        a = max(s.n_actions for s in part)
        torch.testing.assert_close(g["logits"][:, :a], e["logits"][:, :a], atol=2e-3, rtol=1e-2)
        torch.testing.assert_close(g["value"], e["value"], atol=2e-3, rtol=1e-2)
        assert g["value"].shape == (len(part),)
    captures = runner.captures
    runner(samples[:5])
    assert runner.captures == captures  # replayed, not recaptured
    before = runner(samples[:5])["value"].clone()
    other = make_net(tok, catalog, seed=9, config=PRESETS["2.5m"])
    import_weights(net, export_weights(other))
    after = runner(samples[:5])["value"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expect = net(batch_to(tok.collate(samples[:5]), "cuda"))["value"]
    assert not torch.allclose(before, after)
    torch.testing.assert_close(after, expect, atol=2e-3, rtol=1e-2)

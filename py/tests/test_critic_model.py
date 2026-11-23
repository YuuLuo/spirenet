"""Package P2 of the critic contract (notes/critic/contract.md 2.2, 5; build-v1.md 3): the tower, ``forward_critic``,
construction order, optimizer groups, the GraphRunner output and the migration functions (PG2 on synthetic frames;
the real-frame proof is scripts/migrate_critic.py)."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

import critic_data as cd
from spire.critic import CriticConfig, CriticPPOConfig, empty_crit, optimizer_groups, side
from spire.critic_migration import migrate_critic_optimizer, migrate_critic_weights
from spire.learner import _param_groups
from spire.model import OUTPUT_KEYS, PRESETS, GraphRunner, ModelConfig, SpireNet, model_config_dict, packed_layout
from spire.proto import parse_frame
from spire.train import ModelSection, TrainConfig, build, config_dict

SMALL = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_id=16, d_cond=16, d_static=32)
TOWER_4M = 2 * 662_368 + 50_400 + 7_168 + 3_584 + 51_075 + 496  # contract 2.2, by hand: 1,437,459


def _batch(tok, n: int = 48, lo: int = 0):
    frames = [parse_frame(p) for p in cd.stream("calc").frames[lo : lo + n]]
    enc = tok.encode_batch(frames)
    crit = np.stack([side(f, int(enc.card_base[i]), int(enc.counts[i, 0]), 32) for i, f in enumerate(frames)])
    return tok.collate(enc.samples()), torch.from_numpy(crit), frames


def test_tower_size_and_birth_values(vocab) -> None:
    tok, _ = cd.tokenizers("calc")
    for layers in (4, 8):  # the 4 M trunk and its 8-block version
        cfg = dataclasses.replace(PRESETS["4m"], n_layers=layers, critic=CriticConfig())
        net = SpireNet(tok.layout, cfg).eval()
        assert sum(p.numel() for p in net.critic.parameters()) == TOWER_4M  # type: ignore[union-attr]
        plain = SpireNet(tok.layout, dataclasses.replace(cfg, critic=None))
        assert net.n_params - plain.n_params == TOWER_4M
    net = SpireNet(tok.layout, dataclasses.replace(SMALL, critic=CriticConfig())).eval()
    batch, crit, _ = _batch(tok)
    with torch.no_grad():
        o = net.forward_critic(batch, crit)
    assert torch.equal(o["vc"], torch.tensor([[0.5, 0.0, 0.0]]).expand_as(o["vc"]))
    assert torch.equal(o["win_logit_c"], torch.zeros_like(o["win_logit_c"]))


def test_only_the_4m_tower_is_built() -> None:
    with pytest.raises(AssertionError, match="4 M tower"):
        CriticConfig(d=256, layers=3, heads=8, d_ff=1024)
    with pytest.raises(AssertionError, match="adv_weights"):
        CriticPPOConfig(adv_weights={"10": [1.0, 0.3, 1.0]})
    assert CriticPPOConfig.weights(10) == (1.0, 1.0, 1.0)


def test_construction_order_and_rng(vocab) -> None:
    """A network with the tower: every trunk parameter equal to the network without it (same seed), the tower's
    parameters last, and the global RNG left where the network without the tower leaves it."""
    tok, _ = cd.tokenizers("calc", "teach")
    torch.manual_seed(5)
    a = SpireNet(tok.layout, SMALL)
    ra = torch.rand(4)
    torch.manual_seed(5)
    b = SpireNet(tok.layout, dataclasses.replace(SMALL, critic=CriticConfig()))
    rb = torch.rand(4)
    assert torch.equal(ra, rb)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == [k for k in sb if not k.startswith("critic.")]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    names = [n for n, _ in b.named_parameters()]
    assert names[-sum(1 for n in names if n.startswith("critic.")):] == [n for n in names if n.startswith("critic.")]
    c = SpireNet(tok.layout, dataclasses.replace(SMALL, critic=CriticConfig()))
    assert all(torch.equal(v, c.state_dict()[k]) for k, v in sb.items() if k.startswith("critic."))  # own generator
    batch, crit, _ = _batch(tok)
    a.eval(), b.eval()
    with torch.no_grad():
        oa, ob, oc = a(batch), b(batch), b.forward_critic(batch, crit)
    assert oa.keys() == ob.keys() and all(torch.equal(oa[k], ob[k]) and torch.equal(oa[k], oc[k]) for k in oa)


def test_optimizer_groups_extend_the_trunk_groups(vocab) -> None:
    tok, _ = cd.tokenizers("calc")
    a = SpireNet(tok.layout, SMALL)
    b = SpireNet(tok.layout, dataclasses.replace(SMALL, critic=CriticConfig()))
    ga, gb = _param_groups(a, 0.01), optimizer_groups(b, 0.01, 3e-4)
    na = {id(p): n for n, p in a.named_parameters()}
    nb = {id(p): n for n, p in b.named_parameters()}
    assert len(gb) == len(ga) + 2
    for x, y in zip(ga, gb):
        assert [na[id(p)] for p in x["params"]] == [nb[id(p)] for p in y["params"]] and x["weight_decay"] == \
            y["weight_decay"] and "critic" not in y
    tower = [nb[id(p)] for g in gb[len(ga):] for p in g["params"]]
    assert sorted(tower) == sorted(n for n in nb.values() if n.startswith("critic.")) and all(
        g["critic"] and g["lr"] == 3e-4 for g in gb[len(ga):])


def test_model_config_round_trip_and_absent_keys() -> None:
    cfg = dataclasses.replace(SMALL, critic=CriticConfig())
    d = model_config_dict(cfg)
    assert d["critic"] == dataclasses.asdict(CriticConfig())
    assert ModelConfig(**json.loads(json.dumps(d))) == cfg == build(ModelConfig, d)
    assert "critic" not in model_config_dict(SMALL)
    tc = build(TrainConfig, {"run_name": "x", "model": {"preset": "2.5m"}})
    cd_ = config_dict(tc)
    assert "critic" not in cd_["model"] and '"critic"' not in json.dumps(cd_)
    tc2 = build(TrainConfig, {"run_name": "x", "model": {"preset": "2.5m", "critic": {}}})
    assert tc2.model.config().critic == CriticConfig() and config_dict(tc2)["model"]["critic"]["draw_k"] == 32
    assert ModelSection().config().critic is None


def test_pg2_migration_functions(vocab) -> None:
    """Weights: every old key exact, the tower as SpireNet draws it; old outputs bit-identical (online, EMA);
    optimizer: saved entries and group settings exact, the tower's groups zero, step 0; one AdamW step runs."""
    tok, _ = cd.tokenizers("calc", "teach")
    torch.manual_seed(1)
    src = SpireNet(tok.layout, SMALL)
    opt = torch.optim.AdamW(_param_groups(src, 0.01), lr=1e-3)
    batch, crit, _ = _batch(tok)
    for _ in range(2):
        opt.zero_grad()
        o = src(batch)
        (o["value"] ** 2).mean().add(o["pwin_logit"].mean()).backward()
        opt.step()
    ema = {k: v * 0.5 for k, v in src.state_dict().items()}
    dst = SpireNet(tok.layout, dataclasses.replace(SMALL, critic=CriticConfig())).eval()
    w = migrate_critic_weights(src.state_dict(), dst)
    e = migrate_critic_weights(ema, dst)
    fresh = SpireNet(tok.layout, dataclasses.replace(SMALL, critic=CriticConfig()))
    assert all(torch.equal(w[k], fresh.state_dict()[k]) for k in w if k.startswith("critic."))
    src.eval()
    for state, ref in ((w, src.state_dict()), (e, ema)):
        dst.load_state_dict(state)
        src.load_state_dict(ref)
        with torch.no_grad():
            x, y, z = src(batch), dst(batch), dst.forward_critic(batch, crit)
        assert all(torch.equal(x[k], y[k]) and torch.equal(x[k], z[k]) for k in OUTPUT_KEYS)
    new_opt = torch.optim.AdamW(optimizer_groups(dst, 0.01, 3e-4), lr=1e-3)
    saved = opt.state_dict()
    m = migrate_critic_optimizer(saved, dst, new_opt)
    new_opt.load_state_dict(m)
    got = new_opt.state_dict()
    for i, entry in saved["state"].items():
        assert all(torch.equal(got["state"][i][k], v) for k, v in entry.items())
    for g_old, g_new in zip(saved["param_groups"], got["param_groups"]):
        assert {k: v for k, v in g_old.items() if k != "params"} == {k: v for k, v in g_new.items() if k != "params"}
    n_old = len(saved["param_groups"])
    for g in got["param_groups"][n_old:]:
        assert g["critic"] and g["lr"] == 3e-4
        for i in g["params"]:
            assert int(got["state"][i]["step"]) == 0 and not got["state"][i]["exp_avg"].any()
    with pytest.raises(AssertionError, match="pre-critic"):
        migrate_critic_weights({k: v for k, v in list(src.state_dict().items())[1:]}, dst)
    dst.train()
    new_opt.zero_grad()
    o = dst.forward_critic(batch, crit)
    (o["vc"] ** 2).sum().backward()
    new_opt.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_runner_critic_layout(vocab) -> None:
    tok, _ = cd.tokenizers("calc")
    net = SpireNet(tok.layout, dataclasses.replace(SMALL, critic=CriticConfig())).cuda().eval()
    with torch.no_grad():
        for p in net.critic.parameters():  # type: ignore[union-attr]
            p.add_(0.05 * torch.randn_like(p))
    frames = [parse_frame(p) for p in cd.stream("calc").frames[:64]]
    enc = tok.encode_batch(frames)
    crit = np.stack([side(f, int(enc.card_base[i]), int(enc.counts[i, 0]), 32) for i, f in enumerate(frames)])
    runner = GraphRunner(net, tok, critic=True)
    nat = tok.shape_encoded(enc, 0, 64)
    shape = runner.pad_shape(nat)
    layout, size = packed_layout(shape, tok.width, 32)
    assert layout["crit"][1] == (shape.batch, 34) and layout["crit"][2] == torch.int16
    buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    from spire.model import packed_views
    views = {k: v.numpy() for k, v in packed_views(buf, layout).items()}
    tok.collate_encoded_into(enc, 0, 64, shape, {k: v for k, v in views.items() if k != "crit"}, nat=nat)
    views["crit"][:] = empty_crit(shape.batch, 32)
    views["crit"][:64] = crit
    out = runner.run_packed(buf, shape, 64).clone()
    assert out.shape == (64, shape.actions + 4)
    batch = {k: v.cuda() for k, v in tok.collate_encoded(enc, 0, 64, shape=shape).items()}
    pad = torch.from_numpy(views["crit"].copy()).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager = net.forward_critic(batch, pad)
    assert torch.allclose(out[:, shape.actions + 1 :], eager["vc"][:64].float(), atol=2e-2)
    with pytest.raises(AssertionError, match="model.critic"):
        GraphRunner(SpireNet(tok.layout, SMALL).cuda().eval(), tok, critic=True)

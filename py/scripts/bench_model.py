"""GPU benchmark for SpireNet: inference / training throughput, bucket sweep, two-thread contention.

    python -B scripts/bench_model.py [--presets 2.5m,4m,6.5m] [--buckets 48,80,112,160,224]
                                     [--batches 16,32,64,128,256] [--seconds 2] [--out PATH]

Data: synthetic frames (``spire.synth.FrameGen``, schema 11 "typical" sizes; 2/4-seat and "extreme"
frames fill the large buckets). The training mix is single-seat with a decision-kind mix close to a
full run (combat ~72%). Sections:

1. tokenizer cost (encode us/frame, collate us/sample) on the training mix;
2. inference samples/s per preset x token bucket x batch size, CUDA-graph replay (``GraphRunner``,
   collate + H2D copy included) and eager (bf16 autocast, ``inference_mode``) for reference;
3. training samples/s per preset: minibatch 4096 split into same-bucket sub-batches under a token
   budget, forward + backward + AdamW step (bf16 autocast, grad clip 1.0), per bucket and on the mix;
4. contention: inference thread (graphs, batch 64, mix) and training thread in one process, alone
   vs concurrent (default stream; high-priority inference stream; plus a 0.5 ms GIL switch interval),
   with inference latency p50/p99 (each call includes reading the logits back to the CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spire.model import PRESETS, GraphRunner, SpireNet  # noqa: E402
from spire.proto import parse_frame  # noqa: E402
from spire.schema import RESET_FLAGS_DEFAULT, CardSelectSub, Kind, ViewMode  # noqa: E402
from spire.synth import COMBAT_SELECT_SUBS, FrameGen  # noqa: E402
from spire.tokenize import Batch, Sample, Tokenizer, batch_to  # noqa: E402
from spire.vocab import Catalog, Vocab, make_synthetic_catalog, make_synthetic_vocab  # noqa: E402

KIND_MIX = {
    Kind.COMBAT: 0.72, Kind.CARD_SELECT: 0.06, Kind.MAP: 0.06, Kind.CARD_REWARD: 0.04, Kind.REWARD: 0.04,
    Kind.EVENT: 0.03, Kind.SHOP: 0.02, Kind.REST: 0.015, Kind.TREASURE: 0.01, Kind.CONFIRM: 0.002,
    Kind.BUNDLE: 0.002, Kind.CRYSTAL: 0.001,
}
MINIBATCH = 4096
DEV = torch.device("cuda")


# --------------------------------------------------------------------------- data


def gen_frames(gen: FrameGen, rng: np.random.Generator, n: int, *, seats: int = 1, size: str = "typical",
               kinds: dict[Kind, float] = KIND_MIX) -> list[bytes]:
    ks, ps = list(kinds), np.array(list(kinds.values()))
    out = []
    for kind in rng.choice(len(ks), n, p=ps / ps.sum()):
        k = ks[int(kind)]
        ctx = gen.random_ctx(rng, seats, view=ViewMode.TEAM, size=size)  # type: ignore[arg-type]
        ctx.reset_flags = int(RESET_FLAGS_DEFAULT)
        sub = None
        if k == Kind.CARD_SELECT:  # in-combat prompts dominate during a run
            subs = sorted(COMBAT_SELECT_SUBS) if rng.random() < 0.7 else [s for s in CardSelectSub]
            sub = int(rng.choice([int(s) for s in subs]))
        out.append(gen.frame(rng, k, ctx, kind_sub=sub, size=size, hashes=False))  # type: ignore[arg-type]
    return out


def build_pool(tok: Tokenizer, gen: FrameGen, n_mix: int, seed: int) -> tuple[list[Sample], dict[int, list[Sample]]]:
    rng = np.random.default_rng(seed)
    mix = [tok.encode(parse_frame(p)) for p in gen_frames(gen, rng, n_mix)]
    extra = (gen_frames(gen, rng, n_mix // 4, seats=2) + gen_frames(gen, rng, n_mix // 4, seats=4)
             + gen_frames(gen, rng, n_mix // 8, seats=4, size="extreme")
             + gen_frames(gen, rng, n_mix // 8, seats=1, size="extreme"))
    allsamples = mix + [tok.encode(parse_frame(p)) for p in extra]
    by: dict[int, list[Sample]] = {}
    for s in allsamples:
        by.setdefault(tok.config.bucket(s.n_tokens), []).append(s)
    return mix, by


def take(samples: Sequence[Sample], n: int, rng: np.random.Generator) -> list[Sample]:
    idx = rng.choice(len(samples), n, replace=len(samples) < n)
    return [samples[int(i)] for i in idx]


# --------------------------------------------------------------------------- timing helpers


def timed(fn: Callable[[], Any], seconds: float, min_iters: int = 5) -> float:
    """Seconds per call (after 2 warm-up calls), synchronising the GPU."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while n < min_iters or time.perf_counter() - t0 < seconds:
        fn()
        n += 1
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def bench_tokenizer(tok: Tokenizer, gen: FrameGen, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    frames = [parse_frame(p) for p in gen_frames(gen, rng, 1000)]
    combat = [parse_frame(p) for p in gen_frames(gen, rng, 300, kinds={Kind.COMBAT: 1.0})]
    combat4 = [parse_frame(p) for p in gen_frames(gen, rng, 200, seats=4, kinds={Kind.COMBAT: 1.0})]
    out: dict[str, float] = {}
    for name, fs in (("mix", frames), ("combat_1seat", combat), ("combat_4seat", combat4)):
        best = min(_per_item(lambda: [tok.encode(f) for f in fs], len(fs)) for _ in range(5))
        out[f"encode_us_{name}"] = best * 1e6
        ss = [tok.encode(f) for f in fs]
        for B in (64, 256):
            out[f"collate_us_per_sample_{name}_B{B}"] = 1e6 * min(
                _per_item(lambda: tok.collate(ss[:B]), B) for _ in range(10))
        out[f"bytes_per_sample_{name}"] = float(np.mean([s.nbytes for s in ss]))
        out[f"tokens_mean_{name}"] = float(np.mean([s.n_tokens for s in ss]))
    return out


def _per_item(fn: Callable[[], Any], n: int) -> float:
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) / n


# --------------------------------------------------------------------------- inference


def bench_inference(net: SpireNet, tok: Tokenizer, by: dict[int, list[Sample]], buckets: Sequence[int],
                    batches: Sequence[int], seconds: float, rng: np.random.Generator) -> list[dict[str, Any]]:
    net.eval()
    runner = GraphRunner(net, tok, max_graphs=64)
    rows = []
    for T in buckets:
        pool = by.get(T, [])
        if not pool:
            continue
        for B in batches:
            sub = take(pool, B, rng)
            dt = timed(lambda: runner(sub), seconds)
            batch = batch_to(tok.collate(sub), DEV)

            def eager() -> None:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    net(batch)
            de = timed(eager, seconds / 2)
            rows.append({"bucket": T, "batch": B, "graph_sps": B / dt, "graph_ms": dt * 1e3,
                         "eager_sps": B / de, "eager_ms": de * 1e3})
    runner.clear()
    return rows


# --------------------------------------------------------------------------- training


class Trainer:
    """Forward + backward + AdamW over one 4096-sample minibatch given as pre-collated sub-batches."""

    def __init__(self, net: SpireNet) -> None:
        self.net = net
        self.opt = torch.optim.AdamW(net.parameters(), lr=3e-4, fused=True)

    def step(self, subs: Sequence[tuple[Batch, int]]) -> torch.Tensor:
        net = self.net
        net.train()
        total = sum(n for _, n in subs)
        loss_sum = torch.zeros((), device=DEV)
        for batch, n in subs:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(batch)
            logp = torch.log_softmax(out["logits"], -1)
            act = batch["bench_action"]
            lp = logp.gather(1, act[:, None]).squeeze(1)
            adv = batch["bench_adv"]
            loss = -(lp * adv).sum() + 0.5 * ((out["value"] - adv) ** 2).sum()
            loss = loss + (torch.nn.functional.binary_cross_entropy_with_logits(
                out["pwin_logit"], (adv > 0).float(), reduction="sum") + (out["floors_pred"] ** 2).sum()
                + (out["hp_pred"] ** 2).sum()) * 0.1
            (loss / total).backward()
            loss_sum += loss.detach()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return loss_sum / total


def make_minibatch(tok: Tokenizer, samples: Sequence[Sample], max_tokens: int,
                   rng: np.random.Generator) -> list[tuple[Batch, int]]:
    subs = []
    for idx in tok.bucket_batches(samples, max_tokens):
        part = [samples[i] for i in idx]
        b = tok.collate(part)
        na = np.array([s.n_actions for s in part])
        b["bench_action"] = torch.from_numpy((rng.random(len(part)) * np.maximum(na, 1)).astype(np.int64))
        b["bench_adv"] = torch.from_numpy(rng.standard_normal(len(part)).astype(np.float32))
        subs.append((batch_to(b, DEV), len(part)))
    return subs


def bench_training(net: SpireNet, tok: Tokenizer, mix: Sequence[Sample], by: dict[int, list[Sample]],
                   buckets: Sequence[int], budgets: Sequence[int], bucket_budget: int, seconds: float,
                   rng: np.random.Generator) -> list[dict[str, Any]]:
    trainer = Trainer(net)
    rows = []
    cases: list[tuple[str, Sequence[Sample]]] = [("mix", mix)] + [(str(T), by[T]) for T in buckets if T in by]
    for name, pool in cases:
        samples = take(pool, MINIBATCH, rng)
        for budget in budgets if name == "mix" else [bucket_budget]:
            t0 = time.perf_counter()
            subs = make_minibatch(tok, samples, budget, rng)
            prep = time.perf_counter() - t0
            torch.cuda.reset_peak_memory_stats()
            dt = timed(lambda: trainer.step(subs), seconds, min_iters=3)
            rows.append({"case": name, "token_budget": budget, "sub_batches": len(subs), "train_sps": MINIBATCH / dt,
                         "step_s": dt, "collate_h2d_s": prep,
                         "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30})
            del subs
    return rows


# --------------------------------------------------------------------------- contention


def bench_contention(net_inf: SpireNet, net_train: SpireNet, tok: Tokenizer, mix: Sequence[Sample],
                     seconds: float, rng: np.random.Generator, budget: int, infer_batch: int) -> dict[str, Any]:
    net_inf.eval()
    runner = GraphRunner(net_inf, tok)
    inf_batches = [take(mix, infer_batch, rng) for _ in range(16)]
    for b in inf_batches:  # capture every shape up front
        runner(b)
    trainer = Trainer(net_train)
    subs = make_minibatch(tok, take(mix, MINIBATCH, rng), budget, rng)
    trainer.step(subs)
    torch.cuda.synchronize()

    def run(do_inf: bool, do_train: bool, prio: bool, switch_s: float | None = None) -> dict[str, Any]:
        old_switch = sys.getswitchinterval()
        if switch_s is not None:
            sys.setswitchinterval(switch_s)
        stop = threading.Event()
        lat: list[float] = []
        steps = [0]
        stream = torch.cuda.Stream(priority=-1) if prio else None

        def inf_loop() -> None:
            i = 0
            with torch.cuda.stream(stream) if stream is not None else _null():
                while not stop.is_set():
                    t = time.perf_counter()
                    out = runner(inf_batches[i % len(inf_batches)])
                    out["logits"].sum().item()  # the actor must read actions back: synchronise
                    lat.append(time.perf_counter() - t)
                    i += 1

        def train_loop() -> None:
            while not stop.is_set():
                trainer.step(subs)
                torch.cuda.current_stream().synchronize()
                steps[0] += 1

        threads = ([threading.Thread(target=inf_loop)] if do_inf else []) + (
            [threading.Thread(target=train_loop)] if do_train else [])
        t0 = time.perf_counter()
        for th in threads:
            th.start()
        time.sleep(seconds)
        stop.set()
        for th in threads:
            th.join()
        el = time.perf_counter() - t0
        sys.setswitchinterval(old_switch)
        res: dict[str, Any] = {}
        if do_inf:
            a = np.array(lat[2:]) * 1e3
            res.update(inf_sps=len(lat) * infer_batch / el, inf_p50_ms=float(np.percentile(a, 50)),
                       inf_p99_ms=float(np.percentile(a, 99)))
        if do_train:
            res.update(train_sps=steps[0] * MINIBATCH / el)
        return res

    out = {"inference_alone": run(True, False, False), "training_alone": run(False, True, False),
           "concurrent": run(True, True, False), "concurrent_prio_stream": run(True, True, True),
           "concurrent_prio_switch0.5ms": run(True, True, True, 0.0005)}
    runner.clear()
    return out


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: object) -> None:
        return None


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="2.5m,4m,6.5m")
    ap.add_argument("--buckets", default="48,80,112,160,224")
    ap.add_argument("--batches", default="16,32,64,128,256")
    ap.add_argument("--budgets", default="32768,65536,131072", help="training sub-batch token budgets (B*T_pad)")
    ap.add_argument("--bucket-budget", type=int, default=65536, help="token budget for the per-bucket training rows")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--pool", type=int, default=6000, help="frames in the training-mix pool")
    ap.add_argument("--contention-seconds", type=float, default=8.0)
    ap.add_argument("--infer-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "bench" / "bench_model.json"))
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    presets = a.presets.split(",")
    buckets = [int(x) for x in a.buckets.split(",")]
    batches = [int(x) for x in a.batches.split(",")]
    budgets = [int(x) for x in a.budgets.split(",")]

    vocab = Vocab(make_synthetic_vocab())
    catalog = Catalog(make_synthetic_catalog(vocab, seed=1), vocab)
    gen = FrameGen(vocab, catalog.hash)
    tok = Tokenizer(vocab)
    rng = np.random.default_rng(a.seed)
    print(f"device: {torch.cuda.get_device_name()} torch {torch.__version__}", flush=True)
    t0 = time.perf_counter()
    mix, by = build_pool(tok, gen, a.pool, a.seed)
    tok_lens = np.array([s.n_tokens for s in mix])
    hist = {T: len(v) for T, v in sorted(by.items())}
    print(f"pool built in {time.perf_counter() - t0:.0f}s; mix tokens mean {tok_lens.mean():.1f} p50 "
          f"{np.median(tok_lens):.0f} p99 {np.percentile(tok_lens, 99):.0f}; bucket counts {hist}", flush=True)
    mix_buckets = {T: float(np.mean([tok.config.bucket(int(x)) == T for x in tok_lens])) for T in sorted(by)}
    result: dict[str, Any] = {"mix_bucket_share": mix_buckets, "mix_tokens_mean": float(tok_lens.mean()),
                              "tokenizer": bench_tokenizer(tok, gen, a.seed + 1), "presets": {}}
    print(json.dumps(result["tokenizer"], indent=1), flush=True)

    for name in presets:
        torch.manual_seed(a.seed)
        net = SpireNet(tok.layout, PRESETS[name], catalog).to(DEV)
        res: dict[str, Any] = {"params": net.n_params}
        print(f"\n== preset {name}: {net.n_params / 1e6:.2f}M params", flush=True)
        res["inference"] = bench_inference(net, tok, by, buckets, batches, a.seconds, rng)
        for r in res["inference"]:
            print(f"  infer T={r['bucket']:4d} B={r['batch']:4d}  graph {r['graph_sps']:9.0f}/s ({r['graph_ms']:.2f} ms)"
                  f"  eager {r['eager_sps']:9.0f}/s ({r['eager_ms']:.2f} ms)", flush=True)
        res["training"] = bench_training(net, tok, mix, by, buckets, budgets, a.bucket_budget, a.seconds * 2, rng)
        for r in res["training"]:
            print(f"  train {r['case']:>4s} budget {r['token_budget']:6d}: {r['train_sps']:8.0f} samples/s "
                  f"({r['sub_batches']} sub-batches, {r['step_s'] * 1e3:.0f} ms/step, collate+h2d "
                  f"{r['collate_h2d_s'] * 1e3:.0f} ms, peak {r['peak_mem_gb']:.1f} GB)", flush=True)
        best_budget = max((r for r in res["training"] if r["case"] == "mix"), key=lambda r: r["train_sps"])
        net_train = SpireNet(tok.layout, PRESETS[name], catalog).to(DEV)
        res["contention"] = bench_contention(net, net_train, tok, mix, a.contention_seconds, rng,
                                             best_budget["token_budget"], a.infer_batch)
        for k, v in res["contention"].items():
            print(f"  {k:24s} " + "  ".join(f"{kk}={vv:.1f}" for kk, vv in v.items()), flush=True)
        inf_mix = res["contention"]["inference_alone"]["inf_sps"]
        tr_mix = best_budget["train_sps"]
        conc = max((v for k, v in res["contention"].items() if k.startswith("concurrent")),
                   key=lambda v: min(v["inf_sps"], v["train_sps"] / 2.0))
        res["decisions_per_s_2epochs"] = {
            "serial_model": 1.0 / (1.0 / inf_mix + 2.0 / tr_mix),
            "concurrent_measured": min(conc["inf_sps"], conc["train_sps"] / 2.0),
        }
        print(f"  sustainable decisions/s with 2 PPO epochs: {res['decisions_per_s_2epochs']}", flush=True)
        result["presets"][name] = res
        del net, net_train
        torch.cuda.empty_cache()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

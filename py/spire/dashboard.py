"""Self-contained HTML dashboard from ``metrics.jsonl`` and panel evaluation results.

    python -B -m spire.dashboard --run ../runs/<name> [--out dashboard.html]

One file, no external resources: inline SVG charts, a small inline script for the crosshair /
hover read-outs (all values are also in the tables), light and dark palettes. Regenerating it
reads the metrics once and costs well under a second for days of records (series are
down-sampled to at most ``MAX_POINTS`` points), so a scheduler can rebuild it every few minutes.

Contents (labels in Chinese): headline tiles; rolling training win rate per lineup x ascension
with Wilson 95% bands against decisions and against wall-clock hours, with the legacy baselines
(fast agent Silent A0 32.4% N=256, A10 0%; search stack ~58% pooled N=251) and the targets
(single-player A0 80%; 95% - reached by any one lineup is enough) as reference lines; mean final
floor; throughput (decisions/s, games/hour, per-stage microseconds per decision, latencies);
learner diagnostics (losses, entropy, KL, clip fraction, explained variance); death floors and
top killers per lineup; health counters; GPU utilisation / memory; the panel evaluation table
with Wilson intervals next to the baselines.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .metrics import read_jsonl, wilson_interval

MAX_POINTS = 600
REPO = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO / "eval" / "results"


@dataclass(frozen=True)
class Baseline:
    label: str
    key: str  # lineup|A<asc> it applies to
    rate: float
    n: int | None  # None: sample size not documented

    def interval(self) -> tuple[float, float] | None:
        return None if self.n is None else wilson_interval(round(self.rate * self.n), self.n)


BASELINES = (
    Baseline("旧 fast agent（Silent A0）", "SILENT|A0", 83 / 256, 256),
    Baseline("旧 fast agent（Silent A10）", "SILENT|A10", 0.0, None),
    Baseline("旧搜索栈（Silent A0，合并）", "SILENT|A0", 0.58, 251),
)
TARGETS = ((0.95, "目标 95%（任一阵容达到即可）"), (0.80, "单人 A0 目标 80%"))
SLOTS = 8


# =========================================================================== data


@dataclass
class RunData:
    train: list[dict[str, Any]] = field(default_factory=list)
    learner: list[dict[str, Any]] = field(default_factory=list)
    gpu: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    evals: list[dict[str, Any]] = field(default_factory=list)  # periodic panel evaluations ("eval" records)
    keys: list[str] = field(default_factory=list)  # lineup|A<asc> in order of first appearance

    @classmethod
    def load(cls, metrics_path: Path) -> "RunData":
        d = cls()
        seen: dict[str, None] = {}
        for r in read_jsonl(metrics_path):
            k = r.get("kind")
            if k == "train":
                d.train.append(r)
                for key in r.get("outcomes", {}):
                    seen.setdefault(key, None)
            elif k == "learner":
                d.learner.append(r)
            elif k == "gpu":
                d.gpu.append(r)
            elif k in ("event", "checkpoint"):
                d.events.append(r)
            elif k == "eval":
                d.evals.append(r)
        d.keys = list(seen)
        return d


def load_results(directory: Path, run: str | None = None) -> list[dict[str, Any]]:
    """Summary JSONs of panel evaluations (all, or those of checkpoints of ``run`` plus baselines)."""
    out = []
    for p in sorted(directory.glob("*/*.json")) if directory.exists() else []:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "summary" not in r:
            continue
        pol = r.get("policy", {})
        if run is None or pol.get("run") in (run, None):
            r["_path"] = str(p)
            out.append(r)
    return out


def _thin(n: int) -> list[int]:
    if n <= MAX_POINTS:
        return list(range(n))
    step = n / MAX_POINTS
    idx = sorted({int(i * step) for i in range(MAX_POINTS)} | {n - 1})
    return idx


# =========================================================================== formatting


def compact(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    a = abs(x)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{x / div:.{digits}f}{suf}"
    return f"{x:.0f}" if a >= 100 or float(x).is_integer() else f"{x:.{digits + 1}g}"


def pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9 * step:
        out.append(round(t, 10))
        t += step
    return out


# =========================================================================== charts


@dataclass
class Series:
    name: str
    slot: int  # 1-based categorical slot
    xs: list[float]
    ys: list[float | None]
    lo: list[float | None] | None = None
    hi: list[float | None] | None = None


W, H = 720, 260
PAD_L, PAD_R, PAD_T, PAD_B = 56, 16, 14, 40


class _Ids:
    n = 0

    @classmethod
    def next(cls, prefix: str) -> str:
        cls.n += 1
        return f"{prefix}{cls.n}"


def line_chart(title: str, series: Sequence[Series], *, x_label: str, y_label: str = "",
               y_fmt: Callable[[float], str] = compact, x_fmt: Callable[[float], str] = compact,
               y_range: tuple[float | None, float | None] = (None, None),
               refs: Sequence[tuple[float, str]] = (), subtitle: str = "") -> str:
    """SVG line chart with optional bands, horizontal reference lines, legend and hover read-out."""
    cid = _Ids.next("c")
    pts = [(x, y) for s in series for x, y in zip(s.xs, s.ys) if y is not None and math.isfinite(y)]
    if not pts:
        return _card(title, subtitle, '<p class="empty">暂无数据</p>')
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts] + [r[0] for r in refs]
    for s in series:
        ys += [v for v in (s.lo or []) + (s.hi or []) if v is not None]
    x0, x1 = min(xs), max(xs)
    if x1 <= x0:
        x1 = x0 + 1
    y0 = y_range[0] if y_range[0] is not None else min(ys)
    y1 = y_range[1] if y_range[1] is not None else max(ys)
    if y1 <= y0:
        y1 = y0 + (abs(y0) or 1.0)
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def sx(x: float) -> float:
        return PAD_L + (x - x0) / (x1 - x0) * pw

    def sy(y: float) -> float:
        return PAD_T + (1 - (min(max(y, y0), y1) - y0) / (y1 - y0)) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" id="{cid}" role="img" aria-label="{esc(title)}">']
    for t in _nice_ticks(y0, y1):
        parts.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                     f'<text class="tick" x="{PAD_L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{esc(y_fmt(t))}</text>')
    for t in _nice_ticks(x0, x1, 6):
        parts.append(f'<text class="tick" x="{sx(t):.1f}" y="{H - PAD_B + 16}" text-anchor="middle">'
                     f'{esc(x_fmt(t))}</text>')
    parts.append(f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{PAD_T + ph}" y2="{PAD_T + ph}"/>')
    parts.append(f'<text class="axlabel" x="{PAD_L + pw / 2}" y="{H - 6}" text-anchor="middle">{esc(x_label)}</text>')
    if y_label:
        parts.append(f'<text class="axlabel" transform="translate(12 {PAD_T + ph / 2}) rotate(-90)" '
                     f'text-anchor="middle">{esc(y_label)}</text>')
    for y, label in refs:
        if y0 <= y <= y1:
            parts.append(f'<line class="ref" x1="{PAD_L}" x2="{W - PAD_R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}"/>'
                         f'<text class="reflabel" x="{PAD_L + 6}" y="{sy(y) - 4:.1f}">'
                         f'{esc(label)}</text>')
    data = []
    for s in series:
        idx = _thin(len(s.xs))
        if s.lo is not None and s.hi is not None:
            up = [(sx(s.xs[i]), sy(h)) for i in idx if (h := s.hi[i]) is not None and s.ys[i] is not None]
            dn = [(sx(s.xs[i]), sy(lo)) for i in idx if (lo := s.lo[i]) is not None and s.ys[i] is not None]
            if len(up) > 1:
                poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in up + dn[::-1])
                parts.append(f'<polygon class="band s{s.slot}" points="{poly}"/>')
        seg: list[str] = []
        for i in idx:
            yv = s.ys[i]
            if yv is None or not math.isfinite(yv):
                if len(seg) > 1:
                    parts.append(f'<polyline class="line s{s.slot}" points="{" ".join(seg)}"/>')
                seg = []
                continue
            seg.append(f"{sx(s.xs[i]):.1f},{sy(yv):.1f}")
        if len(seg) > 1:
            parts.append(f'<polyline class="line s{s.slot}" points="{" ".join(seg)}"/>')
        last = next((i for i in reversed(idx) if s.ys[i] is not None), None)
        if last is not None:
            ly = s.ys[last]
            assert ly is not None
            parts.append(f'<circle class="dot s{s.slot}" cx="{sx(s.xs[last]):.1f}" cy="{sy(ly):.1f}" r="4"/>')
        data.append({"name": s.name, "slot": s.slot, "x": [s.xs[i] for i in idx],
                     "y": [None if s.ys[i] is None else round(float(s.ys[i]), 5) for i in idx],  # type: ignore[arg-type]
                     "lo": None if s.lo is None else [s.lo[i] for i in idx],
                     "hi": None if s.hi is None else [s.hi[i] for i in idx]})
    parts.append(f'<line class="cross" x1="0" x2="0" y1="{PAD_T}" y2="{PAD_T + ph}" visibility="hidden"/>')
    parts.append(f'<rect class="hit" x="{PAD_L}" y="{PAD_T}" width="{pw}" height="{ph}"/>')
    parts.append("</svg>")
    meta = {"x0": x0, "x1": x1, "pl": PAD_L, "pw": pw, "W": W, "pct": y_fmt is pct, "series": data}
    legend = "" if len(series) < 2 else '<div class="legend">' + "".join(
        f'<span><i class="key s{s.slot}"></i>{esc(s.name)}</span>' for s in series) + "</div>"
    body = (legend + '<div class="plot">' + "".join(parts) + '<div class="tip" hidden></div></div>'
            + f'<script type="application/json" class="chart-data" data-for="{cid}">'
            + json.dumps(meta, ensure_ascii=False).replace("<", "\\u003c") + "</script>")
    return _card(title, subtitle, body)


def bar_chart(title: str, labels: Sequence[str], values: Sequence[float], *, subtitle: str = "",
              fmt: Callable[[float], str] = compact, horizontal: bool = False, unit: str = "") -> str:
    """Single-series bar chart (slot 1); every bar has a hover title, values in the table below."""
    if not values:
        return _card(title, subtitle, '<p class="empty">暂无数据</p>')
    vmax = max(max(values), 1e-12)
    parts = []
    if horizontal:
        row = 22
        h = row * len(values) + 10
        lw = 120
        parts.append(f'<svg viewBox="0 0 {W} {h}" class="chart" role="img" aria-label="{esc(title)}">')
        for i, (lab, v) in enumerate(zip(labels, values)):
            y = 5 + i * row
            bw = max(0.0, v / vmax * (W - lw - 90))
            parts.append(f'<text class="tick" x="{lw - 8}" y="{y + 14}" text-anchor="end">{esc(lab)}</text>'
                         f'<rect class="bar s1" x="{lw}" y="{y + 3}" width="{bw:.1f}" height="14" rx="3">'
                         f'<title>{esc(lab)}: {esc(fmt(v))}{esc(unit)}</title></rect>'
                         f'<text class="val" x="{lw + bw + 6:.1f}" y="{y + 14}">{esc(fmt(v))}{esc(unit)}</text>')
    else:
        h = 200
        n = len(values)
        band = (W - PAD_L - PAD_R) / n
        bw = min(24.0, band - 2)
        ph = h - 40
        parts.append(f'<svg viewBox="0 0 {W} {h}" class="chart" role="img" aria-label="{esc(title)}">')
        for t in _nice_ticks(0, vmax, 4):
            gy = 10 + (1 - t / vmax) * ph
            parts.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{gy:.1f}" y2="{gy:.1f}"/>'
                         f'<text class="tick" x="{PAD_L - 6}" y="{gy + 4:.1f}" text-anchor="end">{esc(fmt(t))}</text>')
        step = max(1, math.ceil(n / 24))
        for i, (lab, v) in enumerate(zip(labels, values)):
            x = PAD_L + i * band + (band - bw) / 2
            bh = v / vmax * ph
            parts.append(f'<rect class="bar s1" x="{x:.1f}" y="{10 + ph - bh:.1f}" width="{bw:.1f}" '
                         f'height="{bh:.1f}" rx="3"><title>{esc(lab)}: {esc(fmt(v))}{esc(unit)}</title></rect>')
            if i % step == 0:
                parts.append(f'<text class="tick" x="{x + bw / 2:.1f}" y="{h - 14}" text-anchor="middle">{esc(lab)}</text>')
        parts.append(f'<line class="axis" x1="{PAD_L}" x2="{W - PAD_R}" y1="{10 + ph}" y2="{10 + ph}"/>')
    parts.append("</svg>")
    return _card(title, subtitle, "".join(parts))


def _card(title: str, subtitle: str, body: str) -> str:
    sub = f'<p class="sub">{esc(subtitle)}</p>' if subtitle else ""
    return f'<section class="card"><h3>{esc(title)}</h3>{sub}{body}</section>'


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], caption: str = "") -> str:
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def tile(label: str, value: str, note: str = "") -> str:
    n = f'<div class="note">{esc(note)}</div>' if note else ""
    return f'<div class="tile"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>{n}</div>'


# =========================================================================== page


def _refs_for(keys: Sequence[str]) -> list[tuple[float, str]]:
    refs = [(t, lab) for t, lab in TARGETS]
    for b in BASELINES:
        if b.key in keys and b.rate > 0:
            refs.append((b.rate, f"{b.label} {pct(b.rate)}"))
    return refs


ALARM_RULES = (
    "致命：运行因错误停止（FAILED.json / fatal 事件）——按 notes/l1-runbook.md 续训",
    "学习器瓶颈：actor 被背压节流 >20% 的时间，或有样本因陈旧 / 积压被丢弃",
    "策略滞后：数据进入更新时的年龄中位数 >60 s，或平均陈旧度 >2 个版本",
    "epoch-0 裁剪比例 >2%：数据陈旧或学习率过大",
    "非有限损失 / 梯度（跳过的 minibatch）或选中非法动作：数值或协议 bug",
    "ERROR 帧 / 溢出局：协议或编码器 bug（errors.jsonl 有复现信息）",
    "宿主超时 / 崩溃 / actor 重启：宿主或机器不稳定",
    "吞吐下降：最新区间决策/s 低于全程中位数的 60%",
    "面板评估超预算：评估决策占比超过 eval.max_fraction 的 1.5 倍",
    "看板数据停更：最新训练记录早于 5 个指标区间（运行已停或卡住）",
    "GPU 温度 >85°C",
)


def alarms(d: "RunData", run_dir: Path) -> list[tuple[str, str]]:
    """(level, message) of the alarm rules (:data:`ALARM_RULES`) that fire on the latest records."""
    out: list[tuple[str, str]] = []
    tr, L = d.train, d.learner
    last = tr[-1] if tr else {}
    if (run_dir / "FAILED.json").exists() or any(e.get("what") == "fatal" for e in d.events):
        try:
            reason = json.loads((run_dir / "FAILED.json").read_text(encoding="utf-8")).get("reason", "")
        except (OSError, json.JSONDecodeError):
            reason = next((e.get("reason", "") for e in reversed(d.events) if e.get("what") == "fatal"), "")
        out.append(("fatal", f"运行因错误停止：{reason}"))
    recent = tr[-6:]
    thr = [r.get("throttled_frac") or 0.0 for r in recent]
    if thr and max(thr) > 0.2:
        out.append(("warn", f"学习器瓶颈：actor 被背压节流 {max(thr):.0%} 的时间（最近区间最大值）"))
    if L:
        ring = L[-1].get("ring") or {}
        dropped = int(ring.get("evicted_samples", 0)) + int(ring.get("stale_dropped_samples", 0))
        if dropped:
            out.append(("warn", f"学习器瓶颈：累计 {dropped} 个样本因积压 / 陈旧被丢弃"))
        lag = L[-1].get("lag_s") or {}
        if (lag.get("p50") or 0) > 60 or (L[-1].get("staleness_mean") or 0) > 2:
            out.append(("warn", f"策略滞后：数据年龄中位数 {lag.get('p50', 0):.0f} s，平均陈旧度 "
                                f"{L[-1].get('staleness_mean', 0):.2f} 版本"))
        c0 = [r.get("clipfrac_epoch0") or 0.0 for r in L[-5:]]
        if c0 and min(c0) > 0.02:
            out.append(("warn", f"epoch-0 裁剪比例 {min(c0):.1%}（最近 5 次更新都 >2%）"))
        nf = sum(int(r.get("nonfinite_skipped") or 0) for r in L)
        ill = sum(int(r.get("illegal_chosen") or 0) for r in L)
        if nf or ill:
            out.append(("fatal" if ill else "warn", f"非有限 minibatch 跳过 {nf} 次，选中非法动作 {ill} 次"))
    counts = last.get("counts") or {}
    if counts.get("error") or counts.get("overflow") or counts.get("errors"):
        out.append(("warn", f"ERROR 帧 {counts.get('errors', 0)}、错误局 {counts.get('error', 0)}、溢出局 "
                            f"{counts.get('overflow', 0)}（errors.jsonl）"))
    hosty = {k: counts.get(k, 0) for k in ("timeouts", "crashes", "actor_restarts", "lost") if counts.get(k)}
    if hosty:
        out.append(("warn", "宿主 / 进程不稳定：" + "，".join(f"{k} {v}" for k, v in hosty.items())))
    rates = [r.get("decisions_per_s") or 0.0 for r in tr]
    if len(rates) >= 6:
        med = float(sorted(rates)[len(rates) // 2])
        if med > 0 and rates[-1] < 0.6 * med:
            out.append(("warn", f"吞吐下降：最新 {rates[-1]:.0f} 决策/s，全程中位数 {med:.0f}"))
    if d.evals and last.get("eval_fraction") is not None:
        cfg_path = run_dir / "config.json"
        try:
            mf = json.loads(cfg_path.read_text(encoding="utf-8")).get("eval", {}).get("max_fraction")
        except (OSError, json.JSONDecodeError):
            mf = None
        if mf and last["eval_fraction"] > 1.5 * mf:
            out.append(("warn", f"面板评估超预算：评估决策占比 {last['eval_fraction']:.1%} > 1.5 × {mf:.1%}"))
    if last.get("t") and (run_dir / "config.json").exists():
        try:
            interval = float(json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get(
                "metrics_interval_s", 30.0))
        except (OSError, json.JSONDecodeError, ValueError):
            interval = 30.0
        stopped = any(e.get("what") == "stop" for e in d.events[-3:])
        age = time.time() - float(last["t"])
        if not stopped and age > 5 * interval:
            out.append(("warn", f"看板数据停更：最新训练记录是 {age / 60:.0f} 分钟前（运行已停或卡住？）"))
    if d.gpu:
        temp = d.gpu[-1]["gpus"][0].get("temp_c") or 0
        if temp > 85:
            out.append(("warn", f"GPU 温度 {temp:.0f}°C"))
    return out


def build_html(run_dir: Path, result_dir: Path = RESULT_DIR) -> str:
    _Ids.n = 0
    d = RunData.load(run_dir / "metrics.jsonl")
    results = load_results(result_dir, run_dir.name)
    keys = d.keys[:SLOTS]
    slot = {k: i + 1 for i, k in enumerate(keys)}
    tr = d.train
    last = tr[-1] if tr else {}
    sections: list[str] = []

    # ---- headline tiles
    best_train = None
    for k, o in (last.get("outcomes") or {}).items():
        if o.get("n"):
            if best_train is None or o["lo"] > best_train[1]["lo"]:
                best_train = (k, o)
    best_eval = None
    for r in results:
        for k, s in r["summary"].items():
            if k != "all" and s.get("n") and (best_eval is None or s["lower_one_sided"] > best_eval[2]["lower_one_sided"]):
                best_eval = (r["tag"], k, s)
    tiles = [
        tile("累计决策", compact(last.get("decisions")), f"版本 v{last.get('version', 0)}，更新 {last.get('updates', 0)} 次"),
        tile("吞吐（决策/秒）", compact(last.get("decisions_per_s")), f"局/小时 {compact(last.get('games_per_hour'))}"),
        tile("训练局数（完赛）", compact(last.get("games")), f"运行 {(last.get('elapsed') or 0) / 3600:.2f} 小时"),
        tile("训练滚动胜率最佳阵容", pct(best_train[1]["win_rate"]) if best_train else "–",
             f"{best_train[0]}，Wilson 95% [{pct(best_train[1]['lo'])}, {pct(best_train[1]['hi'])}]，N={best_train[1]['n']}"
             if best_train else "尚无完赛局"),
        tile("面板评估最佳下界（单侧 95%）", pct(best_eval[2]["lower_one_sided"]) if best_eval else "–",
             f"{best_eval[1]} @ {best_eval[0]}，{best_eval[2]['wins']}/{best_eval[2]['n']}；目标 >95% 任一阵容即可"
             if best_eval else "尚无面板评估"),
    ]
    counts = last.get("counts", {})
    tiles.append(tile("异常局（作废/丢失/错误/溢出）",
                      f"{counts.get('aborted', 0)}/{counts.get('lost', 0)}/{counts.get('error', 0)}/{counts.get('overflow', 0)}",
                      f"宿主超时 {counts.get('timeouts', 0)}，崩溃 {counts.get('crashes', 0)}，"
                      f"actor 重启 {counts.get('actor_restarts', 0)}"))
    sections.append('<div class="tiles">' + "".join(tiles) + "</div>")
    fired = alarms(d, run_dir)
    sections.append("<h2>告警</h2>" + (table(["级别", "告警"], [("致命" if lv == "fatal" else "注意", m) for lv, m in fired],
                                             "规则见下表；按 notes/l1-runbook.md 处理")
                                       if fired else '<p class="empty">无告警</p>')
                    + table(["告警规则"], [(r,) for r in ALARM_RULES], ""))

    # ---- sampling weights and compute per configuration
    samp = last.get("sampling") or {}
    outs = last.get("outcomes") or {}
    if samp or outs:
        w = {lu: wt for lu, wt in samp.get("lineups", [])}
        wsum = sum(w.values()) or 1.0
        srows = []
        for k, o in outs.items():
            lu = k.rsplit("|", 1)[0]
            srows.append((k, pct(w[lu] / wsum) if lu in w else "（已移出）", pct(o.get("decision_share")),
                          compact(o.get("decisions")), o.get("total_finished", 0), pct(o.get("win_rate")),
                          f"[{pct(o.get('lo'))}, {pct(o.get('hi'))}]"))
        sections.append("<h2>采样权重与算力分配</h2>" + table(
            ["阵容|进阶", "当前采样权重", "累计决策占比", "累计决策", "累计完赛", "滚动胜率", "Wilson 95%"], srows,
            f"采样权重文件版本 v{samp.get('version', '–')}（{esc(samp.get('source', ''))}）；"
            "改 sampling.json 即可调整，见 runbook") + '<div class="grid2">' + line_chart(
            "各配置累计决策占比", [Series(k, slot[k], [r["decisions"] for r in tr if k in (r.get("outcomes") or {})],
                                    [r["outcomes"][k].get("decision_share") for r in tr if k in (r.get("outcomes") or {})])
                             for k in keys], x_label="累计决策数", y_fmt=pct, y_range=(0.0, 1.0),
            subtitle="算力在各阵容间的分配（随采样权重变化）") + "</div>")

    # ---- win rate / floor curves
    def outcome_series(xkey: Callable[[dict[str, Any]], float], field_: str, band: bool) -> list[Series]:
        out = []
        for k in keys:
            xs, ys, lo, hi = [], [], [], []
            for r in tr:
                o = r.get("outcomes", {}).get(k)
                if not o or not o.get("n"):
                    continue
                xs.append(xkey(r))
                ys.append(o.get(field_))
                lo.append(o.get("lo"))
                hi.append(o.get("hi"))
            if xs:
                out.append(Series(k, slot[k], xs, ys, lo if band else None, hi if band else None))
        return out

    refs = _refs_for(keys)
    win_note = f"滚动窗口：最近 {tr[-1]['outcomes'][keys[0]]['n'] if tr and keys and keys[0] in tr[-1]['outcomes'] else '–'} 局完赛训练局（采样策略），阴影 = Wilson 95%"
    sections.append('<h2>学习曲线</h2><div class="grid2">' + line_chart(
        "训练胜率 vs 决策数", outcome_series(lambda r: r["decisions"], "win_rate", True), x_label="累计决策数",
        y_fmt=pct, y_range=(0.0, 1.0), refs=refs, subtitle=win_note) + line_chart(
        "训练胜率 vs 墙钟时间", outcome_series(lambda r: r["elapsed"] / 3600, "win_rate", True), x_label="小时",
        x_fmt=lambda h: f"{h:.2g}", y_fmt=pct, y_range=(0.0, 1.0), refs=refs, subtitle="同上，横轴为运行时长") + line_chart(
        "平均到达楼层 vs 决策数", outcome_series(lambda r: r["decisions"], "mean_floor", False), x_label="累计决策数",
        y_label="楼层", subtitle="滚动窗口内完赛局的最终楼层均值") + "</div>")

    # ---- throughput
    def scalar(rows: Sequence[dict[str, Any]], x: Callable[[dict[str, Any]], float], f: Callable[[dict[str, Any]], Any],
               name: str, s: int = 1) -> Series:
        return Series(name, s, [x(r) for r in rows], [f(r) for r in rows])

    hours = lambda r: r["elapsed"] / 3600  # noqa: E731
    hfmt = lambda h: f"{h:.2g}"  # noqa: E731
    stage = last.get("stage_us") or {}
    stage_rows = [(k, v) for k, v in stage.items() if k not in ("python",)]
    sections.append('<h2>吞吐与开销</h2><div class="grid2">' + line_chart(
        "整机决策/秒", [scalar(tr, hours, lambda r: r.get("decisions_per_s"), "决策/秒")], x_label="小时", x_fmt=hfmt,
        subtitle="每个指标区间内全部 actor 的决策数 / 区间时长") + line_chart(
        "局/小时", [scalar(tr, hours, lambda r: r.get("games_per_hour"), "局/小时")], x_label="小时", x_fmt=hfmt) + line_chart(
        "背压与积压", [scalar(tr, hours, lambda r: r.get("throttled_frac"), "actor 节流时间占比", 1),
                    scalar(tr, hours, lambda r: (r.get("backlog_samples") or 0) / max(1, (r.get("decisions_per_s") or 1) * 10),
                           "积压 / (10 s 产出)", 2)],
        x_label="小时", x_fmt=hfmt, y_fmt=lambda v: f"{v:.2g}",
        subtitle="节流 >0 说明学习器跟不上采集（数据保持新鲜，宿主空等）") + line_chart(
        "延迟（毫秒）", [scalar(tr, hours, lambda r: r.get("infer_ms_p50"), "推理 p50", 1),
                      scalar(tr, hours, lambda r: r.get("infer_ms_p99"), "推理 p99", 2),
                      scalar(tr, hours, lambda r: r.get("host_latency_ms_p50"), "宿主回复 p50", 3),
                      scalar(tr, hours, lambda r: r.get("actor_latency_ms_p99"), "帧到动作 p99", 7)],
        x_label="小时", x_fmt=hfmt, y_fmt=lambda v: f"{v:.2g}") + bar_chart(
        "actor 每决策各阶段耗时（最新区间）", [k for k, _ in stage_rows], [v for _, v in stage_rows], horizontal=True,
        unit=" µs", fmt=lambda v: f"{v:.1f}",
        subtitle=f"Python 侧合计 {stage.get('python', 0):.1f} µs/决策（wait=等待宿主，gpu=推理+采样）") + "</div>")

    # ---- learner
    L = d.learner
    upd = lambda r: r.get("updates", 0)  # noqa: E731
    f3 = lambda v: f"{v:.3g}"  # noqa: E731
    sections.append('<h2>学习器诊断</h2><div class="grid2">' + "".join([
        line_chart("策略损失", [scalar(L, upd, lambda r: r.get("loss_pi"), "loss_pi")], x_label="更新次数", y_fmt=f3),
        line_chart("价值损失", [scalar(L, upd, lambda r: r.get("loss_v"), "loss_v")], x_label="更新次数", y_fmt=f3),
        line_chart("熵", [scalar(L, upd, lambda r: r.get("entropy"), "熵"),
                         scalar(L, upd, lambda r: r.get("entropy_norm"), "归一化熵", 2)], x_label="更新次数", y_fmt=f3),
        line_chart("近似 KL", [scalar(L, upd, lambda r: r.get("approx_kl"), "approx_kl")], x_label="更新次数", y_fmt=f3),
        line_chart("裁剪比例", [scalar(L, upd, lambda r: r.get("clipfrac"), "全部 epoch"),
                            scalar(L, upd, lambda r: r.get("clipfrac_epoch0"), "epoch 0（陈旧度指示，应 <2%）", 2)],
                   x_label="更新次数", y_fmt=pct),
        line_chart("解释方差", [scalar(L, upd, lambda r: r.get("explained_variance"), "EV")], x_label="更新次数",
                   y_fmt=f3),
        line_chart("策略陈旧度（版本差）", [scalar(L, upd, lambda r: r.get("staleness_mean"), "均值"),
                                  scalar(L, upd, lambda r: r.get("staleness_max"), "最大", 2)], x_label="更新次数",
                   y_fmt=f3),
        line_chart("学习器耗时（秒/迭代）", [scalar(L, upd, lambda r: r.get("time_train"), "训练"),
                                   scalar(L, upd, lambda r: r.get("wait_s"), "等待样本", 2)], x_label="更新次数",
                   y_fmt=f3),
        line_chart("策略滞后（秒）", [scalar(L, upd, lambda r: (r.get("lag_s") or {}).get("p50"), "数据年龄 p50"),
                                 scalar(L, upd, lambda r: (r.get("lag_s") or {}).get("p90"), "p90", 2)],
                   x_label="更新次数", y_fmt=f3, subtitle="决策发生到被用于更新的时间"),
        line_chart("学习器吞吐（样本·epoch/秒）", [scalar(L, upd, lambda r: r.get("samples_per_s"), "训练时")],
                   x_label="更新次数", y_fmt=compact),
    ]) + "</div>")
    if L:
        lk = L[-1].get("entropy_by_kind", {})
        sections.append(table(["决策种类", "熵", "归一化熵", "样本数"],
                              [(k, f"{v['entropy']:.3f}", f"{v['entropy_norm']:.3f}", v["samples"]) for k, v in lk.items()],
                              "最新一次更新：各决策种类的策略熵"))

    # ---- deaths
    death_cards = []
    for k in keys:
        o = (last.get("outcomes") or {}).get(k)
        if not o or not o.get("death_floor"):
            continue
        floors = sorted((int(f), c) for f, c in o["death_floor"].items())
        lo_f, hi_f = floors[0][0], floors[-1][0]
        cnt = dict(floors)
        death_cards.append(bar_chart(f"死亡楼层分布：{k}", [str(f) for f in range(lo_f, hi_f + 1)],
                                     [cnt.get(f, 0) for f in range(lo_f, hi_f + 1)],
                                     subtitle=f"滚动窗口内 {sum(cnt.values())} 局败局"))
        if o.get("top_killers"):
            death_cards.append(_card(f"致死遭遇 Top：{k}", "最后一场战斗的敌人组合",
                                     table(["敌人", "败局数"], [(a or "（非战斗）", b) for a, b in o["top_killers"]])))
    if death_cards:
        sections.append('<h2>死亡位置</h2><div class="grid2">' + "".join(death_cards) + "</div>")

    # ---- latest outcome table (table view of the curves)
    if last.get("outcomes"):
        rows = []
        for k, o in last["outcomes"].items():
            rows.append((k, o["n"], o["wins"], pct(o["win_rate"]), f"[{pct(o['lo'])}, {pct(o['hi'])}]",
                         "–" if o["mean_floor"] is None else f"{o['mean_floor']:.1f}", o["total_finished"],
                         ", ".join(f"{a}:{b}" for a, b in o.get("failures", {}).items()) or "0"))
        sections.append(table(["阵容|进阶", "窗口局数", "胜", "胜率", "Wilson 95%", "平均楼层", "累计完赛", "异常局"],
                              rows, "训练滚动胜率（最新，曲线的表格视图）"))

    # ---- panel evaluations + baselines
    sections.append("<h2>固定种子面板评估</h2>")
    if d.evals:
        ekeys = sorted({k for e in d.evals for k in (e.get("summary") or {}) if k != "all"})
        eslot = {k: slot.get(k, (i % SLOTS) + 1) for i, k in enumerate(ekeys)}
        sections.append('<div class="grid2">' + line_chart(
            "面板胜率 vs 版本（训练中周期评估，EMA 贪心）",
            [Series(k, eslot[k], [e["version"] for e in d.evals if k in e["summary"]],
                    [e["summary"][k]["win_rate"] for e in d.evals if k in e["summary"]],
                    [e["summary"][k]["lo"] for e in d.evals if k in e["summary"]],
                    [e["summary"][k]["hi"] for e in d.evals if k in e["summary"]]) for k in ekeys],
            x_label="权重版本", y_fmt=pct, y_range=(0.0, 1.0), refs=_refs_for(ekeys),
            subtitle="每轮受时间预算限制；未完成轮次见下表 partial；阴影 = Wilson 95%；评估局不进训练数据") + "</div>")
    erows = []
    for r in results:
        pol = r.get("policy", {})
        # Earlier result files predate round-completion metadata; display that fact explicitly.
        round_status = (f"{'partial' if r['partial'] else 'complete'} "
                        f"{r['completed_episodes']}/{r['expected_episodes']} ({r['stop_reason']})"
                        if "partial" in r else "legacy: completion not recorded")
        for k, s in r["summary"].items():
            erows.append((r["panel"], r["tag"], round_status, pol.get("version", "–"),
                          pol.get("mode", pol.get("kind", "")), k,
                          s["n"], s["wins"], pct(s["win_rate"]), f"[{pct(s['lo'])}, {pct(s['hi'])}]",
                          pct(s["lower_one_sided"]),
                          ", ".join(f"{a}:{b}" for a, b in s.get("failures", {}).items()) or "0"))
    sections.append(table(["面板", "评估", "轮次状态（完成/计划）", "版本", "策略", "阵容|进阶", "局数", "胜", "胜率", "Wilson 95%",
                           "单侧 95% 下界", "失败局(计入分母)"], erows,
                          "每个种子每个配置只打一次；失败局按负计") if erows else '<p class="empty">暂无面板评估结果</p>')
    brows = []
    for b in BASELINES:
        ci = b.interval()
        brows.append((b.label, b.key, pct(b.rate), b.n or "未记录", "–" if ci is None else f"[{pct(ci[0])}, {pct(ci[1])}]"))
    sections.append(table(["基线", "阵容|进阶", "胜率", "N", "Wilson 95%"], brows, "旧栈基线（未配对种子，仅作参照）"))

    # ---- health and GPU
    G = d.gpu
    gx = lambda r: r["elapsed"] / 3600  # noqa: E731
    sections.append('<h2>运行健康</h2><div class="grid2">' + line_chart(
        "GPU 利用率", [Series("利用率", 1, [gx(r) for r in G], [r["gpus"][0].get("util") / 100 for r in G])],
        x_label="小时", x_fmt=hfmt, y_fmt=pct, y_range=(0.0, 1.0), subtitle="nvidia-smi 采样（含其他程序）") + line_chart(
        "GPU 显存（MB）", [Series("已用显存", 1, [gx(r) for r in G], [r["gpus"][0].get("mem_used_mb") for r in G])],
        x_label="小时", x_fmt=hfmt) + "</div>")
    if counts:
        sections.append(table(["计数", "值"], sorted(counts.items()), "累计异常与恢复计数（无静默丢弃）"))
    ev = [(f"{(e.get('elapsed') or 0) / 3600:.3f} h", e.get("what", e.get("kind")),
           json.dumps({k: v for k, v in e.items() if k not in ("kind", "t", "elapsed", "what")}, ensure_ascii=False))
          for e in d.events[-30:]]
    if ev:
        sections.append(table(["时间", "事件", "详情"], ev, "最近事件"))

    title = f"训练看板 · {run_dir.name}"
    gen = last.get("t")
    return _PAGE.replace("__TITLE__", esc(title)).replace("__BODY__", "\n".join(sections)).replace(
        "__META__", esc(f"生成自 {run_dir / 'metrics.jsonl'}；{len(tr)} 条训练记录、{len(L)} 次更新"
                        + (f"；最新记录 t={gen}" if gen else "")))


def write_dashboard(run_dir: str | Path, out: str | Path | None = None, result_dir: Path = RESULT_DIR) -> Path:
    run = Path(run_dir)
    path = Path(out) if out else run / "dashboard.html"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(build_html(run, result_dir), encoding="utf-8")
    tmp.replace(path)
    return path


_CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}
body{margin:0;background:var(--page)}
.viz-root{font:14px system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:var(--ink);background:var(--page);
padding:16px clamp(16px,3vw,32px) 48px;max-width:1500px;margin:0 auto}
h1{font-size:22px;margin:4px 0}h2{font-size:17px;margin:28px 0 10px}h3{font-size:14px;margin:0 0 2px}
.meta,.sub,.note,.empty,caption{color:var(--ink2);font-size:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:12px}
.tile,.card{background:var(--surface-1);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}
.tile .label{color:var(--ink2);font-size:12px}.tile .value{font-size:26px;font-weight:600;margin:4px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:12px}
.plot{position:relative}svg.chart{width:100%;height:auto;display:block}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.tick,.axlabel,.val{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.val{fill:var(--ink2)}
.ref{stroke:var(--ink2);stroke-width:1;stroke-dasharray:4 4;opacity:.7}.reflabel{fill:var(--ink2);font-size:11px}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:var(--surface-1);stroke-width:2}.band{opacity:.12;stroke:none}.bar{stroke:none}
.cross{stroke:var(--axis);stroke-width:1}.hit{fill:transparent;cursor:crosshair}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:6px 0;color:var(--ink2);font-size:12px}
.key{display:inline-block;width:14px;height:2px;margin-right:6px;vertical-align:middle;border-radius:1px}
.tip{position:absolute;pointer-events:none;background:var(--surface-1);border:1px solid var(--ring);border-radius:8px;
padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);white-space:nowrap;z-index:2}
.tip b{font-weight:600;color:var(--ink)}.tip .row{display:flex;align-items:center;gap:6px;color:var(--ink2)}
.tablewrap{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;font-size:12px;background:var(--surface-1);
min-width:100%}th,td{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:left;font-variant-numeric:tabular-nums}
th{color:var(--ink2);font-weight:600}caption{text-align:left;padding:4px 0}
""" + "".join(f".s{i}{{stroke:var(--s{i})}}.band.s{i},.bar.s{i},.dot.s{i},.key.s{i}{{fill:var(--s{i});background:var(--s{i})}}"
              for i in range(1, SLOTS + 1))

_JS = """
(function(){
function fmt(v,p){if(v===null||v===undefined)return'–';if(p)return(100*v).toFixed(1)+'%';
var a=Math.abs(v);if(a>=1e6)return(v/1e6).toFixed(2)+'M';if(a>=1e3)return(v/1e3).toFixed(1)+'K';
return a>=100?v.toFixed(0):v.toPrecision(3);}
function near(xs,x){var lo=0,hi=xs.length-1;if(hi<0)return-1;while(hi-lo>1){var m=(lo+hi)>>1;if(xs[m]<x)lo=m;else hi=m;}
return Math.abs(xs[lo]-x)<=Math.abs(xs[hi]-x)?lo:hi;}
document.querySelectorAll('script.chart-data').forEach(function(sc){
 var meta=JSON.parse(sc.textContent),svg=document.getElementById(sc.dataset.for);if(!svg)return;
 var plot=svg.parentNode,tip=plot.querySelector('.tip'),cross=svg.querySelector('.cross'),hit=svg.querySelector('.hit');
 function show(ev){var r=svg.getBoundingClientRect(),k=meta.W/r.width,px=(ev.clientX-r.left)*k;
  var x=meta.x0+(px-meta.pl)/meta.pw*(meta.x1-meta.x0);var best=null;
  while(tip.firstChild)tip.removeChild(tip.firstChild);
  meta.series.forEach(function(s){var i=near(s.x,x);if(i<0)return;if(best===null)best=s.x[i];
   var row=document.createElement('div');row.className='row';var key=document.createElement('i');
   key.className='key s'+s.slot;row.appendChild(key);var b=document.createElement('b');var t=fmt(s.y[i],meta.pct);
   if(s.lo&&s.lo[i]!==null)t+='  ['+fmt(s.lo[i],meta.pct)+', '+fmt(s.hi[i],meta.pct)+']';b.textContent=t;
   row.appendChild(b);var n=document.createElement('span');n.textContent=s.name;row.appendChild(n);tip.appendChild(row);});
  if(best===null)return;var head=document.createElement('div');head.textContent='x = '+fmt(best,false);
  tip.insertBefore(head,tip.firstChild);var cx=meta.pl+(best-meta.x0)/(meta.x1-meta.x0)*meta.pw;
  cross.setAttribute('x1',cx);cross.setAttribute('x2',cx);cross.setAttribute('visibility','visible');
  tip.hidden=false;var left=cx/k+12;if(left+tip.offsetWidth>r.width)left=cx/k-tip.offsetWidth-12;
  tip.style.left=Math.max(0,left)+'px';tip.style.top='8px';}
 hit.addEventListener('pointermove',show);
 hit.addEventListener('pointerleave',function(){tip.hidden=true;cross.setAttribute('visibility','hidden');});
});})();
"""

_PAGE = ("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
         "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>__TITLE__</title><style>"
         + _CSS + "</style></head><body><div class=\"viz-root\"><h1>__TITLE__</h1><p class=\"meta\">__META__</p>"
         "__BODY__</div><script>" + _JS + "</script></body></html>")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build the HTML dashboard of a run")
    ap.add_argument("--run", required=True, help="run directory (with metrics.jsonl)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--results", default=str(RESULT_DIR))
    a = ap.parse_args(argv)
    print(write_dashboard(a.run, a.out, Path(a.results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

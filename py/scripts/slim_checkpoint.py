"""Strip a full training checkpoint down to the inference-only EMA weights.

usage: python slim_checkpoint.py FULL.pt OUT.pt

Keeps ``format``, ``ema``, ``extra``, ``counters`` and ``config``; drops ``model``, ``optimizer``,
``ret_moments`` and ``rng``. Absolute file paths in ``extra`` are reduced to their file names.
The EMA tensors are copied unchanged and compared with the source after the save.
"""
import hashlib
import sys
from pathlib import PurePosixPath, PureWindowsPath

import torch


def base(p):
    return PureWindowsPath(p).name if "\\" in p else PurePosixPath(p).name


def main(src, dst):
    full = torch.load(src, map_location="cpu", weights_only=False)
    extra = dict(full["extra"])
    for k in ("vocab", "catalog"):
        if isinstance(extra.get(k), str):
            extra[k] = base(extra[k])
    tok = dict(extra.get("tokenizer") or {})
    if isinstance(tok.get("removal_ratings"), str):
        tok["removal_ratings"] = base(tok["removal_ratings"])
    extra["tokenizer"] = tok
    slim = {"format": full["format"], "ema": full["ema"], "extra": extra,
            "counters": full["counters"], "config": full["config"]}
    torch.save(slim, dst)
    back = torch.load(dst, map_location="cpu", weights_only=False)
    assert list(back["ema"]) == list(full["ema"])
    for k, v in full["ema"].items():
        assert v.dtype == back["ema"][k].dtype and torch.equal(v, back["ema"][k]), k
    n = sum(v.numel() for v in full["ema"].values() if v.is_floating_point())
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    c = full["counters"]
    print(f"{dst} version={c['version']} samples={c['samples_seen']} params={n} tensors={len(full['ema'])} sha256={h}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

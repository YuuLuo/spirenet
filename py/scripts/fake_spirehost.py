"""Stand-in SpireHost process: serves ``spire.synth.FakeGame`` over the binary pipe protocol.

Usage (any HostProcess command line can point here until the real binary exists)::

    python -B scripts/fake_spirehost.py [--seed N] [--win-prob P] [--mean-decisions N] ...

Fault injection for wrapper tests: ``--hang-after N`` stops answering after N STEP commands,
``--crash-after N`` exits abruptly (code 17) after N STEP commands, ``--boot-delay S``
sleeps before serving, ``--stderr-every N`` writes a diagnostic stderr line every N commands.
Load modelling: ``--latency-ms X`` delays every reply by X ms (simulated engine time; delays
below ``--spin-below-ms`` (default 2 ms) are busy-waited, because Windows sleeps have about 1 ms
granularity), ``--frame-pool N`` serves pre-generated frames (microseconds per step, see
``FakeHostConfig``), ``--learnable`` makes the outcome depend on the policy, ``--no-hashes`` skips
table hashes, ``--size extreme`` serves near-limit frame sizes, ``--max-decisions N`` caps episodes
like the real host (TERMINAL LOSS / DECISION_CAP), ``--turn-cap-prob P`` ends episodes with COMBAT_TURN_CAP.

Transport: stdin/stdout by default; ``--pipe NAME`` connects to the actor's named pipe instead
and sends HELLO first (``notes/transport-spec.md``), exactly what SpireHost does in pipe mode.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import IO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A host never does BLAS work. Without this numpy's OpenBLAS starts one thread per core and commits
# ~760 MB per process on this machine (24 cores); 192 fake hosts exhausted the machine's commit and
# hung a benchmark (notes/py-throughput-1-hotpath.md section 5). Must precede the numpy import.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from spire.pipes import hello_message  # noqa: E402
from spire.proto import StepCmd, decode_command, frame_message, read_message  # noqa: E402
from spire.synth import FakeGame, FakeHostConfig  # noqa: E402
from spire.vocab import Vocab  # noqa: E402


def open_transport(pipe: str | None) -> tuple[IO[bytes], IO[bytes]]:
    """(reader, writer) of the command / frame streams."""
    if pipe is None:
        return sys.stdin.buffer, sys.stdout.buffer
    f = open(pipe, "r+b", buffering=0)  # noqa: SIM115 - lives for the whole process
    f.write(hello_message())
    return f, f  # type: ignore[return-value]


def wait_until(t_end: float, spin_below: float) -> None:
    """Sleep until ``perf_counter() >= t_end``; busy-wait the last ``spin_below`` seconds."""
    left = t_end - time.perf_counter()
    if left > spin_below:
        time.sleep(left - spin_below)
    while time.perf_counter() < t_end:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serve", action="store_true", help="accepted for command-line parity with SpireHost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--win-prob", type=float, default=0.5)
    ap.add_argument("--mean-decisions", type=float, default=300.0)
    ap.add_argument("--abort-prob", type=float, default=0.01)
    ap.add_argument("--error-prob", type=float, default=0.0)
    ap.add_argument("--learnable", action="store_true")
    ap.add_argument("--end-turn-penalty", type=int, default=FakeHostConfig.end_turn_penalty)
    ap.add_argument("--frame-pool", type=int, default=0)
    ap.add_argument("--no-hashes", action="store_true")
    ap.add_argument("--size", choices=("typical", "extreme", "h5"), default="typical", help="FakeHostConfig.size")
    ap.add_argument("--vocab", type=Path, help="custom vocabulary for real-checkpoint protocol smoke tests")
    ap.add_argument("--catalog-hash", type=int, default=0)
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--hang-after", type=int, default=-1)
    ap.add_argument("--crash-after", type=int, default=-1)
    ap.add_argument("--boot-delay", type=float, default=0.0)
    ap.add_argument("--stderr-every", type=int, default=0)
    ap.add_argument("--spin-below-ms", type=float, default=2.0)
    ap.add_argument("--pipe", default=None, help="named pipe of the actor (pipe transport)")
    ap.add_argument("--max-decisions", type=int, default=0,
                    help="host default per-episode decision cap (SpireHost --max-decisions; 0 = none)")
    ap.add_argument("--turn-cap-prob", type=float, default=0.0, help="FakeHostConfig.turn_cap_prob")
    a = ap.parse_args(argv)

    cfg = FakeHostConfig(win_prob=a.win_prob, mean_decisions=a.mean_decisions, abort_prob=a.abort_prob,
                         error_prob=a.error_prob, learnable=a.learnable, end_turn_penalty=a.end_turn_penalty,
                         frame_pool=a.frame_pool, hashes=not a.no_hashes, size=a.size,
                         decision_cap=a.max_decisions, turn_cap_prob=a.turn_cap_prob)
    game = FakeGame(Vocab.load(a.vocab) if a.vocab else None, seed=a.seed, config=cfg, catalog_hash=a.catalog_hash)
    latency = a.latency_ms / 1000.0
    spin = a.spin_below_ms / 1000.0
    print(f"fake_spirehost pid={os.getpid()} seed={a.seed} ready", file=sys.stderr, flush=True)
    if a.boot_delay:
        time.sleep(a.boot_delay)
    try:
        stdin, stdout = open_transport(a.pipe)
    except OSError as e:
        print(f"fatal: cannot open transport: {e}", file=sys.stderr, flush=True)
        return 3
    steps = commands = 0
    while True:
        try:
            payload = read_message(stdin)
        except OSError:  # pipe transport: the actor went away
            return 0
        if payload is None:
            return 0
        t0 = time.perf_counter()
        cmd = decode_command(payload)
        commands += 1
        if a.stderr_every and commands % a.stderr_every == 0:
            print(f"diag: {commands} commands, {steps} steps", file=sys.stderr, flush=True)
        if isinstance(cmd, StepCmd):
            steps += 1
            if steps == a.crash_after:
                print("fatal: injected crash", file=sys.stderr, flush=True)
                os._exit(17)
            if steps == a.hang_after:
                print("injected hang", file=sys.stderr, flush=True)
                while True:
                    time.sleep(3600)
        replies = game.handle(cmd)
        if latency:
            wait_until(t0 + latency, spin)
        try:
            stdout.write(b"".join(frame_message(out) for out in replies))
            stdout.flush()
        except OSError:
            return 0
        if game.quit:
            return 0


if __name__ == "__main__":
    sys.exit(main())

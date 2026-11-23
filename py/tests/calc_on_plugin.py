"""pytest plugin: run a test file with the calc input ON (notes/calc/contract.md; the pattern of forecast_on_plugin).

    cd py && PYTHONPATH=.:tests python -m pytest -p calc_on_plugin tests/test_learner.py

Every TokenizerConfig gets ``calc=True`` (also the default Tokenizer config), every ActorConfig the CALC RESET flag
(so fake host processes send the calc trailers too) and every synthetic run context the flag, so all frames carry the
TURN / FOE / ROUTE tables and every sample their kid rows."""
from __future__ import annotations

import dataclasses

from spire import actor, synth, tokenize
from spire.schema import ResetFlag

_tok_init = tokenize.TokenizerConfig.__init__
_ctx_init = synth.RunCtx.__init__
_actor_init = actor.ActorConfig.__init__


def _tok(self, *args, **kwargs) -> None:
    _tok_init(self, *args, **kwargs)
    object.__setattr__(self, "calc", True)
    self.__post_init__()


def _ctx(self, *args, **kwargs) -> None:
    _ctx_init(self, *args, **kwargs)
    self.reset_flags = int(self.reset_flags) | int(ResetFlag.CALC)


def _actor(self, *args, **kwargs) -> None:
    _actor_init(self, *args, **kwargs)
    object.__setattr__(self, "reset_flags", int(self.reset_flags) | int(ResetFlag.CALC))


tokenize.TokenizerConfig.__init__ = _tok
actor.ActorConfig.__init__ = _actor
synth.RunCtx.__init__ = _ctx
tokenize.Tokenizer.__init__.__defaults__ = (tokenize.TokenizerConfig(),)
actor.Actor.__init__.__defaults__ = (actor.ActorConfig(),)
assert actor.ActorConfig().reset_flags & int(ResetFlag.CALC)
assert tokenize.TokenizerConfig().calc and dataclasses.replace(tokenize.TokenizerConfig(), cycle_ban=True).calc


def pytest_report_header(config) -> str:
    return "calc_on_plugin: TokenizerConfig.calc forced on, synthetic frames carry TURN / FOE / ROUTE"

"""pytest plugin: run a test file with the move-forecast input ON (notes/solo/removal-spec.md section 8: "show the
taste tests pass with the forecast on").

    cd py-stage-m && PYTHONPATH=.:tests python -m pytest -p forecast_on_plugin tests/test_taste.py

Every TokenizerConfig gets ``forecast=True`` (also the default Tokenizer config), and every synthetic run context
(FrameGen frames, FakeHost episodes) gets the FORECAST RESET flag, so all frames carry the trailer and every sample
the FORECAST kid rows."""
from __future__ import annotations

import dataclasses

from spire import synth, tokenize
from spire.schema import ResetFlag

_tok_init = tokenize.TokenizerConfig.__init__
_ctx_init = synth.RunCtx.__init__


def _tok(self, *args, **kwargs) -> None:
    _tok_init(self, *args, **kwargs)
    object.__setattr__(self, "forecast", True)
    self.__post_init__()


def _ctx(self, *args, **kwargs) -> None:
    _ctx_init(self, *args, **kwargs)
    self.reset_flags = int(self.reset_flags) | int(ResetFlag.FORECAST)


tokenize.TokenizerConfig.__init__ = _tok
synth.RunCtx.__init__ = _ctx
tokenize.Tokenizer.__init__.__defaults__ = (tokenize.TokenizerConfig(),)
assert tokenize.TokenizerConfig().forecast and dataclasses.replace(tokenize.TokenizerConfig(), cycle_ban=True).forecast


def pytest_report_header(config) -> str:
    return "forecast_on_plugin: TokenizerConfig.forecast forced on, synthetic frames carry FORECAST"

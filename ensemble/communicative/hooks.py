from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch as t
import torch.nn as nn


@dataclass
class HookState:
  value: t.Tensor | None = None


def film(x: t.Tensor, m: t.Tensor) -> t.Tensor:
  gamma, beta = m.chunk(2, dim=1)
  return x * (1 + gamma) + beta


@dataclass(frozen=True)
class Injection:
  combine: Callable[[t.Tensor, t.Tensor], t.Tensor]
  width: int  # decoder output channels per backbone channel
  zero_init: bool


INJECTIONS = {
  "add": Injection(lambda x, m: x + m, 1, True),
  "film": Injection(film, 2, True),
  "replace": Injection(lambda x, m: m, 1, False),
}


@contextmanager
def hook_inject(layer: nn.Module, state: HookState, combine: Callable):
  # HookState is mutated across rounds; None means no message yet (perception pass)
  def fn(_, inputs):
    if state.value is None:
      return
    return (combine(inputs[0], state.value),)
  handle = layer.register_forward_pre_hook(fn)
  try:
    yield
  finally:
    handle.remove()


@contextmanager
def hook_encoder(layer: nn.Module, state: HookState):
  def fn(_, __, output):
    state.value = output
  handle = layer.register_forward_hook(fn)
  try:
    yield
  finally:
    handle.remove()

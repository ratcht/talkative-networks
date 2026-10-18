import math

import torch as t
import torch.nn as nn


class QKVEncoder(nn.Module):
  """Per-specialist TarMAC heads: query (receiver), key/signature and value (sender)."""
  def __init__(self, in_dim: int, key_dim: int, value_dim: int, reduce: nn.Module):
    super().__init__()
    self.reduce = reduce
    self.query_head = nn.Linear(in_dim, key_dim)
    self.key_head = nn.Linear(in_dim, key_dim)
    self.value_head = nn.Linear(in_dim, value_dim)

  def forward(self, x) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    f = self.reduce(x)
    return self.query_head(f), self.key_head(f), self.value_head(f)


class MLPEncoder(nn.Module):
  """QKVEncoder with a shared nonlinear trunk before the three heads diverge,
  instead of three independent linear maps off the raw pooled features —
  tests whether the encoder's expressiveness, not just its output
  dimensionality, is what limits communication."""
  def __init__(self, in_dim: int, key_dim: int, value_dim: int, reduce: nn.Module,
               hidden_dim: int | None = None):
    super().__init__()
    self.reduce = reduce
    hidden_dim = hidden_dim or in_dim
    self.trunk = nn.Sequential(
      nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
    )
    self.query_head = nn.Linear(hidden_dim, key_dim)
    self.key_head = nn.Linear(hidden_dim, key_dim)
    self.value_head = nn.Linear(hidden_dim, value_dim)

  def forward(self, x) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    f = self.trunk(self.reduce(x))
    return self.query_head(f), self.key_head(f), self.value_head(f)


class MLPNoLNEncoder(nn.Module):
  """MLPEncoder with the trunk's LayerNorm removed — isolates whether the
  nonlinearity itself (vs. the normalization riding along with it) is what
  the expressiveness gain depends on."""
  def __init__(self, in_dim: int, key_dim: int, value_dim: int, reduce: nn.Module,
               hidden_dim: int | None = None):
    super().__init__()
    self.reduce = reduce
    hidden_dim = hidden_dim or in_dim
    self.trunk = nn.Sequential(
      nn.Linear(in_dim, hidden_dim), nn.GELU(),
    )
    self.query_head = nn.Linear(hidden_dim, key_dim)
    self.key_head = nn.Linear(hidden_dim, key_dim)
    self.value_head = nn.Linear(hidden_dim, value_dim)

  def forward(self, x) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    f = self.trunk(self.reduce(x))
    return self.query_head(f), self.key_head(f), self.value_head(f)


class EyeQKVEncoder(QKVEncoder):
  """QKVEncoder with the value map initialized to identity."""
  def __init__(self, in_dim: int, key_dim: int, value_dim: int, reduce: nn.Module):
    assert value_dim == in_dim, f"identity value needs value_dim == in_dim ({value_dim} != {in_dim})"
    super().__init__(in_dim, key_dim, value_dim, reduce)
    nn.init.eye_(self.value_head.weight)
    nn.init.zeros_(self.value_head.bias)


class Decoder(nn.Module):
  def __init__(self, value_dim: int, out_dim: int, expand: nn.Module,
               hidden: bool = True, zero_init: bool = True):
    super().__init__()
    layers = [nn.Linear(value_dim, value_dim), nn.GELU()] if hidden else []
    self.net = nn.Sequential(*layers, nn.Linear(value_dim, out_dim))
    self.expand = expand

    if zero_init:
      nn.init.zeros_(self.net[-1].weight)
      nn.init.zeros_(self.net[-1].bias)

  def forward(self, x):
    return self.expand(self.net(x))


class LinearDecoder(Decoder):
  def __init__(self, value_dim: int, out_dim: int, expand: nn.Module, zero_init: bool = True):
    super().__init__(value_dim, out_dim, expand, hidden=False, zero_init=zero_init)


# (n, n) attention masks, True blocks receiver i from reading sender j
def all_mask(n: int) -> t.Tensor:
  return t.zeros(n, n, dtype=t.bool)

# Mask out self
def others_mask(n: int) -> t.Tensor:
  return t.eye(n, dtype=t.bool)

# Only self -> to use as a control
def self_mask(n: int) -> t.Tensor:
  return ~t.eye(n, dtype=t.bool)


class TarMACBus(nn.Module):
  """Bare single-head scaled dot-product attention across ensemble members.

  Each member attends over all members including itself — with n=2, excluding
  self would collapse the softmax to a constant 1.
  """
  mask: t.Tensor

  def __init__(self, mask: t.Tensor):
    super().__init__()
    self.register_buffer("mask", mask, persistent=False)

  def forward(
    self,
    Q: t.Tensor,  # (b, n, key_dim)
    K: t.Tensor,  # (b, n, key_dim)
    V: t.Tensor,  # (b, n, value_dim)
  ) -> tuple[t.Tensor, t.Tensor]:
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])  # (b, n, n)
    scores = scores.masked_fill(self.mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    return attn @ V, attn  # (b, n, value_dim), (b, n, n)


class PatchAttention(nn.Module):
  """Attention across one member's own patches."""
  def __init__(self, dim: int, key_dim: int, n_patches: int):
    super().__init__()
    self.encoder = EyeQKVEncoder(dim, key_dim, dim, nn.Identity())
    self.bus = TarMACBus(all_mask(n_patches))

  def forward(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    return self.bus(*self.encoder(x))  # (b, p, d), (b, p, p)


# ==== 05_minimal_exchange ====
# A member's message is its own late map, rows = channels, columns = grid
# positions, with no value head. Members are mixed by average, gate or
# attention, and the mix is decoded positions first, then channels.

def pair_average(positions: int) -> t.Tensor:
  """(positions // 2, positions) map averaging each pair of horizontally
  adjacent cells of a row-major grid with even width."""
  w = t.zeros(positions // 2, positions)
  rows = t.arange(positions // 2)
  w[rows, 2 * rows] = 0.5
  w[rows, 2 * rows + 1] = 0.5
  return w


def position_projection(positions: int, out: int | None) -> nn.Module:
  """Learned map over grid positions, shared by all channels. Starts as a
  fixed spatial operation: identity when out == positions, pair averaging when
  out == positions // 2. None means no projection."""
  if out is None:
    return nn.Identity()
  proj = nn.Linear(positions, out, bias=False)
  with t.no_grad():
    if out == positions:
      proj.weight.copy_(t.eye(positions))
    elif 2 * out == positions:
      proj.weight.copy_(pair_average(positions))
    else:
      raise ValueError(f"projection {positions} -> {out} has no init, use {positions} or {positions // 2}")
  return proj


class MinimalEncoder(nn.Module):
  """Sends its projected late map, flattened, as the value. The heads it owns
  depend on the mix: query and key for attention, one score for the gate,
  none for the average. zero_query starts attention uniform, like the gate."""
  def __init__(self, channels: int, positions: int, proj: int | None, mix: str, key_dim: int,
               zero_query: bool = False):
    super().__init__()
    self.proj = position_projection(positions, proj)
    dim = channels * (proj or positions)
    self.query_head = nn.Linear(dim, key_dim) if mix == "attention" else None
    self.key_head = nn.Linear(dim, key_dim) if mix == "attention" else None
    self.score_head = nn.Linear(dim, 1) if mix == "gate" else None
    if self.score_head is not None:
      # uniform gate at step 0, so the gate arm starts as the average arm
      nn.init.zeros_(self.score_head.weight)
      nn.init.zeros_(self.score_head.bias)
    if zero_query:
      assert self.query_head is not None, "zero_query needs mix attention"
      # q = 0 makes every score 0, so attention starts as the average. The
      # random keys still give the query head a gradient
      nn.init.zeros_(self.query_head.weight)
      nn.init.zeros_(self.query_head.bias)

  def forward(self, x) -> tuple[t.Tensor | None, t.Tensor | None, t.Tensor]:
    v = self.proj(x.flatten(2)).flatten(1).unsqueeze(1)  # (b, 1, c * s)
    if self.score_head is not None:
      return None, self.score_head(v), v
    if self.query_head is not None:
      return self.query_head(v), self.key_head(v), v
    return None, None, v


class MinimalDecoder(nn.Module):
  """(b, 1, in_ch * s) -> (b, out_ch, h, w). Maps positions s -> h * w, then
  channels in_ch -> out_ch. hidden puts a GELU between the two."""
  def __init__(self, in_ch: int, s: int, out_ch: int, out_hw: tuple[int, int],
               hidden: bool, zero_init: bool):
    super().__init__()
    self.in_ch = in_ch
    self.out_hw = out_hw
    self.positions = nn.Linear(s, out_hw[0] * out_hw[1])
    self.act = nn.GELU() if hidden else nn.Identity()
    self.channels = nn.Linear(in_ch, out_ch)
    if zero_init:
      nn.init.zeros_(self.channels.weight)
      nn.init.zeros_(self.channels.bias)

  def forward(self, x):
    x = x.squeeze(1).unflatten(-1, (self.in_ch, -1))  # (b, in_ch, s)
    x = self.act(self.positions(x))                   # (b, in_ch, h * w)
    x = self.channels(x.transpose(1, 2))              # (b, h * w, out_ch)
    return x.transpose(1, 2).unflatten(-1, self.out_hw)


class AverageBus(nn.Module):
  """Every receiver gets the plain mean over members. No weights to report."""
  def forward(self, Q, K, V: t.Tensor) -> tuple[t.Tensor, None]:
    return V.mean(dim=-2, keepdim=True).expand_as(V), None


class GateBus(nn.Module):
  """Softmax over one score per sender. Every receiver gets the same mix, so
  every row of the reported (b, p, n, n) weights is the gate."""
  def forward(self, Q, K: t.Tensor, V: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    g = K.squeeze(-1).softmax(dim=-1)  # (b, p, n)
    n = g.shape[-1]
    attn = g.unsqueeze(-2).expand(*g.shape[:-1], n, n)  # (b, p, n, n)
    return attn @ V, attn

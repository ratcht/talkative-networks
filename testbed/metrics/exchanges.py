"""
Metrics on one exchange: what the experts sent, how the mixer weighted it, and
what the decoders injected. Definitions: docs/testbed/metrics.md.

An ExchangeMeter reads one exchange's traces batch by batch over a full pass
of joint_val. Messages, mixing weights and, for a backward round, the experts'
answers after the round are small, so it keeps them whole.
Injections are feature maps, so it reduces each batch of them to running sums
as it arrives; the statistics are still exact over the whole split.
"""

import torch
from torch.special import xlogy

from testbed.communication.ensemble import Trace
from testbed.metrics.answers import answer_grid

# An injection with a smaller norm counts as zero. Every channel starts there,
# since the decoders' last layers are zero-initialised.
ZERO_NORM = 1e-12
# Below this share of spatially varying magnitude there is no spatial pattern
# whose rank means anything.
ZERO_RESID = 1e-6


class ExchangeMeter:
  def __init__(self):
    self.values: list[torch.Tensor] = []
    self.received: list[torch.Tensor] = []
    self.weights: list[torch.Tensor] = []
    self.answers: list[torch.Tensor] = []
    self.injections: list[InjectionSums] = []

  def add(self, trace: Trace) -> None:
    self.values.append(trace.reading.sent.value)
    self.received.append(trace.reading.received)
    if trace.reading.weights is not None:
      self.weights.append(per_expert(trace.reading.weights))
    if trace.answers is not None:
      self.answers.append(trace.answers)
    if not self.injections:
      self.injections = [InjectionSums() for _ in trace.injection]
    for sums, injection in zip(self.injections, trace.injection):
      sums.add(injection)

  def result(
    self,
    pre: torch.Tensor,
    post: torch.Tensor,
    y: torch.Tensor,
    previous_routes: torch.Tensor | None = None,
  ) -> tuple[dict[str, float], torch.Tensor | None]:
    """The exchange's metrics over the pass, given the experts' logits on the
    same examples, and its routes, for the next pass's route_flip_rate."""
    values, received = torch.cat(self.values), torch.cat(self.received)
    injections = [sums.result() for sums in self.injections]
    out = {
      "msg_norm/value": values.norm(dim=-1).mean().item(),
      "msg_norm/aggregated": received.norm(dim=-1).mean().item(),
      "msg_norm/injection": mean(i["norm"] for i in injections),
      "inj_spread": mean(i["spread"] for i in injections),
    }
    if all("resid_frac" in i for i in injections):  # written into feature maps
      out["inj_resid_frac"] = mean(i["resid_frac"] for i in injections)
      out["inj_rank"] = mean(i["rank"] for i in injections)
    out |= message_ranks(values, received)
    if self.answers:  # a backward round: the grid as it stands after this round
      grid = answer_grid(torch.cat(self.answers), y)
      out |= {f"answer_grid/{cell}": v for cell, v in grid.items()}
    if not self.weights:  # a mixer without weights has no routing to read
      return out, None
    weights = torch.cat(self.weights)
    routing_metrics, routes = routing(weights, pre, post, y, previous_routes)
    return out | attention(weights) | routing_metrics, routes


def per_expert(weights: torch.Tensor) -> torch.Tensor:
  """(B, M, T, M, T) mixing weights -> (B, M, M): how much each receiving
  expert drew on each sending one. Sums over the sender's messages, averages
  over the receiver's; with one message each it just drops the message axes."""
  return weights.sum(-1).mean(2)


def attention(weights: torch.Tensor) -> dict[str, float]:
  """weights (N, M, M): receiver r's row says how it weighted each sender."""
  return {
    "attn_entropy": -xlogy(weights, weights).sum(-1).mean().item(),
    "attn_top1": weights.max(-1).values.mean().item(),
  }


def routing(
  weights: torch.Tensor,
  pre: torch.Tensor,
  post: torch.Tensor,
  y: torch.Tensor,
  previous: torch.Tensor | None = None,
) -> tuple[dict[str, float], torch.Tensor]:
  """Who each receiver listened to most, routes[n, r] for example n and
  receiver r, and what that says about the routing."""
  routes = weights.argmax(-1)  # (N, M)
  n, m = routes.shape
  is_self = routes == torch.arange(m, device=routes.device)
  out = {
    "self_attn_rate": is_self.float().mean().item(),
    "routing_mode_share/joint": mode_share(routes),
  }
  for r in range(m):
    out[f"routing_mode_share/expert{r}"] = mode_share(routes[:, r : r + 1])
  if previous is not None:
    out["route_flip_rate"] = (routes != previous).float().mean().item()

  right_pre = pre.argmax(-1) == y[:, None]  # (N, M) each expert's own answer
  right_post = post.argmax(-1) == y[:, None]
  sender_right = right_pre.gather(1, routes)  # was the sender listened to right?
  others = ~is_self  # self-attention would make the corrected subset circular
  subsets = {
    "all": others,
    "corrected": others & ~right_pre & right_post,
    "corrupted": others & right_pre & ~right_post,
  }
  for name, subset in subsets.items():
    if subset.any():
      out[f"routed_correct/{name}"] = (
        (sender_right & subset).sum() / subset.sum()
      ).item()
  return out, routes


def mode_share(routes: torch.Tensor) -> float:
  """Share of rows equal to the most common row."""
  _, counts = torch.unique(routes, dim=0, return_counts=True)
  return counts.max().item() / len(routes)


def message_ranks(values: torch.Tensor, received: torch.Tensor) -> dict[str, float]:
  """values, received (N, M, T, value_dim): each expert's sent and received messages."""
  out = {}
  for part, messages in (("value", values), ("aggregated", received)):
    ranks = [
      centred_rank(messages[:, i].flatten(0, 1)) for i in range(messages.shape[1])
    ]
    out |= {f"msg_rank_{part}/expert{i}": r for i, r in enumerate(ranks)}
    out[f"msg_rank_{part}/mean"] = mean(ranks)
  if out["msg_rank_value/mean"] > 0:
    out["msg_rank_ratio"] = out["msg_rank_aggregated/mean"] / out["msg_rank_value/mean"]
  return out


def centred_rank(rows: torch.Tensor) -> float:
  """Participation ratio of rows (N, D) around their mean row. The centring
  matters: an offset shared by every row would dominate and read about 1 for a
  message in perfect health."""
  rows = rows.double()
  rows = rows - rows.mean(0)
  return participation_ratio(rows.T @ rows).item()


def participation_ratio(gram: torch.Tensor) -> torch.Tensor:
  """(sum of eigenvalues)^2 / (sum of squared eigenvalues) of each Gram matrix
  in gram (..., D, D): the number of directions, each weighted by its share of
  the magnitude. The sums are the trace and the squared Frobenius norm, so no
  eigendecomposition is needed. 0 for an all-zero Gram."""
  trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
  frobenius_sq = gram.pow(2).sum((-2, -1))
  return torch.where(
    frobenius_sq > 0, trace**2 / frobenius_sq.clamp_min(torch.finfo(gram.dtype).tiny), 0
  )


class InjectionSums:
  """Running sums over a pass that give one expert's injection statistics
  exactly, without keeping the injections. Float64, since inj_spread subtracts
  two nearly equal sums when the injection is close to constant."""

  def __init__(self):
    self.count = 0
    self.norm = 0.0  # sum over examples of each example's injection norm
    self.square = 0.0  # squared norm of every injection together
    self.total: torch.Tensor | None = None  # sum of the injections
    self.resid_square = 0.0  # squared norm of the parts that vary over positions
    self.gram: torch.Tensor | None = None  # (C, P, P), summed over examples

  def add(self, injection: torch.Tensor) -> None:
    d = injection.double()
    self.count += len(d)
    self.norm += d.flatten(1).norm(dim=1).sum().item()
    self.square += d.pow(2).sum().item()
    self.total = _accumulate(self.total, d.sum(0))
    if d.dim() > 2:  # a feature map (C, H, W): split off each channel's spatial mean
      grid = d.flatten(2)  # (B, C, P)
      resid = grid - grid.mean(-1, keepdim=True)
      self.resid_square += resid.pow(2).sum().item()
      self.gram = _accumulate(self.gram, torch.einsum("bcp,bcq->cpq", resid, resid))

  def result(self) -> dict[str, float]:
    """norm, spread, and resid_frac and rank for a feature map."""
    size = self.square**0.5
    out = {"norm": self.norm / self.count, "spread": 0.0}
    if self.gram is not None:
      out |= {"resid_frac": 0.0, "rank": 0.0}
    if size < ZERO_NORM:
      return out
    mean_square = self.total.pow(2).sum().item() / self.count
    out["spread"] = max(self.square - mean_square, 0.0) ** 0.5 / size
    if self.gram is not None:
      out["resid_frac"] = self.resid_square**0.5 / size
      if out["resid_frac"] >= ZERO_RESID:
        out["rank"] = participation_ratio(self.gram).mean().item()
    return out


def _accumulate(total: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
  return x if total is None else total + x


def mean(xs) -> float:
  xs = list(xs)
  return sum(xs) / len(xs)

"""
The channel metrics of a graft (testbed/grafting.py): what the stitches hand
the obligates, what the obligates send back, how the anchor weighs them and
what is added to its stream. The names are the ensemble's where the meaning
carries over; docs/testbed/grafting.md §6 says what each means for a graft.
"""

import torch
from torch.special import xlogy

from testbed.metrics.exchanges import InjectionSums, mean


class GraftMeter:
  """Reads the graft's trace from every batch of one pass; result() gives its
  metrics without keeping the tensors. Obligates are numbered as in the group,
  from 1, since the anchor is expert 0."""

  def __init__(self):
    self.stitched: list[InjectionSums] = []
    self.replies: list[InjectionSums] = []
    self.injection: InjectionSums | None = None
    self.query, self.key = InjectionSums(), InjectionSums()
    self.shares: torch.Tensor | None = None  # summed over examples and positions, (J,)
    self.entropy = 0.0
    self.top1 = 0.0
    self.count = 0  # (example, anchor position) pairs
    self.answers: list[torch.Tensor] = []  # (B, J) per batch: obligates' own answers

  def add(self, trace) -> None:
    if trace.obligate_answers is not None:
      self.answers.append(trace.obligate_answers.detach())
    if self.injection is None:
      self.stitched = [InjectionSums() for _ in trace.stitched]
      self.replies = [InjectionSums() for _ in trace.replies]
      self.injection = InjectionSums(trace.tokens)
    for sums, s in zip(self.stitched, trace.stitched):
      sums.add(s.detach().flatten(1))
    for sums, r in zip(self.replies, trace.replies):
      sums.add(r.detach().flatten(1))
    mixed = trace.mixed
    self.injection.add(mixed.injection.detach())
    if mixed.query is not None:
      self.query.add(mixed.query.detach().flatten(1))
      self.key.add(mixed.key.detach().flatten(1))
    if mixed.shares is not None:
      s = mixed.shares.detach().double()  # (B, P, J)
      total = s.sum((0, 1))
      self.shares = total if self.shares is None else self.shares + total
      self.entropy += -xlogy(s, s).sum(-1).sum().item()
      self.top1 += s.max(-1).values.sum().item()
      self.count += s.shape[0] * s.shape[1]

  def result(self, pre, post, y, previous=None) -> tuple[dict[str, float], None]:
    """pre and post are the split's answers, which these metrics don't use; y
    scores the obligates' own answers. There is no route to carry over, so the
    second value is always None."""
    out = {}
    if self.answers:  # each obligate run from its stitched stream to its own head
      answers = torch.cat(self.answers)  # (N, J)
      right = (answers == y.to(answers.device)[:, None]).double().mean(0)
      for j, share in enumerate(right.tolist(), start=1):
        out[f"stitched_accuracy/expert{j}"] = share
      out["stitched_accuracy"] = right.mean().item()
    for name, sums in (("stitched", self.stitched), ("reply", self.replies)):
      stats = [s.result() for s in sums]
      out[f"msg_norm/{name}"] = mean(st["norm"] for st in stats)
      out[f"msg_spread/{name}"] = mean(st["spread"] for st in stats)
      for i, st in enumerate(stats, start=1):
        out[f"msg_norm/{name}/expert{i}"] = st["norm"]
        out[f"msg_spread/{name}/expert{i}"] = st["spread"]
    injection = self.injection.result()
    out["msg_norm/injection"] = injection["norm"]
    out["inj_spread"] = injection["spread"]
    for key in ("resid_frac", "rank"):
      if key in injection:
        out[f"inj_{key}"] = injection[key]
    if self.query.count:
      for name, sums in (("query", self.query), ("key", self.key)):
        stats = sums.result()
        out[f"msg_norm/{name}"] = stats["norm"]
        out[f"msg_spread/{name}"] = stats["spread"]
    if self.count:
      for j, share in enumerate((self.shares / self.count).tolist(), start=1):
        out[f"attn_heard/expert0/from{j}"] = share
      out["attn_entropy"] = self.entropy / self.count
      out["attn_top1"] = self.top1 / self.count
    return out, None

"""
Train an ensemble over frozen backbones, given a split and backbone checkpoints.

Methods (one subcommand each; gating will join later):

communicative — TarMAC-aligned ensemble: per-specialist Q/K/V heads, bare
                attention bus, decoder injection via hooks.

A `summarize` subcommand aggregates a group of same-config, different-seed
replicate runs (produced by scripts/replicate_joint.py) into one summary run.

Example
-------
uv run python ensemble/train.py communicative \\
    --manifest selection/manifests/high_gap.json \\
    --split-file splits/three_way_seed42.pt \\
    --epochs 10 --k-rounds 1
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from aim import Repo, Run
from einops import rearrange, repeat
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backbone.resnet import ResNet
from ensemble.communicative.driver import Driver
from ensemble.communicative.bus import QKVEncoder, MLPEncoder, MLPNoLNEncoder, Decoder
from ensemble.communicative.diagnostics import spatial_diagnostics
from ensemble.communicative.spatial import (
  Shape, gap_reduce, PoolReduce, ConvReduce, FlatReduce,
  broadcast_expand, MaskExpand, ConvTDecoder,
)
from ensemble.communicative.orchestrator import Specialist, Orchestrator, prob_average
from ensemble.eval import evaluate
from ensemble import metric_docs

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def seed_everything(seed: int, deterministic: bool) -> None:
  """Seed every RNG a training run can touch, including ones nothing uses yet.

  cudnn is the one that matters most here: left alone it's free to pick a
  different conv algorithm run-to-run even at a fixed seed, which on a
  54-block ResNet-110 stack is a second, unseeded source of variance sitting
  on top of the joint-training seed this whole replicate study exists to
  isolate. deterministic=False trades that guarantee for cudnn's autotuned
  (usually faster) kernels.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = deterministic
  torch.backends.cudnn.benchmark = not deterministic


@dataclass
class TrainConfig:
  epochs: int
  lr: float
  weight_decay: float
  seed: int
  log_every: int = 50
  device: str = "cuda"


@dataclass
class TrainJob:
  model: nn.Module
  loader: DataLoader
  optimizer: Optimizer
  criterion: nn.Module
  config: TrainConfig
  run: Optional[Run] = None
  val_loader: Optional[DataLoader] = None  # if set, evaluated + tracked every epoch
  # if set, test accuracy is tracked every epoch rather than once at the end,
  # which is what lets the endpoint be chosen on validation after the fact
  # (docs/studies/02-spatial-ports.md §4.3) instead of taken at termination
  test_loader: Optional[DataLoader] = None
  # one fixed validation batch, the same batch every epoch and in every arm, on
  # which §5.2's spatial diagnostics are computed
  diag_batch: Optional[torch.Tensor] = None
  # per-epoch accuracies, filled by train(); the endpoint is computed from
  # these rather than re-read from Aim, whose index is not refreshed in-process
  epoch_history: dict[str, list[float]] = field(default_factory=dict)


# trainable components of the orchestrator, matched against parameter name
# tokens; query/key norms are the early warning for vanishing gradients since
# they only receive gradient through the attention softmax.
#
# First hit wins, so a more specific token must precede the broader one it sits
# inside: "expand" before "decoder", since a spatially-structured decoder keeps
# its mask at decoder.expand.mask and would otherwise be lumped in with the MLP.
_GRAD_COMPONENTS = ("reduce", "expand", "query_head", "key_head", "value_head",
                    "decoder", "fc", "trunk")
# Trainable parameters matching no token land here instead of vanishing. Before
# this existed, adding a submodule under a name no token covered (an encoder's
# learned reduce, say) dropped it from every per-component series silently — it
# still counted toward total, so nothing looked wrong.
_GRAD_OTHER = "other"


def _component_of(param_name: str) -> str:
  tokens = param_name.split(".")
  for c in _GRAD_COMPONENTS:
    if c in tokens:
      return c
  return _GRAD_OTHER


def grad_norms(model: nn.Module) -> dict[str, float]:
  """L2 norm of the gradients, total and per component."""
  groups: dict[str, list] = {c: [] for c in (*_GRAD_COMPONENTS, _GRAD_OTHER)}
  everything = []
  for name, p in model.named_parameters():
    if p.grad is None:
      continue
    everything.append(p)
    groups[_component_of(name)].append(p)

  def norm(params) -> float:
    if not params:
      return 0.0
    return torch.norm(torch.stack([p.grad.detach().norm(2) for p in params])).item()

  out = {"total": norm(everything)}
  out.update({c: norm(ps) for c, ps in groups.items() if ps})
  return out


def update_to_weight_ratios(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float]:
  """‖Δθ‖/‖θ‖ per component after an optimizer step.

  The 'are the weights actually moving' diagnostic: ~1e-3 per step is healthy,
  ≲1e-6 is effectively frozen, ≳1e-1 is thrashing. Immune to Adam's rescaling.
  """
  num: dict[str, float] = {"total": 0.0}
  den: dict[str, float] = {"total": 0.0}
  for name, p in model.named_parameters():
    old = before.get(name)
    if old is None:
      continue
    d = (p.detach() - old).pow(2).sum().item()
    w = old.pow(2).sum().item()
    for key in ("total", _component_of(name)):
      num[key] = num.get(key, 0.0) + d
      den[key] = den.get(key, 0.0) + w
  return {k: num[k] ** 0.5 / max(den[k] ** 0.5, 1e-12) for k in num}


def layernorm_param_norms(model: nn.Module) -> dict[str, float]:
  """L2 norm of every LayerNorm's affine params, gamma (weight) and beta (bias).

  Only present when the model has any nn.LayerNorm (MLPEncoder's trunk) —
  empty for qkv-encoder arms.
  """
  gammas = [m.weight.detach() for m in model.modules()
            if isinstance(m, nn.LayerNorm) and m.weight is not None]
  betas = [m.bias.detach() for m in model.modules()
           if isinstance(m, nn.LayerNorm) and m.bias is not None]

  def norm(params) -> Optional[float]:
    if not params:
      return None
    return torch.norm(torch.stack([p.norm(2) for p in params])).item()

  out = {}
  gamma_norm, beta_norm = norm(gammas), norm(betas)
  if gamma_norm is not None:
    out["gamma"] = gamma_norm
  if beta_norm is not None:
    out["beta"] = beta_norm
  return out


def answer_shift_stats(
  pre: torch.Tensor, post: torch.Tensor, y: torch.Tensor
) -> dict[str, float]:
  """How communication changed the ENSEMBLE's answer, and at whose hand.

  pre/post: (n_specialists, batch, classes) logits from before/after
  communication (Orchestrator.last_shift); y: (batch,) labels.

  Everything here is read through prob_average(), the prob-averaged readout
  Orchestrator.forward returns and the loss scores. Per-specialist correction
  and corruption rates used to be reported instead, and were dropped: the
  objective is the ensemble's answer, so one member flipping away from the
  label is not intrinsically bad and one flipping toward it is not
  intrinsically good, and averaging those flips over members measures nothing
  anyone optimises.

  Two properties follow from reading the joint answer instead. `joint:net` is
  corrected minus corrupted, which is *exactly* the change in ensemble accuracy
  caused by communication, so a flat accuracy reading decomposes into either
  nothing happening or two large offsetting flows. And because every specialist
  is frozen, `pre` depends only on the input and the frozen weights, so the
  examples eligible for correction are the same set in every arm, every seed and
  every epoch.

  The `marginal:<i>:` keys substitute specialist i's post-communication logits
  into an otherwise pre-communication ensemble, leaving the others untouched:
  what this member's update contributes to the joint decision on its own. They
  do not sum to `joint:net`, the argmax of a mean of softmaxes being nonlinear
  in its members, and the gap is readable in itself — a sum below the joint
  figure means the updates cooperate rather than each moving the vote alone.

  agreement_pre/post are pairwise argmax agreement between members, kept
  because they are the control on `joint:net`: agreement rising without net
  rising is consensus without correctness. kl is the mean per-specialist
  divergence from pre to post, which catches probability movement too small to
  flip any argmax.

  Keys are namespaced with ':' so _track_answer_shift can route them to their
  own metric names and contexts.
  """
  right_pre = prob_average(pre).argmax(-1) == y
  right_post = prob_average(post).argmax(-1) == y

  corrected = (~right_pre & right_post).float().mean().item()
  corrupted = (right_pre & ~right_post).float().mean().item()
  out = {
    "joint:pre_accuracy": right_pre.float().mean().item(),
    "joint:post_accuracy": right_post.float().mean().item(),
    "joint:corrected": corrected,
    "joint:corrupted": corrupted,
    "joint:net": corrected - corrupted,
    "kl": (post.softmax(-1) * (F.log_softmax(post, -1)
                               - F.log_softmax(pre, -1))).sum(-1).mean().item(),
  }

  for i in range(pre.shape[0]):
    mixed = pre.clone()
    mixed[i] = post[i]
    right_mixed = prob_average(mixed).argmax(-1) == y
    c = (~right_pre & right_mixed).float().mean().item()
    x = (right_pre & ~right_mixed).float().mean().item()
    out[f"marginal:{i}:corrected"] = c
    out[f"marginal:{i}:corrupted"] = x
    out[f"marginal:{i}:net"] = c - x

  pre_ans, post_ans = pre.argmax(-1), post.argmax(-1)
  pairs = [(i, j) for i in range(pre.shape[0]) for j in range(i + 1, pre.shape[0])]
  if pairs:
    out["joint:agreement_pre"] = torch.stack(
      [(pre_ans[i] == pre_ans[j]).float().mean() for i, j in pairs]).mean().item()
    out["joint:agreement_post"] = torch.stack(
      [(post_ans[i] == post_ans[j]).float().mean() for i, j in pairs]).mean().item()
  return out


def routing_counts(
  attn: torch.Tensor,
  pre: torch.Tensor,
  post: torch.Tensor,
  y: torch.Tensor,
) -> dict[str, tuple[int, int]]:
  """Who each receiver routed to, and whether that specialist was right.

  attn: (b, p, n, n) post-softmax attention, receiver-major, p messages per
  specialist (1 when pooled). pre/post: (n, b, C) per-specialist logits from
  before/after communication. y: (b,) labels.

  Entropy says how concentrated routing is; this says who it concentrated *on*,
  which is the part that can be right or wrong. A sharp gate onto the wrong
  specialist is worse than a soft gate onto the right one, and the two are
  indistinguishable by entropy.

  Returns (numerator, denominator) counts rather than rates so a caller
  averaging over batches can sum them: the corrected/corrupted subsets vary
  wildly in size per batch, and averaging per-batch rates would weight a batch
  holding two corrected examples the same as one holding fifty.

  Every rate but self_attn is restricted to receivers that attended to
  *another* specialist. Self-attention would otherwise make the corrected split
  circular — a corrected receiver was wrong before the round by definition, so
  scoring its self-attention as a routing miss measures the definition rather
  than the routing.
  """
  if attn.dim() == 3:
    attn = attn[:, None]                          # pooled: one message, p = 1
  p = attn.shape[1]
  attn = rearrange(attn, "b p i j -> (b p) i j")
  pre, post = (repeat(x, "n b c -> n (b p) c", p=p) for x in (pre, post))
  y = repeat(y, "b -> (b p)", p=p)

  n = attn.shape[-1]
  attended = attn.argmax(-1)                                       # (b, n)
  is_self = attended == torch.arange(n, device=attn.device)[None, :]
  non_self = ~is_self

  right_pre = (pre.argmax(-1) == y).T                              # (b, n)
  right_post = (post.argmax(-1) == y).T
  attended_right = right_pre.gather(1, attended)                   # (b, n)

  corrected = ~right_pre & right_post
  corrupted = right_pre & ~right_post

  def counts(mask: torch.Tensor) -> tuple[int, int]:
    return int((attended_right & mask).sum()), int(mask.sum())

  return {
    "self_attn": (int(is_self.sum()), is_self.numel()),
    "all": counts(non_self),
    "corrected": counts(non_self & corrected),
    "corrupted": counts(non_self & corrupted),
  }


# joint routing patterns are encoded base-n into a single int64, so n**(n-1)
# has to stay inside int64; well before that bound the joint histogram stops
# being readable anyway (a mode share over n**n patterns is ~0 for any large
# group), so the joint series is simply dropped past this size
_MAX_JOINT_N = 12


def _tally(x: torch.Tensor) -> dict[int, int]:
  """Histogram of a 1-D integer tensor as a plain dict, so histograms from
  different batches can be pooled by key without agreeing on a bin count."""
  vals, counts = torch.unique(x, return_counts=True)
  return dict(zip(vals.tolist(), counts.tolist()))


def _merge_tally(into: dict[int, int], new: dict[int, int]) -> None:
  for k, v in new.items():
    into[k] = into.get(k, 0) + v


def routing_pattern_counts(attn: torch.Tensor) -> dict[str, dict[int, int]]:
  """How often each routing pattern occurs, jointly and per receiver.

  attn: (b, p, n, n) post-softmax attention, receiver-major.

  A *pattern* is one example's whole wiring diagram — the tuple of argmax
  senders across all n receivers at once, e.g. (0, 2, 2) — encoded base-n into
  one integer so it can be tallied. The per-receiver entries are the marginals
  of that tuple: the histogram of receiver i's argmax on its own.

  Returns raw histograms rather than the mode share itself because the share is
  not poolable — a caller accumulating over batches has to sum the counts and
  take the mode once at the end, since the modal pattern of the whole set need
  not be the modal pattern of any single batch.
  """
  if attn.dim() == 3:
    attn = attn[:, None]                          # pooled: one message, p = 1
  attn = rearrange(attn, "b p i j -> (b p) i j")
  n = attn.shape[-1]
  attended = attn.argmax(-1)                                       # (b, n)

  out: dict[str, dict[int, int]] = {}
  if n <= _MAX_JOINT_N:
    codes = (attended * (n ** torch.arange(n, device=attn.device))).sum(-1)
    out["joint"] = _tally(codes)
  for i in range(n):
    out[str(i)] = _tally(attended[:, i])
  return out


def route_flip_rate(current: torch.Tensor, previous: Optional[torch.Tensor]) -> Optional[float]:
  """Fraction of (example, receiver) pairs that changed attended-to sender.

  Both are (N, n) argmax tensors over the same examples in the same order — the
  val loader is shuffle=False over a fixed Subset with no augmentation, so row
  r is the same image every epoch and the comparison is meaningful.

  None when there is nothing to compare against (the first epoch) or when the
  shapes disagree, which would mean the evaluation set changed underneath us —
  better to skip the series than to report a flip rate against a different set
  of images.
  """
  if previous is None or previous.shape != current.shape:
    return None
  return (current != previous).float().mean().item()


@dataclass
class RoutingSummary:
  """Routing diagnostics pooled over one full evaluation pass."""
  counts: dict[str, tuple[int, int]]          # routing_counts(), pooled
  patterns: dict[str, dict[int, int]]         # routing_pattern_counts(), pooled
  routes: Optional[torch.Tensor] = None       # (N, n) argmax, in loader order


def _track_mode_share(run: Run, patterns: dict[str, dict[int, int]], prefix: str,
                      step: int, epoch: int) -> None:
  for receiver, hist in patterns.items():
    total = sum(hist.values())
    if total:
      run.track(max(hist.values()) / total, name=f"{prefix}routing_mode_share",
                step=step, epoch=epoch, context={"receiver": receiver})


def _track_routing(run: Run, counts: dict[str, tuple[int, int]], prefix: str,
                   step: int, epoch: int) -> None:
  num, den = counts["self_attn"]
  if den:
    run.track(num / den, name=f"{prefix}self_attn_rate", step=step, epoch=epoch)
  for subset in ("all", "corrected", "corrupted"):
    num, den = counts[subset]
    # empty subsets are normal (early training corrects nothing); tracking a
    # NaN would poison the series rather than leave a gap in it
    if den:
      run.track(num / den, name=f"{prefix}routed_correct", step=step, epoch=epoch,
                context={"subset": subset})


def _track_answer_shift(run: Run, stats: dict[str, float], prefix: str, step: int, epoch: int) -> None:
  """Route answer_shift_stats' namespaced keys to their own metric names.

  prefix is "answer_shift" for the training-set series and "val_answer_shift"
  for the validation ones; the base ("" or "val_") is what varies.
  """
  base = prefix.removesuffix("answer_shift")
  for key, v in stats.items():
    if key == "kl":
      # unbounded, where every other value lives in [0, 1] — its own metric
      # name so it never shares a y-axis with the rates
      run.track(v, name=f"{base}answer_shift_kl", step=step, epoch=epoch)
      continue
    parts = key.split(":")
    if parts[0] == "joint":
      run.track(v, name=f"{base}joint_answer_shift", step=step, epoch=epoch,
                context={"kind": parts[1]})
    elif parts[0] == "marginal":
      run.track(v, name=f"{base}marginal_answer_shift", step=step, epoch=epoch,
                context={"specialist": int(parts[1]), "kind": parts[2]})


@torch.no_grad()
def _evaluate_with_shift(
  model: nn.Module,
  loader: DataLoader,
  device: str,
  criterion: nn.Module,
  **model_kwargs,
) -> tuple[dict[str, float], dict[str, float], RoutingSummary]:
  """Single val pass returning (loss/accuracy, answer-shift stats, routing)."""
  model.eval()
  correct = total = 0
  loss_sum = 0.0
  shift_sums: dict[str, float] = {}
  routing_sums: dict[str, list[int]] = {}
  pattern_sums: dict[str, dict[int, int]] = {}
  route_chunks: list[torch.Tensor] = []
  for x, y in loader:
    x, y = x.to(device), y.to(device)
    out = model(x, **model_kwargs)
    correct += (out.argmax(-1) == y).sum().item()
    loss_sum += criterion(out, y).item() * y.size(0)
    total += y.size(0)
    attn = getattr(model, "last_attn", None)
    if attn is not None:
      for k, hist in routing_pattern_counts(attn).items():
        _merge_tally(pattern_sums.setdefault(k, {}), hist)
      # kept in loader order and on CPU: the next epoch diffs against this to
      # get the flip rate, and n is small enough that int8 always holds it
      route_chunks.append(attn.argmax(-1).to(torch.int8).cpu())
    shift = getattr(model, "last_shift", None)
    if shift is not None:
      for k, v in answer_shift_stats(shift["pre"], shift["post"], y).items():
        shift_sums[k] = shift_sums.get(k, 0.0) + v * y.size(0)
      if attn is not None:
        # summed as counts, not averaged as per-batch rates: the subsets are
        # small and uneven, so only a pooled numerator/denominator is right
        for k, (num, den) in routing_counts(attn, shift["pre"], shift["post"], y).items():
          acc = routing_sums.setdefault(k, [0, 0])
          acc[0] += num
          acc[1] += den
  metrics = {"accuracy": correct / total, "loss": loss_sum / total}
  return (metrics,
          {k: v / total for k, v in shift_sums.items()},
          RoutingSummary(
            counts={k: (v[0], v[1]) for k, v in routing_sums.items()},
            patterns=pattern_sums,
            routes=torch.cat(route_chunks) if route_chunks else None,
          ))


def train(job: TrainJob, *args, **kwargs) -> list[float]:
  cfg = job.config
  job.model.to(cfg.device)
  job.model.train()
  losses: list[float] = []
  step = 0
  # last epoch's val routing decisions, (N, n) argmax — see route_flip_rate
  prev_routes: Optional[torch.Tensor] = None

  # endless val-batch stream: one batch per training step gives val_loss the
  # same frequency (and single-batch noise profile) as train_loss at the cost
  # of one forward pass, instead of a 40x-slower full sweep per step
  val_iter = None
  if job.val_loader is not None and job.run is not None:
    def _cycle(loader):
      while True:
        yield from loader
    val_iter = _cycle(job.val_loader)

  pbar = tqdm(total=cfg.epochs * len(job.loader), desc="train")
  try:
    for epoch in range(cfg.epochs):
      correct = seen = 0
      # answer-shift accumulates across the epoch and is tracked once at its
      # end, batch-weighted: the same treatment train_accuracy gets, and what
      # makes the training and validation figures comparable quantities rather
      # than a per-step series against a per-epoch one
      shift_acc: dict[str, float] = {}
      shift_seen = 0
      for x, y in job.loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        job.optimizer.zero_grad()

        out = job.model(x, *args, **kwargs)
        loss = job.criterion(out, y)

        correct += (out.detach().argmax(1) == y).sum().item()
        seen += y.size(0)

        loss.backward()
        grads = grad_norms(job.model)
        before = {name: p.detach().clone()
                  for name, p in job.model.named_parameters() if p.requires_grad}
        job.optimizer.step()
        losses.append(loss.item())
        if job.run is not None:
          # epoch goes into track()'s epoch axis, NOT the context — context is
          # part of the sequence identity and would fragment it per epoch
          job.run.track(loss.item(), name="train_loss", step=step, epoch=epoch)
          for component, g in grads.items():
            job.run.track(g, name="grad_norm", step=step, epoch=epoch,
                          context={"component": component})
          for component, r in update_to_weight_ratios(job.model, before).items():
            job.run.track(r, name="update_ratio", step=step, epoch=epoch,
                          context={"component": component})
          # message magnitudes, if the model exposes them (Orchestrator does)
          stats = getattr(job.model, "last_stats", None)
          if stats:
            for part, v in stats.items():
              job.run.track(v, name="msg_norm", step=step, epoch=epoch,
                            context={"part": part})
          # gamma/beta norms, if the model has any LayerNorm (MLPEncoder's trunk)
          for param, v in layernorm_param_norms(job.model).items():
            job.run.track(v, name="layernorm_param_norm", step=step, epoch=epoch,
                          context={"param": param})
          # per-row attention entropy, if the model exposes it (Orchestrator does)
          attn_entropy = getattr(job.model, "last_attn_entropy", None)
          if attn_entropy is not None:
            job.run.track(attn_entropy, name="attn_entropy", step=step, epoch=epoch)
          # peak attention weight, the same row statistic read as a max instead
          # of an entropy
          attn_top1 = getattr(job.model, "last_attn_top1", None)
          if attn_top1 is not None:
            job.run.track(attn_top1, name="attn_top1", step=step, epoch=epoch)
          # how much of the batch shares one routing pattern — the
          # input-dependence question attn_entropy/attn_top1 structurally
          # cannot answer, since both average away the batch dimension
          attn = getattr(job.model, "last_attn", None)
          if attn is not None:
            _track_mode_share(job.run, routing_pattern_counts(attn), "", step, epoch)
          # how specialists changed their answers in response to messages
          shift = getattr(job.model, "last_shift", None)
          if shift is not None:
            for k, v in answer_shift_stats(shift["pre"], shift["post"], y).items():
              shift_acc[k] = shift_acc.get(k, 0.0) + v * y.size(0)
            shift_seen += y.size(0)
            # whether attention concentrated on a specialist that was actually
            # right — needs both the attention and the pre/post opinions
            if attn is not None:
              _track_routing(job.run,
                             routing_counts(attn, shift["pre"], shift["post"], y),
                             "", step, epoch)
          # streaming single-batch val loss; must run after the train-batch
          # diagnostics above since this forward overwrites last_stats/last_shift
          if val_iter is not None:
            vx, vy = next(val_iter)
            vx, vy = vx.to(cfg.device), vy.to(cfg.device)
            with torch.no_grad():
              val_batch_loss = job.criterion(job.model(vx, *args, **kwargs), vy).item()
            job.run.track(val_batch_loss, name="val_loss", step=step, epoch=epoch)
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grads['total']:.2e}")
        step += 1

      # accuracies live on an epoch axis (step == epoch) so early stopping can
      # be read off directly; per-step val_loss is handled in the batch loop
      train_acc = correct / max(seen, 1)
      epoch_line = f"epoch {epoch + 1}/{cfg.epochs}  train_accuracy {train_acc:.4f}"
      if job.run is not None:
        job.run.track(train_acc, name="train_accuracy", step=epoch, epoch=epoch)
        if shift_seen:
          _track_answer_shift(job.run, {k: v / shift_seen for k, v in shift_acc.items()},
                              "answer_shift", epoch, epoch)
      if job.val_loader is not None:
        val, val_shift, val_routing = _evaluate_with_shift(
          job.model, job.val_loader, cfg.device, job.criterion, **kwargs)
        job.model.train()  # the val pass leaves the model in eval mode
        # how much of the routing changed since last epoch, over the same
        # images in the same order — separates a wiring that was learned from
        # one that was fixed at init and never moved
        flips = route_flip_rate(val_routing.routes, prev_routes) \
          if val_routing.routes is not None else None
        if val_routing.routes is not None:
          prev_routes = val_routing.routes
        job.epoch_history.setdefault("val_accuracy", []).append(val["accuracy"])
        if job.run is not None:
          job.run.track(val["accuracy"], name="val_accuracy", step=epoch, epoch=epoch)
          _track_answer_shift(job.run, val_shift, "val_answer_shift", epoch, epoch)
          _track_routing(job.run, val_routing.counts, "val_", epoch, epoch)
          _track_mode_share(job.run, val_routing.patterns, "val_", epoch, epoch)
          # None on the first epoch — nothing to diff against yet, and a 0.0
          # there would read as "routing never moved", the exact opposite
          if flips is not None:
            job.run.track(flips, name="val_route_flip_rate", step=epoch, epoch=epoch)
        epoch_line += "  " + "  ".join(f"val_{k} {v:.4f}" for k, v in val.items())

      # test accuracy per epoch, so §4.3's endpoint can average over a window
      # around the validation-selected epoch. Selection never consults this.
      if job.test_loader is not None and job.run is not None:
        test = evaluate(job.model, job.test_loader, device=cfg.device,
                        criterion=job.criterion, **kwargs)
        job.model.train()
        job.run.track(test["accuracy"], name="test_accuracy", step=epoch, epoch=epoch)
        job.epoch_history.setdefault("test_accuracy", []).append(test["accuracy"])
        epoch_line += f"  test_accuracy {test['accuracy']:.4f}"

      # whether the adapters used the spatial access their level grants them
      if job.diag_batch is not None and job.run is not None:
        diag = spatial_diagnostics(job.model, job.diag_batch, cfg.device,
                                   kwargs.get("k_rounds", 0))
        job.model.train()
        for name, value in diag.items():
          if isinstance(value, list):
            for i, v in enumerate(value):
              job.run.track(v, name=name, step=epoch, epoch=epoch,
                            context={"specialist": i})
          else:
            job.run.track(value, name=name, step=epoch, epoch=epoch)
      pbar.write(epoch_line)
  except KeyboardInterrupt:
    print(f"interrupted at step {step}")
  return losses


# ---------------------------------------------------------------------------
# Data: ensemble split for training, shared val, official test set
# ---------------------------------------------------------------------------

def make_loaders(
  split_file: Path,
  batch_size: int,
  data_root: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
  train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
  ])
  eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
  ])

  split = torch.load(split_file, weights_only=True)
  train_set = datasets.CIFAR100(data_root, train=True, transform=train_tf, download=True)
  val_set = datasets.CIFAR100(data_root, train=True, transform=eval_tf, download=True)
  test_set = datasets.CIFAR100(data_root, train=False, transform=eval_tf, download=True)

  train_loader = DataLoader(
    Subset(train_set, split["ensemble_indices"]),
    batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
  )
  val_loader = DataLoader(
    Subset(val_set, split["val_indices"]),
    batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
  )
  test_loader = DataLoader(
    test_set,
    batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
  )
  return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Method: communicative (TarMAC-aligned)
# ---------------------------------------------------------------------------

def _load_backbone(path: Path, device: str) -> ResNet:
  ckpt = torch.load(path, map_location=device, weights_only=True)
  # depth comes from the checkpoint so a pool of any CIFAR ResNet loads; older
  # checkpoints predate the field and are all ResNet-20
  depth = ckpt.get("metadata", {}).get("depth", 20)
  model = ResNet(depth=depth).to(device)
  model.load_state_dict(ckpt["state_dict"])
  model.eval()
  return model


@dataclass
class SpecialistSpec:
  """One specialist's construction config: which backbone, which layer it
  reads from/injects into, which head implementations to use.

  early/late and encoder/decoder are legitimately per-specialist — each is
  internal to one specialist's own backbone/heads. key_dim/value_dim are NOT
  here: Orchestrator.forward stacks every specialist's Q/K/V into single
  (b, n, key_dim)/(b, n, value_dim) tensors, so every specialist's encoder
  must project into the same dimensionality — that stays an
  Orchestrator-level setting, passed separately to build_communicative.
  """
  backbone_path: Path
  early: str
  late: str
  encoder: str = "qkv"
  decoder: str = "mlp"


# Swappable adapters, keyed by the name SpecialistSpec.encoder / .decoder
# refers to. Additive only: a new architecture is added by registering a
# factory here, never by editing build_communicative.
#
# An encoder name selects a trunk *and* a spatial mapping at the read port,
# since Study 02's encoder levels are one trunk with four different mappings
# (docs/studies/02-spatial-ports.md §3.2). The pre-Study-02 names are kept as
# aliases of the mapping they always had, so checkpoints written before the
# rename still load.
ENCODERS: dict[str, tuple[type, Callable[[Shape], nn.Module]]] = {
  "gap": (MLPEncoder, gap_reduce),         # E1 — no spatial access
  "pool": (MLPEncoder, PoolReduce),        # E2 — input-independent mask
  "conv": (MLPEncoder, ConvReduce),        # E3 — convolutional prior
  "flat": (MLPEncoder, FlatReduce),        # E4 — locality prior only
  "qkv": (QKVEncoder, gap_reduce),
  "mlp": (MLPEncoder, gap_reduce),         # == "gap"
  "mlp_no_ln": (MLPNoLNEncoder, gap_reduce),
}

# A decoder factory takes (value_dim, write-port shape). D1 and D2 share the
# Decoder class and differ only in their expand; D3 replaces it outright,
# its dense stage emitting a spatial grid rather than one value per channel.
def _decoder(expand: Callable[[Shape], nn.Module]) -> Callable[[int, Shape], nn.Module]:
  def make(value_dim: int, shape: Shape) -> nn.Module:
    return Decoder(value_dim, shape[0], expand(shape))
  return make


DECODERS: dict[str, Callable[[int, Shape], nn.Module]] = {
  "broadcast": _decoder(broadcast_expand),  # D1 — uniform per-channel bias
  "mask": _decoder(MaskExpand),             # D2 — input-independent shape
  "convt": ConvTDecoder,                    # D3 — message-conditioned shape
  "mlp": _decoder(broadcast_expand),        # == "broadcast"
}


def resolve_specialist_specs(
  backbone_paths: list[Path],
  defaults: argparse.Namespace,
  overrides_path: Optional[Path],
) -> list[SpecialistSpec]:
  """Build one SpecialistSpec per backbone, in order.

  overrides_path is a sparse dict keyed by *position* in backbone_paths (not
  by backbone path — architecture is identical across the pool, so position
  is simpler and needs no index-alignment with a second file), e.g.
  {"1": {"early": "layer1.1"}}. Only positions that differ from the
  CLI-level homogeneous defaults (defaults.early/late) need an entry.
  """
  overrides: dict[str, dict] = {}
  if overrides_path is not None:
    overrides = json.loads(overrides_path.read_text())

  specs = []
  for i, path in enumerate(backbone_paths):
    o = overrides.get(str(i), {})
    specs.append(SpecialistSpec(
      backbone_path=path,
      early=o.get("early", defaults.early),
      late=o.get("late", defaults.late),
      encoder=o.get("encoder", defaults.encoder),
      decoder=o.get("decoder", defaults.decoder),
    ))
  return specs


@torch.no_grad()
def _adapter_dims(
  backbone: nn.Module, early: str, late: str, device: str
) -> tuple[Shape, Shape]:
  """(C, H, W) of the tensor the decoder injects into / the encoder reads from.

  Spatial extent and not just channel width, since Study 02's adapters are
  sized against it: a learned mask spans H*W entries, a convolutional stack's
  flatten depends on it, and a transpose-convolution stack needs to know how
  many doublings reach the port.
  """
  shapes = {}
  h_early = backbone.get_submodule(early).register_forward_pre_hook(
    lambda m, inp: shapes.__setitem__("early_in", inp[0].shape)
  )
  h_late = backbone.get_submodule(late).register_forward_hook(
    lambda m, i, out: shapes.__setitem__("late_out", out.shape)
  )
  backbone(torch.zeros(1, 3, 32, 32, device=device))
  h_early.remove()
  h_late.remove()
  return tuple(shapes["early_in"][1:]), tuple(shapes["late_out"][1:])


def build_communicative(
  specs: list[SpecialistSpec],
  key_dim: int,
  value_dim: int,
  num_classes: int,
  device: str,
  patch: int = 0,
) -> Orchestrator:
  specialists = []
  for spec in specs:
    model = _load_backbone(spec.backbone_path, device)
    # per-spec, not once from specialists[0]: early/late can now differ per
    # specialist, so each one's own adapter dims must be measured separately
    early_shape, late_shape = _adapter_dims(model, spec.early, spec.late, device)
    encoder_cls, reduce_factory = ENCODERS[spec.encoder]
    if patch == 0:
      # one message per specialist; Study 02's adapters size themselves against
      # the full port shapes and emit (b, d), with no patch axis
      decoder = DECODERS[spec.decoder](value_dim, early_shape)
      encoder = encoder_cls(late_shape[0], key_dim, value_dim, reduce_factory(late_shape))
    else:
      # late grid -> msg_h*msg_w blocks of patch*patch. each block gets its own
      # attention over the n specialists, then expands to up_h*up_w on the early grid.
      # Defined only for the plain adapters: the spatial ones already read and
      # write the whole grid, so patching them has no meaning
      if reduce_factory is not gap_reduce or spec.decoder not in ("broadcast", "mlp"):
        raise ValueError(f"--patch {patch} needs a pooled encoder and the broadcast "
                         f"decoder, not {spec.encoder}/{spec.decoder}")
      from einops.layers.torch import Rearrange
      (early_ch, eh, ew), (late_ch, lh, lw) = early_shape, late_shape
      msg_h, msg_w = lh // patch, lw // patch
      up_h, up_w = eh // msg_h, ew // msg_w
      assert (lh, lw) == (msg_h * patch, msg_w * patch) and (eh, ew) == (msg_h * up_h, msg_w * up_w), \
        f"patch {patch} doesnt tile late {(lh, lw)} / early {(eh, ew)}"
      reduce = Rearrange("b c (msg_h ph) (msg_w pw) -> b (msg_h msg_w) (c ph pw)",
                         ph=patch, pw=patch)
      expand = Rearrange("b (msg_h msg_w) (c up_h up_w) -> b c (msg_h up_h) (msg_w up_w)",
                         msg_h=msg_h, up_h=up_h, up_w=up_w)
      decoder = Decoder(value_dim, early_ch * up_h * up_w, expand)
      encoder = encoder_cls(late_ch * patch * patch, key_dim, value_dim, reduce)
    specialists.append(Specialist(
      Driver(model, spec.early, spec.late), decoder, encoder,
    ))
  return Orchestrator(specialists, key_dim, value_dim, num_classes).to(device)


def _resolve_backbone_paths(manifest: Optional[Path], backbones: Optional[list[Path]]) -> list[Path]:
  """Exactly one of manifest/backbones is set (enforced by an argparse
  mutually-exclusive group), both resolve to the same ordered list."""
  if manifest is not None:
    data = json.loads(manifest.read_text())
    return [Path(p) for p in data["backbones"]]
  return list(backbones)


# ---------------------------------------------------------------------------
# Aggregation across seed replicates
# ---------------------------------------------------------------------------

_SUMMARY_METRICS = ("val_accuracy", "val_loss", "test_accuracy", "test_loss",
                    "test_accuracy_endpoint")

# docs/studies/02-spatial-ports.md §4.3: the accuracy endpoint is test accuracy
# averaged over a window centred on the best validation epoch, not the value at
# termination. Study 01's best validation epoch ranged from 0 to 36 of 70, so
# the window runs off the start of a run in practice and is clipped.
ENDPOINT_WINDOW = 5


def windowed_endpoint(
  val_acc: list[float], test_acc: list[float], width: int = ENDPOINT_WINDOW
) -> tuple[float, int]:
  """(mean test accuracy over the window, the validation-selected epoch).

  Selecting a single epoch would be selecting on noise: the validation curve
  has no sharp optimum, so the window averages away the part of the choice that
  is arbitrary while keeping the part that is not. Test accuracy is read at the
  chosen epochs and never consulted in choosing them.
  """
  if not val_acc or not test_acc:
    raise ValueError("endpoint needs per-epoch val and test accuracy")
  best = max(range(len(val_acc)), key=lambda i: val_acc[i])
  half = width // 2
  lo, hi = max(0, best - half), min(len(test_acc), best + half + 1)
  window = test_acc[lo:hi]
  return sum(window) / len(window), best


def summarize_replicates(experiment: str, seeds: list[int], repo: Optional[str] = None) -> Run:
  """Aggregate a group of N_2 joint-training replicates — same config,
  different --seed, same --experiment (scripts/replicate_joint.py) — into
  one distinguished summary Run in that same experiment: mean/std of each
  final metric across seeds.

  Runs are matched by hparams.seed membership in `seeds`, not by an Aim
  query on a nested field — simpler, and doesn't rely on the query DSL
  reaching into hparams. Every seed must be present: a partial sweep raises
  instead of silently averaging a subset and reporting it as the full group.
  """
  repo_obj = Repo(repo) if repo else Repo.default_repo()
  # A run's hash/experiment are visible to other processes immediately, but
  # its params (hparams, final) and tracked metric sequences are not — Aim
  # only reconciles those into the queryable index on demand (what `aim
  # storage reindex` does), and nothing triggers that automatically for
  # SDK-only writes with no `aim up` server running. Without this, `r["final"]`
  # below raises even for a run that completed and wrote it. Mirrors
  # `aim storage reindex --yes`; private API, but it's what that command
  # itself calls, and a full rebuild over this project's run count is <1s.
  repo_obj._recreate_index()
  candidates = repo_obj.query_runs(f"run.experiment == '{experiment}'").iter_runs()
  matched = [item.run for item in candidates if item.run.get("hparams", {}).get("seed") in seeds]
  if len(matched) != len(seeds):
    found = sorted(r.get("hparams", {}).get("seed") for r in matched)
    raise SystemExit(
      f"expected {len(seeds)} replicate runs for seeds {sorted(seeds)} in "
      f"experiment {experiment!r}, found {len(matched)} (seeds present: {found})"
    )

  summary = Run(experiment=experiment, repo=repo)
  summary.add_tag("summary")
  summary["n_replicates"] = len(matched)
  summary["summarized_seeds"] = sorted(seeds)
  summary["metric_docs"] = metric_docs.summary_as_dict()
  for metric in _SUMMARY_METRICS:
    values = [r["final"][metric] for r in matched]
    summary.track(float(np.mean(values)), name=f"{metric}_mean")
    summary.track(float(np.std(values)), name=f"{metric}_std")
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description="Train an ensemble over frozen backbones")
  sub = parser.add_subparsers(dest="method", required=True)

  p = sub.add_parser("communicative", help="TarMAC-aligned communicative ensemble")
  backbone_group = p.add_mutually_exclusive_group(required=True)
  backbone_group.add_argument("--manifest", type=Path,
                 help="selection/ manifest JSON — trains on its backbones list, in order")
  backbone_group.add_argument("--backbones", type=Path, nargs="+",
                 help="Backbone checkpoint paths (≥2), for ad hoc runs outside selection")
  p.add_argument("--specialist-config", type=Path, default=None,
                 help='JSON {"<index>": {early?, late?, encoder?, decoder?}} sparse '
                      "per-specialist overrides, keyed by position in the resolved "
                      "backbone list — only positions differing from --early/--late "
                      "need an entry")
  p.add_argument("--split-file", type=Path, default=_REPO_ROOT / "splits/three_way_seed42.pt")
  p.add_argument("--early", default="layer1.2", help="Decoder injection layer (default for all specialists)")
  p.add_argument("--late", default="layer3.0", help="Encoder read layer (default for all specialists)")
  p.add_argument("--encoder", default="qkv", choices=sorted(ENCODERS),
                 help="Encoder head architecture (default for all specialists)")
  p.add_argument("--decoder", default="mlp", choices=sorted(DECODERS),
                 help="Decoder head architecture (default for all specialists)")
  p.add_argument("--key-dim", type=int, default=16)
  p.add_argument("--value-dim", type=int, default=64)
  p.add_argument("--patch", type=int, default=0,
                 help="Patch size k over the late grid; one attention per k*k patch "
                      "(1 = per-position, 0 = mean-pool to one message). Must tile "
                      "both the late and early grids.")
  p.add_argument("--k-rounds", type=int, default=1,
                 help="Communication rounds (backbone passes = k_rounds + 1)")
  p.add_argument("--epochs", type=int, default=10)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--weight-decay", type=float, default=1e-4)
  p.add_argument("--batch-size", type=int, default=128)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                 help="cudnn.deterministic (default: on). --no-deterministic trades exact "
                      "reproducibility for cudnn's autotuned kernels.")
  p.add_argument("--num-classes", type=int, default=100)
  p.add_argument("--experiment", default="communicative", help="Aim experiment name")
  p.add_argument("--no-track", action="store_true", help="Disable Aim tracking")
  p.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "ensemble/checkpoints")
  p.add_argument("--data-root", type=str, default=str(_REPO_ROOT / "data"))
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

  ps = sub.add_parser("summarize", help="Aggregate a seed-replicate group into a summary run")
  ps.add_argument("--experiment", required=True, help="Aim experiment name shared by the replicate runs")
  ps.add_argument("--seeds", type=int, nargs="+", required=True, help="Joint seeds to aggregate")
  ps.add_argument("--repo", type=str, default=None, help="Aim repo path (default: Aim's default repo)")

  args = parser.parse_args()

  if args.method == "summarize":
    run = summarize_replicates(args.experiment, args.seeds, repo=args.repo)
    print(f"[summarize] wrote summary run {run.hash} in experiment {args.experiment!r}")
    return

  backbone_paths = _resolve_backbone_paths(args.manifest, args.backbones)
  if len(backbone_paths) < 2:
    parser.error("need at least 2 backbones")

  seed_everything(args.seed, args.deterministic)

  cfg = TrainConfig(
    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
    seed=args.seed, device=args.device,
  )

  train_loader, val_loader, test_loader = make_loaders(
    args.split_file, args.batch_size, args.data_root
  )
  specs = resolve_specialist_specs(backbone_paths, args, args.specialist_config)
  orchestrator = build_communicative(
    specs, args.key_dim, args.value_dim, args.num_classes, args.device, args.patch,
  )
  n_trainable = sum(p.numel() for p in orchestrator.parameters() if p.requires_grad)
  print(f"[communicative] {len(backbone_paths)} backbones  k_rounds={args.k_rounds}  "
        f"trainable params={n_trainable:,}")

  # the orchestrator readout is log of prob-averaged member softmaxes, so the
  # loss pairs it with NLLLoss (CrossEntropyLoss would double-apply log_softmax)
  criterion = nn.NLLLoss()

  # static reference: the untrained prob-averaging ensemble (k_rounds=0 is a
  # plain mean over the frozen backbones' softmaxes — communication switched off)
  baseline = {}
  for name, loader in (("val", val_loader), ("test", test_loader)):
    res = evaluate(orchestrator, loader, device=args.device, criterion=criterion, k_rounds=0)
    baseline.update({f"{name}_{k}": v for k, v in res.items()})
  print(f"[baseline avg-probs] {baseline}")

  run = None
  if not args.no_track:
    run = Run(experiment=args.experiment)
    run["hparams"] = {
      **asdict(cfg),
      "method": "communicative",
      "backbones": [str(p) for p in backbone_paths],
      "manifest": str(args.manifest) if args.manifest else None,
      "specialist_config": str(args.specialist_config) if args.specialist_config else None,
      "split_file": str(args.split_file),
      "early": args.early, "late": args.late,
      "encoder": args.encoder, "decoder": args.decoder,
      "key_dim": args.key_dim, "value_dim": args.value_dim,
      "k_rounds": args.k_rounds, "patch": args.patch,
      "batch_size": args.batch_size,
      "deterministic": args.deterministic,
    }
    run["baseline_avg_probs"] = baseline
    run["metric_docs"] = metric_docs.as_dict()
    print(f"aim run hash: {run.hash}")

  optimizer = torch.optim.AdamW(
    (p for p in orchestrator.parameters() if p.requires_grad),
    lr=cfg.lr, weight_decay=cfg.weight_decay,
  )
  # the same batch every epoch and in every arm: these statistics are partly a
  # property of the batch they are measured on, so a fixed one is what makes
  # them comparable along training and across the grid (§5.2)
  diag_batch, _ = next(iter(val_loader))

  job = TrainJob(
    model=orchestrator, loader=train_loader, optimizer=optimizer,
    criterion=criterion, config=cfg, run=run, val_loader=val_loader,
    test_loader=test_loader, diag_batch=diag_batch,
  )
  train(job, k_rounds=args.k_rounds)

  results = {}
  for name, loader in (("val", val_loader), ("test", test_loader)):
    results[name] = evaluate(
      orchestrator, loader, device=cfg.device, criterion=criterion, k_rounds=args.k_rounds
    )
    print(f"{name}: {results[name]}")
    if run is not None:
      # final_ prefix keeps these single points out of the per-epoch series
      for k, v in results[name].items():
        run.track(v, name=f"final_{name}_{k}")

  # flat {val_accuracy, val_loss, test_accuracy, test_loss} — also stored as a
  # plain run param (not just .track()'d) so summarize_replicates can read it
  # back with a dict lookup instead of a metric-sequence query
  final_flat = {f"{name}_{k}": v for name, res in results.items() for k, v in res.items()}

  history = job.epoch_history
  if history.get("val_accuracy") and history.get("test_accuracy"):
    endpoint, best_epoch = windowed_endpoint(
      history["val_accuracy"], history["test_accuracy"])
    final_flat["test_accuracy_endpoint"] = endpoint
    final_flat["endpoint_epoch"] = best_epoch
    print(f"endpoint: test_accuracy {endpoint:.4f} "
          f"over {ENDPOINT_WINDOW} epochs centred on epoch {best_epoch + 1}")
    if run is not None:
      run.track(endpoint, name="final_test_accuracy_endpoint")
      run.track(best_epoch, name="final_endpoint_epoch")

  if run is not None:
    run["final"] = final_flat

  # architecture + seed are in the name, not just the backbones: every arm of an
  # architecture study shares one fixed expert group, so a backbone-only name
  # makes every arm and every seed collide on one path and silently overwrite.
  # BOTH adapters belong here — with the encoder alone, two arms differing only
  # in --decoder overwrite each other seed for seed.
  stems = "_".join(p.stem for p in backbone_paths)
  arch = (f"{args.encoder}_{args.decoder}_p{args.patch}"
          f"_k{args.key_dim}_v{args.value_dim}_r{args.k_rounds}")
  out = args.output_dir / f"communicative_{stems}_{arch}_seed{args.seed}.pt"
  out.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
    "state_dict": orchestrator.state_dict(),
    "metadata": {
      "method": "communicative",
      "backbones": [str(p) for p in backbone_paths],
      "split_file": str(args.split_file),
      "early": args.early, "late": args.late,
      "encoder": args.encoder, "decoder": args.decoder,
      "key_dim": args.key_dim, "value_dim": args.value_dim,
      "k_rounds": args.k_rounds, "patch": args.patch,
      "baseline_avg_probs": baseline,
      **final_flat,
      **asdict(cfg),
    },
  }, out)
  print(f"saved → {out}")


if __name__ == "__main__":
  main()

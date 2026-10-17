import math
from contextlib import ExitStack

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from ensemble.communicative.driver import Driver
from ensemble.communicative.bus import QKVEncoder, Decoder, PatchAttention, TarMACBus
from ensemble.communicative.hooks import Injection, HookState, hook_encoder, hook_inject

from typing import Annotated

def attn_entropy(attn: t.Tensor) -> float:
  return -(attn * (attn + 1e-12).log()).sum(dim=-1).mean().item()


def attn_top1(attn: t.Tensor) -> float:
  return attn.max(dim=-1).values.mean().item()


class Specialist(nn.Module):
  def __init__(self, driver: Driver, decoder: Decoder, encoder: QKVEncoder,
               inject: Injection, patch_attn: PatchAttention | None = None):
    super().__init__()
    self.driver = driver
    self.decoder = decoder
    self.encoder = encoder
    self.inject = inject
    self.patch_attn = patch_attn


class Orchestrator(nn.Module):
  def __init__(self, specialists: list[Specialist], bus: TarMACBus, key_dim: int, value_dim: int, num_classes: int):
    super().__init__()
    self.specialists = nn.ModuleList(specialists)
    self.bus = bus

    self.key_dim = key_dim
    self.value_dim = value_dim
    self.num_classes = num_classes

    # message magnitudes from the most recent communication round (see forward);
    # stays None until a forward with k_rounds >= 1 runs
    self.last_stats: dict[str, float] | None = None
    # mean per-row entropy of the bus's attention weights, averaged across the
    # batch — same round as last_stats, None under the same conditions
    self.last_attn_entropy: float | None = None
    # mean peak attention weight max_i p_i, averaged over receivers and batch —
    # the other half of the concentration picture: entropy is a whole-row
    # summary, this is just how much mass the winner took
    self.last_attn_top1: float | None = None
    # the bus's post-softmax attention itself, (b, p, n, n), detached — entropy
    # says how concentrated routing is, this says who it concentrated *on*
    self.last_attn: t.Tensor | None = None
    # same as last_attn_entropy, for the attention across patches
    self.last_patch_attn_entropy: float | None = None
    # per-specialist logits before (perception pass) and after communication,
    # detached — lets the trainer measure how messages changed the answers
    self.last_shift: dict[str, t.Tensor] | None = None
    # (n, b, c) logits from the last pass, kept w/ grad so train() can add a loss per member
    self.last_logits: t.Tensor | None = None

  def train(self, mode: bool = True):
    super().train(mode)
    for s in self.specialists:
      assert isinstance(s, Specialist)
      s.driver.backbone.eval() # keep frozen backbones in eval so batchnorm stats dont drift
    return self

  def _register_hooks(self, stack: ExitStack, decoder_outs: list[HookState], encoder_ins: list[HookState]):
    for i, s in enumerate(self.specialists):
      assert isinstance(s, Specialist)
      stack.enter_context(hook_inject(s.driver.early, decoder_outs[i], s.inject.combine))
      stack.enter_context(hook_encoder(s.driver.late, encoder_ins[i]))

  def forward(self, xs: t.Tensor | list[t.Tensor], k_rounds: int = 1) -> t.Tensor:
    n: Annotated[int, "num_specialists"] = len(self.specialists)
    if isinstance(xs, t.Tensor):
      xs = [xs] * n
    b: Annotated[int, "batch_size"] = xs[0].shape[0]

    device = xs[0].device

    # decoder states start as None: the first pass is pure perception, no injection
    decoder_outs: list[HookState] = [HookState() for _ in range(n)]
    encoder_ins: list[HookState] = [HookState() for _ in range(n)]
    outputs = t.zeros((n, b, self.num_classes), device=device)

    self.last_stats = None
    self.last_attn_entropy = None
    self.last_attn_top1 = None
    self.last_attn = None
    self.last_patch_attn_entropy = None
    self.last_shift = None
    pre_logits = None

    with ExitStack() as stack:
      # register encoder/decoder hooks
      self._register_hooks(stack, decoder_outs, encoder_ins)

      # perception pass + k communication rounds; Q/K/V are recomputed each
      # pass from the updated late features (recurrence via the backbone)
      for round_idx in range(k_rounds + 1):
        qkv = []
        for i, s in enumerate(self.specialists):
          assert isinstance(s, Specialist)
          outputs[i] = s.driver.backbone(xs[i])
          qkv.append(s.encoder(encoder_ins[i].value))

        # encoders emit (b, p, d), p=1 when pooled; specialist axis goes just
        # before the feature dim
        Q, K, V = (t.stack(x, dim=-2) for x in zip(*qkv))

        if round_idx == k_rounds:
          break

        if round_idx == 0:
          # opinions before any message arrives
          pre_logits = outputs.detach().clone()

        # aggregate messages and stage them for injection on the next pass
        c, attn = self.bus(Q, K, V)
        patch_attns = []
        for i, s in enumerate(self.specialists):
          assert isinstance(s, Specialist)
          ci = c[..., i, :]  # (b, p, d)
          if s.patch_attn is not None:
            ci, pattn = s.patch_attn(ci)
            patch_attns.append(pattn)
          decoder_outs[i].value = s.decoder(ci)

        # mean L2 norms of what is said, what is heard, and what is injected —
        # a collapse toward zero here means the continuous messages vanished
        with t.no_grad():
          self.last_stats = {
            "value": V.norm(dim=-1).mean().item(),
            "aggregated": c.norm(dim=-1).mean().item(),
            "injection": t.stack([
              decoder_outs[i].value.flatten(1).norm(dim=1).mean() for i in range(n)
            ]).mean().item(),
          }
          self.last_attn_entropy = attn_entropy(attn)
          self.last_attn_top1 = attn_top1(attn)
          self.last_attn = attn.detach()
          if patch_attns:
            self.last_patch_attn_entropy = attn_entropy(t.stack(patch_attns))

    if pre_logits is not None:
      self.last_shift = {"pre": pre_logits, "post": outputs.detach()}
    self.last_logits = outputs

    # prob-averaged readout: log( mean_i softmax(logits_i) ), computed as a
    # logsumexp over per-member log-probs for numerical stability. Returns
    # LOG-PROBABILITIES, not logits — pair with NLLLoss, never CrossEntropyLoss
    log_probs: Annotated[t.Tensor, "b c"] = (
      t.logsumexp(F.log_softmax(outputs, dim=-1), dim=0) - math.log(n)
    )
    return log_probs

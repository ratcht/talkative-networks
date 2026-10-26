"""
A soft-gated mixture of frozen experts: the baseline that weighs the experts'
answers instead of letting them communicate (docs/testbed/moe.md).

Every expert runs unchanged. A gate reads every expert's features at one
point and gives each expert a weight for this input; the ensemble's answer is
the weighted average of the experts' probabilities. Only the gate trains. Its
last layer starts at zero, so the weights start equal and the ensemble starts
as bagging, like an untrained channel.

  architecture  the gate's, by name (GATES)
  read          the point it reads; the experts' last point by default
  width         how many readings it takes of each expert, as the
                linear_taper encoder does
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from testbed.communication.ensemble import Output, average
from testbed.experts.architectures import Expert
from testbed.seeds import seeded


@dataclass(frozen=True)
class GateConfig:
  architecture: str = "linear_taper_gelu"
  read: str | None = None
  width: int = 8


class LinearTaperGELUGate(nn.Module):
  """Reads each expert's whole feature map, every channel at every position,
  through `width` learned templates, as the linear_taper encoder does. A GELU
  and one linear layer over every expert's readings then score each expert.
  The scores start at zero, so the weights start equal."""

  def __init__(self, config: GateConfig, read_shape: tuple[int, ...], num_experts: int):
    super().__init__()
    self.reads = nn.ModuleList(
      nn.Linear(math.prod(read_shape), config.width) for _ in range(num_experts)
    )
    self.score = nn.Linear(num_experts * config.width, num_experts)
    nn.init.zeros_(self.score.weight)
    nn.init.zeros_(self.score.bias)

  def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
    """Every expert's features at the read point -> (B, M) scores."""
    readings = [read(f.flatten(1)) for read, f in zip(self.reads, features)]
    return self.score(F.gelu(torch.cat(readings, dim=-1)))


GATES: dict[str, type[nn.Module]] = {"linear_taper_gelu": LinearTaperGELUGate}


class GatedEnsemble(nn.Module):
  direction = "none"  # nothing is written into any expert

  def __init__(self, experts: list[Expert], gate: nn.Module, read: str):
    super().__init__()
    self.experts = nn.ModuleList(experts)
    self.experts.requires_grad_(False)
    self.gate = gate
    self.read = read

  def train(self, mode: bool = True):
    super().train(mode)
    self.experts.eval()  # frozen experts never leave eval mode
    return self

  @property
  def trainable(self) -> nn.Module:
    """What joint training trains: the gate."""
    return nn.ModuleDict({"gate": self.gate})

  def forward(self, x: torch.Tensor, answer_each_round: bool = False) -> Output:
    """answer_each_round is for a backward exchange's rounds; a gate has none."""
    with torch.no_grad():  # the experts run unchanged, so no gradient flows into them
      features = [expert.run(x, until=self.read) for expert in self.experts]
      logits = torch.stack(
        [expert.run(f, after=self.read) for expert, f in zip(self.experts, features)],
        dim=1,
      )
    log_weights = F.log_softmax(self.gate(features), dim=-1)  # (B, M)
    log_probs = torch.logsumexp(
      F.log_softmax(logits, dim=-1) + log_weights[..., None], dim=1
    )
    return Output(log_probs, logits, [], log_weights.exp())

  def bagging(self, x: torch.Tensor) -> Output:
    """The same experts with equal weights: the zero point."""
    logits = torch.stack([expert(x) for expert in self.experts], dim=1)
    return Output(average(logits), logits, [])


def build_gated(
  experts: list[Expert], config: GateConfig, input_shape: tuple[int, ...], seed: int
) -> GatedEnsemble:
  """Size the gate from the experts' features at its read point and
  initialise it on its own seed."""
  points = experts[0].points
  if any(e.points != points for e in experts):
    raise ValueError("experts must share their points")
  read = config.read or points[-1]
  if read not in points:
    raise ValueError(f"the gate can't read at {read!r}; the experts have {points}")
  shape = experts[0].point_shapes(input_shape)[read]
  with seeded(seed):
    gate = GATES[config.architecture](config, shape, len(experts))
  device = next(experts[0].parameters()).device
  return GatedEnsemble(experts, gate, read).to(device)

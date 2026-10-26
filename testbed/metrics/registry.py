"""
Every metric the testbed logs: its name, its MLflow keys, and one line on what
it is. Each name has a section in docs/testbed/metrics.md with the full
definition and how to read it. The tests check that every key a run logs
matches an entry here, and that every entry has its section.

Placeholders in keys:
  {split}  joint, joint_val or test
  {site}   where in the ensemble: ex0, ex1, ... for forward exchanges, or
           ex0/round0, ex0/round1, ... for the rounds of a backward exchange
  {i}      an expert, numbered from 0 in group order
  anything else in braces is one of the names the section lists
"""

import re
from dataclasses import dataclass

PLACEHOLDER = re.compile(r"\{(\w+)\}")
PATTERNS = {"site": r"ex\d+(?:/round\d+)?", "i": r"\d+"}


@dataclass(frozen=True)
class Metric:
  name: str
  category: str
  keys: tuple[str, ...]
  summary: str

  def matches(self, key: str) -> bool:
    return any(re.fullmatch(_regex(k), key) for k in self.keys)


def _regex(key: str) -> str:
  parts = PLACEHOLDER.split(key)  # literal, name, literal, name, ..., literal
  regex = re.escape(parts[0])
  for name, literal in zip(parts[1::2], parts[2::2]):
    regex += PATTERNS.get(name, "[^/]+") + re.escape(literal)
  return regex


METRICS = (
  Metric(
    "accuracy",
    "Task performance",
    ("{split}/accuracy",),
    "Share of examples the ensemble answers correctly. At step 0 it is bagging's.",
  ),
  Metric(
    "loss",
    "Task performance",
    ("{split}/loss",),
    "The training objective (--loss) on the split.",
  ),
  Metric(
    "expert_accuracy",
    "Task performance",
    ("{split}/expert_accuracy",),
    "Share of the experts' own answers that are correct, after communication.",
  ),
  Metric(
    "answer_grid",
    "Communication interpretation",
    ("{split}/answer_grid/{cell}", "answer_grid/{cell}/{site}"),
    "Shares of examples by how many experts are right (all, some, none) and "
    "whether the averaged answer is. The five cells sum to 1. Also after each "
    "round of a backward exchange.",
  ),
  Metric(
    "joint_answer_shift",
    "Communication interpretation",
    ("{split}/joint_answer_shift/{kind}",),
    "Answers communication corrected and corrupted, their difference (net), "
    "and how often experts agree before and after.",
  ),
  Metric(
    "marginal_answer_shift",
    "Communication interpretation",
    ("joint_val/marginal_answer_shift/{kind}/expert{i}",),
    "joint_answer_shift with only expert i's update applied.",
  ),
  Metric(
    "answer_shift_kl",
    "Communication interpretation",
    ("joint_val/answer_shift_kl",),
    "How far communication moved the experts' answer probabilities, "
    "including moves too small to change an answer.",
  ),
  Metric(
    "routed_correct",
    "Communication interpretation",
    ("routed_correct/{subset}/{site}",),
    "How often the expert a receiver listened to most was right before communication.",
  ),
  Metric(
    "msg_norm",
    "Communication content",
    ("msg_norm/{part}/{site}",),
    "Average size of what is sent (value), received (aggregated) and injected (injection).",
  ),
  Metric(
    "attn_entropy",
    "Communication content",
    ("attn_entropy/{site}",),
    "How evenly each receiver spreads its attention over senders: 0 is one "
    "sender, log M is all equally.",
  ),
  Metric(
    "attn_top1",
    "Communication content",
    ("attn_top1/{site}",),
    "Each receiver's largest attention weight: 1/M is uniform, 1 is one sender.",
  ),
  Metric(
    "routing_mode_share",
    "Communication content",
    ("routing_mode_share/joint/{site}", "routing_mode_share/expert{i}/{site}"),
    "Share of examples routed the most common way. 1 means routing ignores the input.",
  ),
  Metric(
    "self_attn_rate",
    "Communication content",
    ("self_attn_rate/{site}",),
    "How often a receiver listens most to its own message.",
  ),
  Metric(
    "route_flip_rate",
    "Communication content",
    ("route_flip_rate/{site}",),
    "Share of routes that changed since the previous step.",
  ),
  Metric(
    "inj_spread",
    "Communication content",
    ("inj_spread/{site}",),
    "Share of the injection that differs between examples. 0 is a constant shift, not a message.",
  ),
  Metric(
    "inj_resid_frac",
    "Communication content",
    ("inj_resid_frac/{site}",),
    "Share of the injection that differs between positions. 0 for the broadcast decoder.",
  ),
  Metric(
    "inj_rank",
    "Communication content",
    ("inj_rank/{site}",),
    "Effective number of distinct spatial patterns in the injection.",
  ),
  Metric(
    "msg_rank_value",
    "Communication content",
    ("msg_rank_value/expert{i}/{site}", "msg_rank_value/mean/{site}"),
    "Effective number of directions an expert's sent message varies along.",
  ),
  Metric(
    "msg_rank_aggregated",
    "Communication content",
    ("msg_rank_aggregated/expert{i}/{site}", "msg_rank_aggregated/mean/{site}"),
    "Effective number of directions an expert's received message varies along.",
  ),
  Metric(
    "msg_rank_ratio",
    "Communication content",
    ("msg_rank_ratio/{site}",),
    "Directions received relative to directions sent. Above 1, mixing combines senders.",
  ),
  Metric(
    "grad_norm",
    "Training dynamics",
    ("grad_norm/total", "grad_norm/{component}/ex{e}", "grad_norm/gate.{part}"),
    "Size of the gradient reaching each part of the channel, or of the MoE's gate.",
  ),
  Metric(
    "update_ratio",
    "Training dynamics",
    (
      "update_ratio/total",
      "update_ratio/{component}/ex{e}",
      "update_ratio/gate.{part}",
    ),
    "How far one optimiser step moves each part, relative to its size.",
  ),
)


def find(key: str) -> Metric | None:
  return next((m for m in METRICS if m.matches(key)), None)


def description() -> str:
  """The registry as a Markdown table, for the MLflow experiment's description."""
  rows = [
    "Full definitions: docs/testbed/metrics.md. Step 0 is before training.",
    "",
    "| metric | keys | what it is |",
    "|---|---|---|",
  ]
  for m in METRICS:
    keys = "<br>".join(f"`{k}`" for k in m.keys)
    rows.append(f"| `{m.name}` | {keys} | {m.summary} |")
  return "\n".join(rows)

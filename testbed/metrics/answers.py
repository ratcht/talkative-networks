"""
Metrics on the ensemble's answers: how accurate it is, and how communication
changed what it answers. Definitions: docs/testbed/metrics.md.

Logits are (N, M, classes): `pre` from the experts with no communication,
which is bagging, and `post` from the communicating ensemble. The experts are
frozen, so `pre` is fixed for a given split.
"""

import torch
import torch.nn.functional as F

from testbed.communication.ensemble import average


def accuracy(post: torch.Tensor, y: torch.Tensor) -> float:
  """Share of examples the ensemble answers correctly."""
  return rate(average(post).argmax(-1) == y)


def expert_accuracy(post: torch.Tensor, y: torch.Tensor) -> float:
  """Share of (example, expert) pairs where the expert's own answer is correct."""
  return rate(post.argmax(-1) == y[:, None])


def answer_grid(post: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
  """Shares of examples by how many experts answer correctly (all, some or
  none) and whether the averaged answer does. Every example is in exactly one
  cell. With every expert right the average is right too, so that cell has no
  split."""
  right = post.argmax(-1) == y[:, None]  # (N, M) each expert's own answer
  avg_right = average(post).argmax(-1) == y
  all_right, none_right = right.all(1), ~right.any(1)
  some_right = ~all_right & ~none_right
  return {
    "all_right": rate(all_right),
    "some_right_avg_right": rate(some_right & avg_right),
    "some_right_avg_wrong": rate(some_right & ~avg_right),
    "none_right_avg_right": rate(none_right & avg_right),
    "none_right_avg_wrong": rate(none_right & ~avg_right),
  }


def joint_answer_shift(
  pre: torch.Tensor, post: torch.Tensor, y: torch.Tensor
) -> dict[str, float]:
  """Which of the ensemble's answers communication corrected and corrupted."""
  right_pre = average(pre).argmax(-1) == y
  right_post = average(post).argmax(-1) == y
  corrected, corrupted = rate(~right_pre & right_post), rate(right_pre & ~right_post)
  return {
    "pre_accuracy": rate(right_pre),
    "post_accuracy": rate(right_post),
    "corrected": corrected,
    "corrupted": corrupted,
    "net": corrected - corrupted,
    "agreement_pre": agreement(pre),
    "agreement_post": agreement(post),
  }


def marginal_answer_shift(
  pre: torch.Tensor, post: torch.Tensor, y: torch.Tensor
) -> dict[str, float]:
  """joint_answer_shift's corrected, corrupted and net when only expert i's
  post-communication logits replace its pre-communication ones, for each i."""
  right_pre = average(pre).argmax(-1) == y
  out = {}
  for i in range(pre.shape[1]):
    mixed = pre.clone()
    mixed[:, i] = post[:, i]
    right = average(mixed).argmax(-1) == y
    corrected, corrupted = rate(~right_pre & right), rate(right_pre & ~right)
    out |= {
      f"corrected/expert{i}": corrected,
      f"corrupted/expert{i}": corrupted,
      f"net/expert{i}": corrected - corrupted,
    }
  return out


def answer_shift_kl(pre: torch.Tensor, post: torch.Tensor) -> float:
  """KL(post || pre) between each expert's answer probabilities, averaged over
  experts and examples."""
  log_pre, log_post = F.log_softmax(pre, -1), F.log_softmax(post, -1)
  return (log_post.exp() * (log_post - log_pre)).sum(-1).mean().item()


def agreement(logits: torch.Tensor) -> float:
  """Share of expert pairs giving the same answer, averaged over examples."""
  answers = logits.argmax(-1)  # (N, M)
  m = answers.shape[1]
  same = (answers[:, :, None] == answers[:, None, :]).float().mean(0)  # (M, M)
  return ((same.sum() - m) / (m * (m - 1))).item()  # the diagonal is all ones


def rate(mask: torch.Tensor) -> float:
  return mask.float().mean().item()

"""
What the testbed measures. Every metric's definition, and how to read it, is in
docs/testbed/metrics.md.

  answers.py       the ensemble's answers: accuracy, and how communication changed them
  exchanges.py     one exchange's messages: what was sent, how it was weighted, what was injected
  optimisation.py  whether each part of the channel gets gradient and moves
  registry.py      every logged metric, its MLflow keys and a one-line summary
"""

from testbed.metrics.answers import (
  accuracy,
  answer_shift_kl,
  expert_accuracy,
  joint_answer_shift,
  marginal_answer_shift,
)
from testbed.metrics.exchanges import ExchangeMeter
from testbed.metrics.optimisation import OptimisationMeter
from testbed.metrics.registry import METRICS, description

__all__ = [
  "METRICS",
  "ExchangeMeter",
  "OptimisationMeter",
  "accuracy",
  "answer_shift_kl",
  "description",
  "expert_accuracy",
  "joint_answer_shift",
  "marginal_answer_shift",
]

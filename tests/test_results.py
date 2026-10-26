"""
The rule that boils each run down to one number (testbed/results.py): the
median over the 5 epochs around the best joint_val epoch.

  uv run pytest tests/test_results.py
"""

from testbed.results import epochs_around_best, median_around_best


def history(val: list[float], test: list[float]) -> list[dict[str, float]]:
  """Steps 0, 1, 2, ...: step 0 is bagging, the rest are trained epochs."""
  return [{"joint_val/accuracy": v, "test/accuracy": t} for v, t in zip(val, test)]


def test_the_window_is_centred_on_the_best_trained_epoch():
  # step 0's joint_val is the highest, but it is bagging, before any training
  run = history(
    [0.99, 0.1, 0.2, 0.3, 0.9, 0.3, 0.2, 0.1, 0.1], [0.5, 1, 2, 3, 4, 5, 6, 7, 8]
  )
  assert list(epochs_around_best(run)) == [2, 3, 4, 5, 6]
  assert median_around_best(run) == 4


def test_the_window_shifts_inward_at_either_end_of_the_run():
  first = [0, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
  assert list(epochs_around_best(history(first, first))) == [1, 2, 3, 4, 5]
  last = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.9]
  assert list(epochs_around_best(history(last, last))) == [3, 4, 5, 6, 7]


def test_the_earliest_best_epoch_wins_a_tie():
  val = [0, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]
  assert list(epochs_around_best(history(val, val))) == [1, 2, 3, 4, 5]


def test_short_runs_use_what_they_have():
  assert list(epochs_around_best(history([0.5, 0.8], [0.6, 0.7]))) == [1]
  bagging = history([0.5], [0.6])  # no exchange, so no training
  assert list(epochs_around_best(bagging)) == [0]
  assert median_around_best(bagging) == 0.6


def test_one_odd_epoch_does_not_move_the_median():
  run = history([0, 0.1, 0.2, 0.9, 0.2, 0.1], [0, 0.90, 0.91, 0.50, 0.92, 0.93])
  assert median_around_best(run) == 0.91

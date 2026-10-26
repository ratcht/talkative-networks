"""
The one number each run is compared by.

A run logs every metric after every epoch of joint training, so it is a curve.
To compare arms over many trial seeds, each run is boiled down to one number
per metric, by a rule fixed before any run is looked at (docs/docket.md):

  1. take the trained epoch with the highest joint_val accuracy;
  2. take the 5 trained epochs around it;
  3. take the metric's median over those 5 epochs.

Arms are then compared by averaging those numbers over trial seeds. joint_val
picks the epochs and test is only read, so nothing selects on test.
"""

from statistics import median

WIDTH = 5
PICKED_BY = "joint_val/accuracy"


def epochs_around_best(history: list[dict[str, float]], width: int = WIDTH) -> range:
  """The `width` trained epochs centred on the one with the highest joint_val
  accuracy, the earliest on a tie. Near either end of the run the window
  shifts inward, so it keeps its width. Step 0 is bagging, before any
  training, so it is never in the window; a run with no training is step 0."""
  trained = len(history) - 1
  if trained < 1:
    return range(0, 1)
  best = max(range(1, trained + 1), key=lambda step: history[step][PICKED_BY])
  start = max(1, min(best - width // 2, trained - width + 1))
  return range(start, min(start + width, trained + 1))


def median_around_best(
  history: list[dict[str, float]], metric: str = "test/accuracy", width: int = WIDTH
) -> float:
  """A run's number for `metric`: its median over epochs_around_best."""
  return median(history[step][metric] for step in epochs_around_best(history, width))

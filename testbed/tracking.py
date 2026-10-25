"""
Experiment tracking behind a small interface.

MLflow is the backend: one SQLite file at the repo root, artifacts next to it.
NullTracker lets tests and quick checks run without it.

  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # from the repo root; port 5000
"""

from typing import Protocol

from testbed.data import REPO_ROOT

TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
ARTIFACTS = REPO_ROOT / "mlartifacts"


class Tracker(Protocol):
  def log_params(self, params: dict) -> None: ...
  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
  def set_tags(self, tags: dict) -> None: ...


class NullTracker:
  def __enter__(self) -> "NullTracker":
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    pass

  def log_params(self, params: dict) -> None:
    pass

  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    pass

  def set_tags(self, tags: dict) -> None:
    pass


class MlflowTracker:
  """One MLflow run. Use as a context manager; the run is marked FAILED if the block raises.

  `description` becomes the experiment's description in the MLflow UI (Markdown).
  """

  def __init__(
    self, experiment: str, run_name: str, description: str = "", uri: str = TRACKING_URI
  ):
    import mlflow  # imported here so nothing else in the package needs it

    self.mlflow = mlflow
    self.experiment, self.run_name, self.uri = experiment, run_name, uri
    self.description = description

  def __enter__(self) -> "MlflowTracker":
    self.mlflow.set_tracking_uri(self.uri)
    if self.mlflow.get_experiment_by_name(self.experiment) is None:
      self.mlflow.create_experiment(
        self.experiment, artifact_location=ARTIFACTS.as_uri()
      )
    self.mlflow.set_experiment(self.experiment)
    if self.description:
      self.mlflow.set_experiment_tag("mlflow.note.content", self.description)
    self.mlflow.start_run(run_name=self.run_name)
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    self.mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

  def log_params(self, params: dict) -> None:
    self.mlflow.log_params(flatten(params))

  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    self.mlflow.log_metrics(metrics, step=step)

  def set_tags(self, tags: dict) -> None:
    self.mlflow.set_tags(tags)


def flatten(params: dict, prefix: str = "") -> dict[str, object]:
  """{"channel": {"encoder": "pooled"}} -> {"channel.encoder": "pooled"}; MLflow params are flat."""
  flat = {}
  for key, value in params.items():
    name = f"{prefix}{key}"
    if isinstance(value, dict):
      flat |= flatten(value, f"{name}.")
    else:
      flat[name] = value
  return flat

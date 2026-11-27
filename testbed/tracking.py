"""
Experiment tracking behind a small interface.

MLflow is the backend. Runs go to the team's server (SERVER_URI, set up by
infra/mlflow_server/setup.sh) unless MLFLOW_TRACKING_URI names somewhere else.
NullTracker lets tests and quick checks run without either.

Once per machine, put your login in ~/.mlflow/credentials (chmod 600):

  [mlflow]
  mlflow_tracking_username = NAME
  mlflow_tracking_password = PASSWORD

The server's certificate comes from our own CA, whose certificate is in the repo
(SERVER_CA); nothing else needs installing. In a browser, the server is SERVER_URI.

To log to a local file instead (offline work), and browse it:

  export MLFLOW_TRACKING_URI=sqlite:///mlflow.db                # from the repo root
  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db      # port 5000

Runs made into a local file on another machine come in through testbed/merge_runs.py.
"""

import os
from typing import Protocol

from testbed.data import REPO_ROOT

SERVER_URI = "https://aisg-01.uwaterloo.ca:5050"
SERVER_CA = REPO_ROOT / "infra" / "mlflow_server" / "ca.crt"
LOCAL_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
ARTIFACTS = REPO_ROOT / "mlartifacts"


def tracking_uri() -> str:
  """Where runs go: MLFLOW_TRACKING_URI if it is set, otherwise the team's server."""
  return os.environ.get("MLFLOW_TRACKING_URI") or SERVER_URI


def use_server_ca() -> None:
  """Trust the team server's certificate, unless another certificate is already configured."""
  os.environ.setdefault("MLFLOW_TRACKING_SERVER_CERT_PATH", str(SERVER_CA))


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
  `uri` defaults to tracking_uri().
  """

  def __init__(
    self, experiment: str, run_name: str, description: str = "", uri: str | None = None
  ):
    import mlflow  # imported here so nothing else in the package needs it

    self.mlflow = mlflow
    self.experiment, self.run_name = experiment, run_name
    self.uri = uri or tracking_uri()
    self.description = description

  def __enter__(self) -> "MlflowTracker":
    if self.uri == SERVER_URI:
      use_server_ca()
    self.mlflow.set_tracking_uri(self.uri)
    self.mlflow.set_experiment(experiment_id=self._experiment_id())
    if self.description:
      self.mlflow.set_experiment_tag("mlflow.note.content", self.description)
    self.mlflow.start_run(run_name=self.run_name)
    return self

  def _experiment_id(self) -> str:
    """The experiment's id, creating the experiment if it doesn't exist yet.

    Two workers starting together can both find it missing. The second one's create then
    fails with RESOURCE_ALREADY_EXISTS, and it uses the first one's experiment instead.
    """
    from mlflow.exceptions import MlflowException

    found = self.mlflow.get_experiment_by_name(self.experiment)
    if found is not None:
      return found.experiment_id
    # A local file keeps artifacts beside it; a server decides where its own go.
    location = ARTIFACTS.as_uri() if self.uri.startswith("sqlite:") else None
    try:
      return self.mlflow.create_experiment(self.experiment, artifact_location=location)
    except MlflowException as e:
      if e.error_code != "RESOURCE_ALREADY_EXISTS":
        raise
      return self.mlflow.get_experiment_by_name(self.experiment).experiment_id

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

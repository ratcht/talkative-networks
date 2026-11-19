"""
Where MlflowTracker sends runs (testbed/tracking.py), on a real MLflow SQLite file in a
temporary directory. Nothing here talks to the team's server.

  uv run pytest tests/test_tracking.py
"""

import pytest

from testbed import tracking
from testbed.tracking import ARTIFACTS, SERVER_URI, MlflowTracker, tracking_uri

mlflow = pytest.importorskip("mlflow")


@pytest.fixture(scope="module")
def local(tmp_path_factory):
  """A local store. Building one runs every migration, so it is shared."""
  return f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"


def test_runs_go_to_the_team_server_by_default(monkeypatch):
  monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
  assert tracking_uri() == SERVER_URI
  assert MlflowTracker("x", "y").uri == SERVER_URI


def test_the_environment_redirects_runs(monkeypatch, local):
  monkeypatch.setenv("MLFLOW_TRACKING_URI", local)
  with MlflowTracker("redirected", "run-a", "about it") as tracker:
    tracker.log_metrics({"test/accuracy": 0.5}, step=0)

  client = mlflow.MlflowClient(tracking_uri=local)
  experiment = client.get_experiment_by_name("redirected")
  assert experiment.artifact_location == ARTIFACTS.as_uri()  # a local file keeps its own
  assert experiment.tags["mlflow.note.content"] == "about it"
  (run,) = client.search_runs([experiment.experiment_id])
  assert run.info.run_name == "run-a" and run.info.status == "FINISHED"
  assert run.data.metrics == {"test/accuracy": 0.5}


def test_a_worker_that_loses_the_race_joins_the_winners_experiment(monkeypatch, local):
  """The other worker creates the experiment between this one's lookup and its create."""
  existing = mlflow.MlflowClient(tracking_uri=local).create_experiment("raced")
  real_lookup = mlflow.get_experiment_by_name
  lookups = []

  def stale_first_lookup(name):
    lookups.append(name)
    return None if len(lookups) == 1 else real_lookup(name)

  monkeypatch.setattr(mlflow, "get_experiment_by_name", stale_first_lookup)
  with MlflowTracker("raced", "run-b", uri=local):
    pass

  assert len(lookups) == 2  # missed it, failed to create it, then found it
  (run,) = mlflow.MlflowClient(tracking_uri=local).search_runs([existing])
  assert run.info.run_name == "run-b"


def test_the_server_certificate_is_trusted_unless_another_is_set(monkeypatch):
  monkeypatch.delenv("MLFLOW_TRACKING_SERVER_CERT_PATH", raising=False)
  tracking.use_server_ca()
  assert tracking.os.environ["MLFLOW_TRACKING_SERVER_CERT_PATH"] == str(tracking.SERVER_CA)

  monkeypatch.setenv("MLFLOW_TRACKING_SERVER_CERT_PATH", "/elsewhere/ca.crt")
  tracking.use_server_ca()
  assert tracking.os.environ["MLFLOW_TRACKING_SERVER_CERT_PATH"] == "/elsewhere/ca.crt"

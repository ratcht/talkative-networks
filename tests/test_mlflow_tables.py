"""
Reading MLflow's tables directly (testbed/mlflow_tables.py), on a real MLflow SQLite
file in a temporary directory. The server side (Postgres through the tunnel) needs
the live server, so it isn't tested here.

  uv run pytest tests/test_mlflow_tables.py
"""

import pytest
from sqlalchemy.exc import OperationalError

from testbed.mlflow_tables import Tables

mlflow = pytest.importorskip("mlflow")


@pytest.fixture(scope="module")
def db(tmp_path_factory):
  """One experiment with one finished run: a tag, a param, a metric over 3 steps."""
  path = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
  client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{path}")
  experiment = client.create_experiment("study-x")
  run = client.create_run(experiment, run_name="arm-a").info.run_id
  client.set_tag(run, "trial_seed", "3")
  client.log_param(run, "channel.mixer", "mean")
  for step in range(3):
    client.log_metric(run, "test/accuracy", 0.5 + step / 10, step=step)
  client.set_terminated(run)
  return path


def test_named_parameters_read_the_tables(db):
  tables = Tables(db=db)
  ((run,),) = tables.rows(
    """select r.run_uuid from runs r join experiments e using (experiment_id)
       where e.name = :name and r.status = 'FINISHED'""",
    name="study-x",
  )
  assert dict(tables.rows('select "key", value from tags where run_uuid = :run', run=run))[
    "trial_seed"
  ] == "3"
  assert tables.rows("select key, value from params where run_uuid = :run", run=run) == [
    ("channel.mixer", "mean")
  ]
  steps = tables.rows(
    "select step, value from metrics where run_uuid = :run and key like :key order by step",
    run=run,
    key="test/%",
  )
  assert steps == [(0, 0.5), (1, 0.6), (2, 0.7)]


def test_it_cannot_write(db):
  with pytest.raises(OperationalError, match="readonly"):
    Tables(db=db).rows("delete from tags")


def test_a_missing_file_is_an_error_not_an_empty_database(tmp_path):
  with pytest.raises(FileNotFoundError):
    Tables(db=tmp_path / "nothing.db")

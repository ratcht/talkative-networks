"""
Merging an experiment from another machine's MLflow database (testbed/merge_runs.py),
on real MLflow databases in a temporary directory. What matters is what MLflow
itself reads back, so that is what these check.

  uv run pytest tests/test_merge_runs.py
"""

import shutil
import sqlite3

import pytest

from testbed.merge_runs import merge

mlflow = pytest.importorskip("mlflow")

NAME = "study-phase9"
LAB = "file:///home/lab/repo/mlartifacts"
HERE = "file:///home/here/repo/mlartifacts"


def client(path):
  return mlflow.MlflowClient(tracking_uri=f"sqlite:///{path}")


def finished_run(store, experiment, name, value):
  run = store.create_run(experiment, run_name=name)
  run_id = run.info.run_id
  store.log_param(run_id, "channel.mixer", "rmsnorm_attention")
  store.set_tag(run_id, "arm", name)
  for step in range(3):
    store.log_metric(run_id, "test/accuracy", value + step, step=step)
  store.set_terminated(run_id)
  return run_id


@pytest.fixture(scope="module")
def blank(tmp_path_factory):
  """An empty database. Building one runs every migration, so it is shared."""
  path = tmp_path_factory.mktemp("blank") / "blank.db"
  client(path).search_experiments()
  return path


@pytest.fixture
def machines(blank, tmp_path):
  """Two databases where the same experiment number means different things:
  the lab's first experiment is the one to merge, this machine's is another."""
  lab, here = tmp_path / "lab.db", tmp_path / "here.db"
  shutil.copy(blank, lab)
  shutil.copy(blank, here)
  store = client(lab)
  experiment = store.create_experiment(NAME, artifact_location=LAB)
  store.set_experiment_tag(experiment, "mlflow.note.content", "what the metrics are")
  runs = [finished_run(store, experiment, f"arm-{i}", float(i)) for i in range(3)]
  cut_off = store.create_run(experiment, run_name="cut-off").info.run_id
  noise = store.create_experiment("not-this-one", artifact_location=LAB)
  finished_run(store, noise, "other", 9.0)

  store = client(here)
  theirs = store.create_experiment("already-here", artifact_location=HERE)
  mine = finished_run(store, theirs, "mine", 5.0)
  return lab, here, runs, cut_off, mine, theirs


def test_runs_arrive_with_everything_they_own(machines):
  lab, here, runs, cut_off, mine, theirs = machines
  assert merge(lab, NAME, here, HERE, pause=0) == {"finished": 3, "copied": 3, "left": 1}

  store = client(here)
  experiment = store.get_experiment_by_name(NAME)
  assert experiment.experiment_id != theirs  # numbered here, not as on the lab
  assert experiment.artifact_location == HERE
  assert experiment.tags["mlflow.note.content"] == "what the metrics are"
  arrived = {r.info.run_id: r for r in store.search_runs([experiment.experiment_id])}
  assert set(arrived) == set(runs)
  assert client(lab).get_experiment_by_name("not-this-one")
  assert store.get_experiment_by_name("not-this-one") is None

  for i, run_id in enumerate(runs):
    there = client(lab).get_run(run_id)
    run = arrived[run_id]
    assert run.data.params == there.data.params
    assert run.data.tags == there.data.tags
    assert run.data.metrics == there.data.metrics == {"test/accuracy": i + 2.0}
    assert run.info.start_time == there.info.start_time
    assert run.info.end_time == there.info.end_time
    assert run.info.artifact_uri.startswith(f"{HERE}/")
    assert run.info.artifact_uri.endswith(there.info.artifact_uri.removeprefix(LAB))
    history = store.get_metric_history(run_id, "test/accuracy")
    assert [(m.step, m.value) for m in history] == [(s, i + s + 0.0) for s in range(3)]

  assert store.get_run(mine).data.metrics == {"test/accuracy": 7.0}  # untouched


def test_merging_again_adds_nothing(machines):
  lab, here, runs, *_ = machines
  merge(lab, NAME, here, HERE, pause=0)
  assert merge(lab, NAME, here, HERE, pause=0)["copied"] == 0
  store = client(here)
  experiment = store.get_experiment_by_name(NAME).experiment_id
  assert len(store.search_runs([experiment])) == len(runs)
  assert len(store.get_metric_history(runs[0], "test/accuracy")) == 3


def test_an_interrupted_merge_finishes_where_it_stopped(machines):
  lab, here, runs, *_ = machines
  merge(lab, NAME, here, HERE, pause=0)
  with sqlite3.connect(here) as db:  # as if it had stopped before the last run
    for table in ("params", "tags", "metrics", "latest_metrics", "runs"):
      db.execute(f"DELETE FROM {table} WHERE run_uuid = ?", (runs[-1],))
  assert merge(lab, NAME, here, HERE, pause=0)["copied"] == 1
  assert client(here).get_run(runs[-1]).data.metrics == {"test/accuracy": 4.0}


def test_the_experiment_joins_one_of_the_same_name(machines):
  lab, here, runs, *_ = machines
  store = client(here)
  experiment = store.create_experiment(NAME, artifact_location=HERE)
  already = finished_run(store, experiment, "already", 1.0)
  merge(lab, NAME, here, HERE, pause=0)
  arrived = {r.info.run_id for r in store.search_runs([experiment])}
  assert arrived == {already, *runs}


def test_it_refuses_a_different_schema_or_an_unknown_experiment(machines):
  lab, here, *_ = machines
  with pytest.raises(SystemExit):
    merge(lab, "no-such-experiment", here, HERE, pause=0)
  with sqlite3.connect(lab) as db:
    db.execute("UPDATE alembic_version SET version_num = 'older'")
  with pytest.raises(SystemExit):
    merge(lab, NAME, here, HERE, pause=0)
  assert client(here).get_experiment_by_name(NAME) is None

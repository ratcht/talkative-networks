"""
Copy one experiment's finished runs from another machine's MLflow database
into this one's.

  uv run python -m testbed.merge_runs lab_mlflow.db channel-ladder-phase4

Copying the database file over would lose this machine's runs. Each machine
numbers its experiments on its own, so the same number can mean different
experiments, while a run's id is unique across machines. So the experiment is
matched by name and made here if it is new, every run keeps its id, and each
run's artifact path is rewritten to this machine's. The testbed logs params,
tags and metrics and no artifacts, so nothing else needs to travel.

A study may still be logging here. Each run is copied in a transaction of its
own, with a pause after it, so the database is never locked for more than a
moment. A run already here is skipped, so an interrupted merge can be run again.
Only finished runs are copied: a run that failed or was cut off stays behind.
"""

import argparse
import sqlite3
import time
from pathlib import Path

from testbed.data import REPO_ROOT
from testbed.tracking import ARTIFACTS

DATABASE = REPO_ROOT / "mlflow.db"
OWNED_BY_A_RUN = ("params", "tags", "metrics", "latest_metrics")
WAIT_FOR_LOCK = 60.0  # seconds; longer than any write on this database takes
PAUSE = 0.2  # seconds between runs, for a study logging at the same time


def columns(db: sqlite3.Connection, table: str) -> str:
  """The table's columns, quoted, for a statement. Both databases have the
  same schema: merge checks that first."""
  names = [row[1] for row in db.execute(f'PRAGMA main.table_info("{table}")')]
  return ", ".join(f'"{name}"' for name in names)


def check_same_schema(db: sqlite3.Connection) -> None:
  versions = {
    name: db.execute(f"SELECT version_num FROM {name}.alembic_version").fetchone()[0]
    for name in ("main", "source")
  }
  if versions["main"] != versions["source"]:
    raise SystemExit(
      f"the databases have different schemas ({versions}): "
      "run the same MLflow version on both machines"
    )


def experiment_here(
  db: sqlite3.Connection, name: str, source_id: int, artifacts: str
) -> tuple[int, str]:
  """This database's id and artifact location for the experiment, making it,
  with its tags, if it is new."""
  query = "SELECT experiment_id, artifact_location FROM main.experiments WHERE name = ?"
  found = db.execute(query, (name,)).fetchone()
  if found:
    return found
  db.execute("BEGIN IMMEDIATE")
  db.execute(
    """INSERT INTO main.experiments
       (name, artifact_location, lifecycle_stage, creation_time, last_update_time)
       SELECT name, ?, lifecycle_stage, creation_time, last_update_time
       FROM source.experiments WHERE experiment_id = ?""",
    (artifacts, source_id),
  )
  made = db.execute(query, (name,)).fetchone()
  db.execute(
    """INSERT INTO main.experiment_tags ("key", value, experiment_id)
       SELECT "key", value, ? FROM source.experiment_tags WHERE experiment_id = ?""",
    (made[0], source_id),
  )
  db.execute("COMMIT")
  return made


def copy_run(
  db: sqlite3.Connection, run: str, experiment: int, location: str, source_location: str
) -> None:
  """One run and everything it owns, in one transaction."""
  ((uri,),) = db.execute(
    "SELECT artifact_uri FROM source.runs WHERE run_uuid = ?", (run,)
  )
  if not uri.startswith(source_location):
    raise SystemExit(f"run {run}: {uri} isn't under {source_location}")
  names = columns(db, "runs")
  db.execute("BEGIN IMMEDIATE")
  try:
    db.execute(f"INSERT INTO main.runs ({names}) SELECT {names} FROM source.runs WHERE run_uuid = ?", (run,))
    db.execute(
      "UPDATE main.runs SET experiment_id = ?, artifact_uri = ? WHERE run_uuid = ?",
      (experiment, location + uri[len(source_location) :], run),
    )
    for table in OWNED_BY_A_RUN:
      names = columns(db, table)
      db.execute(
        f"INSERT INTO main.{table} ({names}) "
        f"SELECT {names} FROM source.{table} WHERE run_uuid = ?",
        (run,),
      )
  except BaseException:
    db.execute("ROLLBACK")
    raise
  db.execute("COMMIT")


def merge(
  source: Path,
  experiment: str,
  target: Path = DATABASE,
  artifacts: str = ARTIFACTS.as_uri(),
  pause: float = PAUSE,
) -> dict[str, int]:
  """Copy the experiment's finished runs from source into target, checking
  every table's row count afterwards. Returns how many runs were found,
  copied and left behind."""
  db = sqlite3.connect(
    Path(target).resolve().as_uri(),
    uri=True,
    timeout=WAIT_FOR_LOCK,
    isolation_level=None,  # the transactions are spelled out above
  )
  try:
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("ATTACH DATABASE ? AS source", (f"{Path(source).resolve().as_uri()}?mode=ro",))
    check_same_schema(db)
    found = db.execute(
      "SELECT experiment_id, artifact_location FROM source.experiments WHERE name = ?",
      (experiment,),
    ).fetchone()
    if found is None:
      raise SystemExit(f"{source} has no experiment {experiment}")
    source_id, source_location = found
    here, location = experiment_here(db, experiment, source_id, artifacts)

    runs = [
      row[0]
      for row in db.execute(
        """SELECT run_uuid FROM source.runs WHERE experiment_id = ?
           AND status = 'FINISHED' AND lifecycle_stage = 'active' ORDER BY start_time""",
        (source_id,),
      )
    ]
    ((everything,),) = db.execute(
      "SELECT count(*) FROM source.runs WHERE experiment_id = ?", (source_id,)
    )
    present = {row[0] for row in db.execute("SELECT run_uuid FROM main.runs")}
    new = [run for run in runs if run not in present]
    for run in new:
      copy_run(db, run, here, location, source_location)
      time.sleep(pause)

    for table in ("runs", *OWNED_BY_A_RUN):
      counts = [
        db.execute(
          f"SELECT count(*) FROM {database}.{table} WHERE run_uuid IN "
          """(SELECT run_uuid FROM source.runs WHERE experiment_id = ?
              AND status = 'FINISHED' AND lifecycle_stage = 'active')""",
          (source_id,),
        ).fetchone()[0]
        for database in ("source", "main")
      ]
      if counts[0] != counts[1]:
        raise SystemExit(f"{table}: {counts[0]} rows there but {counts[1]} here")
    return {"finished": len(runs), "copied": len(new), "left": everything - len(runs)}
  finally:
    db.close()


def main(argv: list[str] | None = None) -> None:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("source", type=Path, help="the other machine's mlflow.db")
  p.add_argument("experiment", help="its name, the same on both machines")
  args = p.parse_args(argv)
  counts = merge(args.source, args.experiment)
  print(
    f"{args.experiment}: {counts['finished']} finished runs, "
    f"{counts['copied']} copied, {counts['left']} left behind unfinished"
  )


if __name__ == "__main__":
  main()

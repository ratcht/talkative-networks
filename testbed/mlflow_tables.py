"""
Reading runs straight from MLflow's own tables (experiments, runs, tags, params,
metrics), for a study's read script. One SQL query pulls every metric at every step
of a run; MLflow's web API would need a request per run and metric.

Two places hold those tables, and the same SQL reads both:
  - a local mlflow.db file, the default: the laptop's archive of the studies run
    before the team server existed;
  - the team server's Postgres (server=True), as the read-only login mlflow_reader,
    through an ssh tunnel to aisg-01.

Parameters in the SQL are named, `where name = :name`, since the two databases'
drivers spell unnamed ones differently.

Once per machine, for the server: put mlflow_reader's password (on aisg-01:
sudo cat /etc/mlflow/reader-password) in ~/.pgpass, chmod 600, as the one line

  localhost:15432:mlflow:mlflow_reader:PASSWORD

Each time, leave the tunnel running while reading (any ssh login to aisg-01 works):

  ssh -N -L 15432:127.0.0.1:5432 unipc_slow
"""

import sqlite3
from pathlib import Path

from testbed.data import REPO_ROOT

LOCAL_DB = REPO_ROOT / "mlflow.db"
TUNNEL_PORT = 15432
SERVER_DB = f"postgresql+psycopg://mlflow_reader@localhost:{TUNNEL_PORT}/mlflow"


class Tables:
  """Read-only access to one MLflow database's tables."""

  def __init__(self, server: bool = False, db: Path | str = LOCAL_DB):
    from sqlalchemy import create_engine
    from sqlalchemy.exc import OperationalError

    if server:
      self.where = f"the team server (tunnel on localhost:{TUNNEL_PORT})"
      self.engine = create_engine(SERVER_DB)
      try:
        self.engine.connect().close()
      except OperationalError as e:
        raise SystemExit(
          f"Can't reach the team server's database: {e.orig}\n"
          f"Is the tunnel open (ssh -N -L {TUNNEL_PORT}:127.0.0.1:5432 unipc_slow), "
          "and is mlflow_reader's password in ~/.pgpass? See testbed/mlflow_tables.py."
        ) from None
    else:
      db = Path(db)
      if not db.exists():
        raise FileNotFoundError(f"no MLflow database at {db}")
      self.where = str(db)
      # read-only, and willing to wait out a writer's lock
      self.engine = create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30),
      )

  def rows(self, sql: str, **params) -> list[tuple]:
    """Every row `sql` returns, as tuples; `params` fill its :name placeholders."""
    from sqlalchemy import text

    with self.engine.connect() as connection:
      return [tuple(row) for row in connection.execute(text(sql), params)]

"""
Logins on the team's MLflow server: add a teammate, or change a password.

  .venv/bin/python -m infra.mlflow_server.users add NAME        # needs the admin login
  .venv/bin/python -m infra.mlflow_server.users password NAME   # admin, or NAME themself

Run from the repo root in your own terminal: it prompts for every password, so none of
them lands in shell history or a file. Every login can read and log into every experiment
(default_permission = EDIT in the server's config); deleting needs the experiment's creator
or an admin.
"""

import argparse
import getpass
import os
import sys

from testbed.tracking import SERVER_URI, use_server_ca


def ask_new_password(username: str) -> str:
  while True:
    password = getpass.getpass(f"New password for {username} (12+ characters): ")
    if len(password) >= 12 and getpass.getpass("Again: ") == password:
      return password
    print("Too short, or they didn't match. Try again.")


def main() -> None:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("action", choices=["add", "password"])
  p.add_argument("username")
  args = p.parse_args()

  default = "admin" if args.action == "add" else args.username
  login = input(f"Log in as [{default}]: ").strip() or default
  # The environment wins over ~/.mlflow/credentials, and lives only in this process.
  os.environ["MLFLOW_TRACKING_USERNAME"] = login
  os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass(f"Password for {login}: ")
  password = ask_new_password(args.username)

  use_server_ca()
  from mlflow.server import get_app_client

  client = get_app_client("basic-auth", tracking_uri=SERVER_URI)
  if args.action == "add":
    client.create_user(args.username, password)
    print(f"Added {args.username}. They put this in ~/.mlflow/credentials (chmod 600):")
    print(f"  [mlflow]\n  mlflow_tracking_username = {args.username}")
    print("  mlflow_tracking_password = THEIR PASSWORD")
  else:
    client.update_user_password(args.username, password)
    print(f"Changed the password for {args.username}; it applies within 5 minutes.")


if __name__ == "__main__":
  sys.exit(main())

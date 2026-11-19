#!/usr/bin/env bash
# One-time setup of the team's MLflow server on aisg-01 (ssh alias unipc_slow).
#
# Copy this folder there, then run it yourself; it asks for the sudo password and then
# for an MLflow admin password, neither of which is stored anywhere but where it belongs:
#
#   scp -r infra/mlflow_server unipc_slow:mlflow-setup
#   ssh -t unipc_slow 'sudo bash ~/mlflow-setup/setup.sh'
#
# Safe to rerun: every step skips what already exists.
#
# What it sets up:
#   Postgres (apt)        databases `mlflow` (runs) and `mlflow_auth` (logins), owned by the
#                         role `mlflow`, reached over the local socket, so no database password
#                         exists for it; plus `mlflow_reader`, read-only on `mlflow`, for analysis
#                         through `ssh -L 5432:localhost:5432 unipc_slow`
#   /opt/mlflow           Python 3.12 and mlflow==$MLFLOW_VERSION (the version requirements.txt pins)
#   /etc/mlflow           logins config, secrets (root only), TLS certificates from our own small CA
#   mlflow.service        `mlflow server` as the system user `mlflow`, HTTPS on $PORT, logins
#                         required, started at boot and restarted if it dies
#   /etc/cron.d/...       a nightly pg_dump of both databases, kept 14 days
#
# The CA certificate is copied next to this script as ca.crt: it is what clients trust.

set -euo pipefail
trap 'echo "setup.sh stopped at line $LINENO; nothing after it ran. Rerunning is safe." >&2' ERR

HOST=aisg-01.uwaterloo.ca
IP=129.97.72.102
PORT=5050
MLFLOW_VERSION=3.17.2
PYTHON=3.12

ETC=/etc/mlflow
TLS=$ETC/tls
STATE=/var/lib/mlflow
OPT=/opt/mlflow
HERE=$(cd "$(dirname "$0")" && pwd)

[[ $EUID -eq 0 ]] || { echo "Run with sudo: sudo bash $0" >&2; exit 1; }

echo "== 1/8 Postgres"
if ! dpkg -s postgresql &>/dev/null; then
  apt-get update -q
  apt-get install -y -q postgresql
fi

echo "== 2/8 the mlflow system user and its folders"
id mlflow &>/dev/null ||
  useradd --system --home-dir $STATE --create-home --shell /usr/sbin/nologin mlflow
install -d -o mlflow -g mlflow -m 750 $STATE $STATE/artifacts
install -d -o root -g mlflow -m 750 $ETC
install -d -o root -g root -m 755 $OPT

echo "== 3/8 Python $PYTHON and mlflow $MLFLOW_VERSION in $OPT"
# The interpreter goes under /opt, not root's home, so the mlflow user can read it.
export UV_PYTHON_INSTALL_DIR=$OPT/python
[[ -x $OPT/venv/bin/python ]] ||
  /snap/bin/uv venv --managed-python --python $PYTHON $OPT/venv
# compiled now, since the mlflow user can't write __pycache__ under /opt
/snap/bin/uv pip install -q --compile-bytecode --python $OPT/venv/bin/python \
  "mlflow[db,auth]==$MLFLOW_VERSION"

echo "== 4/8 databases"
sudo -u postgres psql -q -v ON_ERROR_STOP=1 <<'SQL'
SELECT 'CREATE ROLE mlflow LOGIN'
  WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mlflow')\gexec
SELECT 'CREATE ROLE mlflow_reader LOGIN'
  WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mlflow_reader')\gexec
SELECT 'CREATE DATABASE mlflow OWNER mlflow'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
SELECT 'CREATE DATABASE mlflow_auth OWNER mlflow'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_auth')\gexec
REVOKE CONNECT ON DATABASE mlflow_auth FROM PUBLIC;
GRANT CONNECT ON DATABASE mlflow TO mlflow_reader;
SQL
# Tables appear when the server first starts; the default privilege covers them and any
# a later MLflow migration adds.
sudo -u postgres psql -q -v ON_ERROR_STOP=1 -d mlflow <<'SQL'
GRANT USAGE ON SCHEMA public TO mlflow_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mlflow_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE mlflow IN SCHEMA public GRANT SELECT ON TABLES TO mlflow_reader;
SQL
if [[ ! -s $ETC/reader-password ]]; then
  (umask 077; openssl rand -base64 30 | tr -d '/+=\n' > $ETC/reader-password)
  # through stdin, so the password never shows up in a process listing
  sudo -u postgres psql -q -v ON_ERROR_STOP=1 <<SQL
ALTER ROLE mlflow_reader PASSWORD '$(cat $ETC/reader-password)';
SQL
fi
chown root:root $ETC/reader-password
chmod 600 $ETC/reader-password

echo "== 5/8 TLS certificate for $HOST"
install -d -o root -g mlflow -m 750 $TLS
if [[ ! -s $TLS/server.crt ]]; then
  openssl req -x509 -new -nodes -newkey rsa:3072 -sha256 -days 3650 \
    -subj "/CN=aisg-01 MLflow CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -keyout $TLS/ca.key -out $TLS/ca.crt 2>/dev/null
  openssl req -new -nodes -newkey rsa:3072 -subj "/CN=$HOST" \
    -keyout $TLS/server.key -out $TLS/server.csr 2>/dev/null
  # 825 days: the longest some browsers accept. Renew by deleting server.crt and rerunning.
  openssl x509 -req -sha256 -days 825 -in $TLS/server.csr \
    -CA $TLS/ca.crt -CAkey $TLS/ca.key -CAcreateserial -out $TLS/server.crt \
    -extfile <(printf '%s\n' \
      "subjectAltName=DNS:$HOST,IP:$IP" \
      "basicConstraints=critical,CA:FALSE" \
      "keyUsage=critical,digitalSignature,keyEncipherment" \
      "extendedKeyUsage=serverAuth" \
      "subjectKeyIdentifier=hash" \
      "authorityKeyIdentifier=keyid,issuer") 2>/dev/null
  rm -f $TLS/server.csr
fi
chown root:root $TLS/ca.key && chmod 600 $TLS/ca.key
chown root:mlflow $TLS/server.key && chmod 640 $TLS/server.key
chmod 644 $TLS/ca.crt $TLS/server.crt
install -m 644 -o "${SUDO_USER:-root}" $TLS/ca.crt "$HERE/ca.crt"

echo "== 6/8 configuration"
cat > $ETC/basic_auth.ini <<'INI'
[mlflow]
# every login can read and log into every experiment; deleting one needs MANAGE,
# which only its creator (and admins) get
default_permission = EDIT
database_uri = postgresql://mlflow@/mlflow_auth?host=/var/run/postgresql
admin_username = admin
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
grant_default_workspace_access = false
# checking a password is deliberately slow; remember a good one for 5 minutes per worker,
# so a changed password takes up to that long to apply
auth_cache_ttl_seconds = 300
INI
chown root:mlflow $ETC/basic_auth.ini && chmod 640 $ETC/basic_auth.ini
if [[ ! -s $ETC/server.env ]]; then
  (umask 077; printf 'MLFLOW_FLASK_SERVER_SECRET_KEY=%s\n' "$(openssl rand -hex 32)" > $ETC/server.env)
fi
chown root:root $ETC/server.env && chmod 600 $ETC/server.env   # systemd reads it as root

echo "== 7/8 the admin login"
cd $STATE   # the mlflow user can't enter the home folder this was started from
has_admin=$(sudo -u postgres psql -d mlflow_auth -tAc \
  "SELECT 1 FROM users WHERE username = 'admin'" 2>/dev/null || true)
if [[ $has_admin != 1 ]]; then
  while true; do
    read -rsp "Choose a password for the MLflow login 'admin' (12+ characters): " pw; echo
    read -rsp "Again: " pw2; echo
    [[ $pw == "$pw2" ]] && (( ${#pw} >= 12 )) && break
    echo "They didn't match, or it's shorter than 12 characters. Try again."
  done
  # through stdin, so it never shows up in a process listing or a file
  printf '%s' "$pw" | sudo -u mlflow env MLFLOW_AUTH_CONFIG_PATH=$ETC/basic_auth.ini \
    $OPT/venv/bin/python -c '
import os, sys
os.environ["MLFLOW_AUTH_ADMIN_PASSWORD"] = sys.stdin.read()
from mlflow.server.auth import bootstrap_admin_user
bootstrap_admin_user()
print("created the MLflow login admin")'
  unset pw pw2
fi

echo "== 8/8 the service, firewall and backups"
cat > /etc/systemd/system/mlflow.service <<UNIT
[Unit]
Description=MLflow tracking server (the team's runs)
After=network-online.target postgresql.service
Wants=network-online.target postgresql.service

[Service]
User=mlflow
Group=mlflow
WorkingDirectory=$STATE
EnvironmentFile=$ETC/server.env
Environment=MLFLOW_AUTH_CONFIG_PATH=$ETC/basic_auth.ini
ExecStart=$OPT/venv/bin/mlflow server \\
  --backend-store-uri postgresql://mlflow@/mlflow?host=/var/run/postgresql \\
  --artifacts-destination $STATE/artifacts \\
  --host 0.0.0.0 --port $PORT --workers 4 \\
  --app-name basic-auth \\
  --allowed-hosts $HOST:$PORT,$IP:$PORT \\
  --uvicorn-opts "--ssl-keyfile $TLS/server.key --ssl-certfile $TLS/server.crt"
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable mlflow
systemctl restart mlflow

if ufw status 2>/dev/null | grep -q "Status: active"; then
  ufw allow $PORT/tcp comment "MLflow"
fi

install -d -o postgres -g postgres -m 700 /var/backups/mlflow
cat > /etc/cron.d/mlflow-backup <<'CRON'
# nightly dumps of the team's MLflow databases, kept 14 days
30 3 * * * postgres for db in mlflow mlflow_auth; do pg_dump -Fc "$db" > /var/backups/mlflow/"$db"-$(date +\%F).dump; done; find /var/backups/mlflow -name '*.dump' -mtime +14 -delete
CRON
chmod 644 /etc/cron.d/mlflow-backup

healthy() {
  $OPT/venv/bin/python - "https://$HOST:$PORT/health" "$TLS/ca.crt" <<'PY'
import ssl, sys, urllib.request
url, ca = sys.argv[1:]
urllib.request.urlopen(url, context=ssl.create_default_context(cafile=ca), timeout=5).read()
PY
}
# The first start builds every table in the empty database, so give it a few minutes.
echo -n "waiting for the server"
for _ in $(seq 90); do healthy 2>/dev/null && break; echo -n .; sleep 2; done; echo
if healthy; then
  echo
  echo "Up: https://$HOST:$PORT"
  echo "CA certificate for clients: $HERE/ca.crt"
  echo "Read-only Postgres login: mlflow_reader, password in $ETC/reader-password (sudo cat)"
else
  echo "The server didn't answer. Its log: journalctl -u mlflow -n 80 --no-pager" >&2
  exit 1
fi

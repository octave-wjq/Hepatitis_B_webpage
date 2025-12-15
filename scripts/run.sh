#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AUTH_USER="${AUTH_USER:-admin}"
AUTH_PASSWORD="${AUTH_PASSWORD:-changeit}"
APP_DB_PATH="${APP_DB_PATH:-/data/app.db}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
SERVER_PID=""

export AUTH_USER AUTH_PASSWORD APP_DB_PATH
mkdir -p "$(dirname "${APP_DB_PATH}")"

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${ROOT_DIR}"

echo "Starting Streamlit on port ${STREAMLIT_PORT} (db at ${APP_DB_PATH})"
streamlit run app/main.py --server.address 0.0.0.0 --server.port "${STREAMLIT_PORT}" --server.headless true &
SERVER_PID=$!

for _ in {1..30}; do
  if curl -fs "http://localhost:${STREAMLIT_PORT}" >/dev/null 2>&1; then
    echo "Streamlit is healthy (pid ${SERVER_PID})"
    wait "${SERVER_PID}"
    exit $?
  fi

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Streamlit process exited before becoming healthy"
    exit 1
  fi

  sleep 1
done

echo "Streamlit did not become healthy in time; terminating"
exit 1

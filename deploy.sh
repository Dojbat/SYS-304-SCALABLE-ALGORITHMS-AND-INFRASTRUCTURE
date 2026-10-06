#!/usr/bin/env bash
set -euo pipefail

if ! docker info >/dev/null 2>&1; then
  echo "Docker does not appear to be running. Start Docker and try again." >&2
  exit 1
fi

echo "Building and starting the stack (waits until every service is healthy)..."
if ! docker compose up --build -d --wait --wait-timeout 600; then
  echo "Stack did not become healthy. Check: docker compose logs backend" >&2
  exit 1
fi

echo ""
echo "App is running:"
echo "  Frontend:  http://localhost:3000"
echo "  API:       http://localhost:8000/health  (load balancer over $(docker compose ps -q backend | wc -l | tr -d ' ') backend replicas)"
echo "  Dashboard: http://localhost:8501"
echo ""
echo "Tail logs with: docker compose logs -f backend"

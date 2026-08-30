#!/usr/bin/env bash
set -euo pipefail

if ! docker info >/dev/null 2>&1; then
  echo "Docker does not appear to be running. Start Docker and try again." >&2
  exit 1
fi

echo "Building and starting the stack..."
docker compose up --build -d

echo "Waiting for backend health check..."
for _ in $(seq 1 30); do
  status=$(docker compose ps backend --format json 2>/dev/null | grep -o '"Health":"[a-z]*"' | cut -d'"' -f4 || true)
  if [ "$status" = "healthy" ]; then
    echo "Backend is healthy."
    echo ""
    echo "App is running:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend:  http://localhost:8000/health"
    echo ""
    echo "Tail logs with: docker compose logs -f backend"
    exit 0
  fi
  sleep 2
done

echo "Timed out waiting for backend to become healthy. Check: docker compose logs backend" >&2
exit 1

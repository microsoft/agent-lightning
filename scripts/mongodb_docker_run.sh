#!/bin/bash

set -euo pipefail

cd docker

# Setup data directories
./setup.sh

# Start Dockers
docker compose -f compose.mongo.yml up -d

SERVICE_NAME=mongo
TIMEOUT=60   # seconds
SLEEP=2

cid="$(docker compose -f compose.mongo.yml ps -q "$SERVICE_NAME")"
if [ -z "$cid" ]; then
    echo "Service $SERVICE_NAME is not running"
    exit 1
fi

echo "Waiting for $SERVICE_NAME to become healthy..."
end=$((SECONDS + TIMEOUT))

while [ "$SECONDS" -lt "$end" ]; do
    status="$(docker inspect -f '{{.State.Health.Status}}' "$cid")"
echo "Current status: $status"

if [ "$status" = "healthy" ]; then
    echo "$SERVICE_NAME is healthy ✅"
    exit 0
elif [ "$status" = "unhealthy" ]; then
    echo "$SERVICE_NAME is unhealthy ❌"
    docker logs "$cid" || true
    exit 1
fi

sleep "$SLEEP"
done

echo "Timed out waiting for $SERVICE_NAME to become healthy after ${TIMEOUT}s"
docker logs "$cid" || true
exit 1

#!/usr/bin/env bash
# Load the committed Neo4j graph export into the running neo4j container.
#
# Usage:
#   docker compose up -d          # start the stack first
#   scripts/restore_neo4j.sh      # then load the graph
#
# Honors NEO4J_CONTAINER / NEO4J_USER / NEO4J_PASSWORD env vars.
set -euo pipefail

CONTAINER="${NEO4J_CONTAINER:-orggraph-neo4j}"
USER_="${NEO4J_USER:-neo4j}"
PASS="${NEO4J_PASSWORD:-orggraph2026}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EXPORT="$HERE/../data/neo4j_export.cypher.gz"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "Container '${CONTAINER}' is not running. Start the stack with: docker compose up -d" >&2
  exit 1
fi

echo "Loading $(basename "$EXPORT") into '${CONTAINER}' ..."
gzip -dc "$EXPORT" | docker exec -i "$CONTAINER" cypher-shell -u "$USER_" -p "$PASS"
echo "Done. Node count:"
docker exec "$CONTAINER" cypher-shell -u "$USER_" -p "$PASS" --format plain "MATCH (n) RETURN count(n) AS nodes;"

#!/usr/bin/env bash
# Re-index Data/all_chunks.jsonl into Qdrant Cloud.
#
# Usage:
#   export QDRANT_URL=https://xxx-xxx-xxx.eu-central-0-0.aws.cloud.qdrant.io
#   export QDRANT_API_KEY=eyJh...
#   bash deploy/scripts/upload_qdrant_to_cloud.sh

set -euo pipefail

if [ -z "${QDRANT_URL:-}" ] || [ -z "${QDRANT_API_KEY:-}" ]; then
  echo "ERROR: set QDRANT_URL and QDRANT_API_KEY first."
  echo "  export QDRANT_URL=https://xxx.qdrant.cloud"
  echo "  export QDRANT_API_KEY=eyJh..."
  exit 1
fi

# Run from traffic_rag/
cd "$(dirname "$0")/../.."

echo "==> Indexing Data/all_chunks.jsonl → $QDRANT_URL"
python -m source.indexing.indexer --recreate

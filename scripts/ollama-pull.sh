#!/bin/sh
# One-shot: pull every model listed in config/models.yaml (key `ollama_tag:`), then exit.
# Entries with `enabled: false` on the line after are not special-cased; remove or comment them.
set -eu

MODELS_FILE="${MODELS_FILE:-/app/config/models.yaml}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"

if [ ! -f "$MODELS_FILE" ]; then
    echo "ollama-pull: $MODELS_FILE not found" >&2
    exit 1
fi

tags=$(grep -E '^[[:space:]]*(-[[:space:]]*)?ollama_tag:' "$MODELS_FILE" \
    | sed -E 's/^[^:]*:[[:space:]]*//; s/[[:space:]]*#.*$//; s/["'"'"']//g' | grep -v '^$' || true)
if [ -z "$tags" ]; then
    echo "ollama-pull: no ollama_tag entries in $MODELS_FILE" >&2
    exit 1
fi

for tag in $tags; do
    echo "ollama-pull: pulling $tag"
    ollama pull "$tag"
done
echo "ollama-pull: done"

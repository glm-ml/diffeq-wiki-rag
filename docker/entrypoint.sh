#!/bin/sh
set -e

if [ ! -f "${CORPUS_PATH:-data/corpus.jsonl}" ]; then
    echo "Корпус не найден — собираю из Википедии"
    python -m app.ingest.build_corpus
fi

python -m app.ingest.ensure_index

exec "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NO_BUILD=0
if [[ "${1:-}" == "--no-build" ]]; then
  NO_BUILD=1
fi

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

require_command() {
  local name="$1"
  local hint="$2"
  command -v "$name" >/dev/null 2>&1 || fail "$name is not installed or not in PATH. $hint"
}

dotenv_value() {
  local key="$1"
  local file="$2"
  awk -F= -v key="$key" '
    $0 !~ /^[[:space:]]*#/ && $1 == key {
      sub(/^[^=]*=/, "")
      gsub(/^[[:space:]]+|[[:space:]]+$/, "")
      gsub(/^["'"'"']|["'"'"']$/, "")
      print
      exit
    }
  ' "$file"
}

require_command docker "Install Docker first."
require_command ollama "Install Ollama first: https://ollama.com/download"
require_command curl "Install curl first."

docker info >/dev/null 2>&1 || fail "Docker is not running."
docker compose version >/dev/null 2>&1 || fail "Docker Compose is not available."

curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null \
  || fail "Ollama is not running at http://127.0.0.1:11434. Start Ollama and retry."

if [[ ! -f ".env" ]]; then
  [[ -f ".env.example" ]] || fail ".env.example is missing."
  cp ".env.example" ".env"
  fail ".env was created from .env.example. Fill RAG_ANSWER_API_KEY, then rerun ./run.sh."
fi

api_key="$(dotenv_value RAG_ANSWER_API_KEY .env || true)"
if [[ -z "$api_key" || "$api_key" == "your_answer_api_key_here" ]]; then
  fail "RAG_ANSWER_API_KEY is missing in .env. Fill it with your answer model API key, then rerun ./run.sh."
fi

rewrite_model="$(dotenv_value RAG_QUERY_REWRITE_MODEL .env || true)"
embedding_model="$(dotenv_value RAG_EMBEDDING_MODEL .env || true)"
rewrite_model="${rewrite_model:-qwen3.5:4b}"
embedding_model="${embedding_model:-bge-m3:latest}"

installed_models="$(ollama list | awk 'NR > 1 {print $1}')"
for model in "$rewrite_model" "$embedding_model"; do
  if ! printf '%s\n' "$installed_models" | grep -Fxq "$model"; then
    printf 'Pulling Ollama model: %s\n' "$model"
    ollama pull "$model" || fail "Failed to pull Ollama model: $model"
  else
    printf 'Ollama model ready: %s\n' "$model"
  fi
done

printf '\nStarting services...\n'
printf 'App:     http://localhost:8000\n'
printf 'Swagger: http://localhost:8000/swagger\n\n'

if [[ "$NO_BUILD" == "1" ]]; then
  docker compose up
else
  docker compose up --build
fi

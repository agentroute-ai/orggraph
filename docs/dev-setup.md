# Dev Setup

Three environments for running the OrgGraph stack. Code stays the same - only the `.env` file changes.

---

## Mode A - Mac development (Apple Silicon)

**Goal:** iterate on pipeline code fast, using small dev-stand-in models.

### 1. Databases via Docker

```bash
cp .env.example .env
cp litellm-config.example.yaml litellm-config.yaml   # edit endpoint URLs as needed
docker compose up -d            # starts postgres + neo4j + vllm + litellm
docker compose ps               # verify healthy
```

Connect:
- Postgres: `postgresql://orggraph:orggraph2026@localhost:5432/orggraph`
- Neo4j UI: http://localhost:7474 (neo4j / orggraph2026)

### 2. LLM/Embeddings via Ollama (native, Metal-accelerated)

vLLM has no stable CUDA path on Mac, and Docker has no GPU passthrough on Apple Silicon. Run Ollama natively so it uses Metal.

```bash
brew install ollama
ollama serve &                  # runs on localhost:11434

# Pull dev-sized models
ollama pull embeddinggemma:300m # ~300MB, 768-dim output
ollama pull gemma3:4b           # ~3GB, dev stand-in for Gemma 4 31B
```

The default `.env.example` is pre-configured for this mode. Code reads `OPENAI_BASE_URL` + `OPENAI_API_KEY` - Ollama exposes an OpenAI-compatible API at `/v1`.

### 3. Smoke test

```bash
# Embedding endpoint
curl http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "embeddinggemma:300m", "input": "test"}' | jq '.data[0].embedding | length'
# expect: 768

# Chat endpoint
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3:4b", "messages": [{"role":"user","content":"say hi"}]}'
```

---

## Mode B - DGX Spark (production runs)

**Goal:** full-scale extraction (517K emails, 148 personas, ~700 LLM calls for Stage 3 deferred, all RQ2/RQ3 experiments).

### 1. Get an HF token

Gemma 4 is a gated model. Accept the license at https://huggingface.co/google/gemma-4-31B-it, create a read token in your HF settings, paste into `.env` as `HF_TOKEN`.

### 2. Bring up everything

```bash
docker compose --profile dgx up -d
docker compose logs -f vllm-inference    # first boot downloads ~62GB, 5-10 min
```

Switch `.env` to **Mode B** (comment out Mode A, uncomment Mode B). Or maintain two env files (`.env.mac`, `.env.dgx`) and symlink.

### 3. Memory budget on one DGX Spark (128GB unified)

| Service | Memory |
|---|---|
| vLLM inference (Gemma 4 31B BF16) | ~61GB weights + ~10GB KV cache = ~71GB |
| vLLM embed (EmbeddingGemma 300M) | ~1GB |
| Postgres + pgvector | ~2GB |
| Neo4j (heap 2GB + pagecache 1GB) | ~3GB |
| OS + headroom | ~50GB |

`--gpu-memory-utilization 0.55` on the inference service caps vLLM at ~70GB, leaving room for everything else.

---

## Mode C - CSCS Alps (batch evaluation)

No outbound internet. Pre-stage models in `$STORE`, launch vLLM via SLURM, same OpenAI-compatible API. Out of scope for local development.

---

## Embedding dimensionality

EmbeddingGemma 300M outputs **768-dim** vectors by default. The pgvector schema in [docker/postgres/init.sql](../docker/postgres/init.sql) creates `vector(768)` columns + HNSW cosine indexes.

Matryoshka truncation is available: slice the first 512, 256, or 128 floats of the output and it remains a valid embedding. Useful for an RQ3 ablation: **retrieval quality vs embedding dimension**. To use a smaller dimension, change the column type in `init.sql` and regenerate indexes before embedding.

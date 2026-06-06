-- Initialize pgvector extension for the orggraph database.
-- Runs once on first container boot (when the data volume is empty).
--
-- This file is BOTH the bootstrap script for fresh containers AND the
-- running ledger of schema evolution. Every change is appended below as
-- ALTER TABLE / CREATE TABLE IF NOT EXISTS so the file stays idempotent
-- and re-applicable to a live DB. The source of truth for the current
-- schema is "init.sql + whatever has been applied to the live DB".
-- If you wipe the data volume, re-running this file recreates the full
-- schema. No separate migration tool is in use yet.

CREATE EXTENSION IF NOT EXISTS vector;

-- Minimal schema scaffolding for RQ3 (can be replaced by a migration tool later).
-- Kept intentionally small — the real schema is built by the embedding pipeline.

CREATE TABLE IF NOT EXISTS embeddings_person (
    person_id      TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    role           TEXT,
    department     TEXT,
    embedding      vector(768),
    metadata       JSONB,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embeddings_entity (
    entity_id      TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    category       TEXT,
    embedding      vector(768),
    metadata       JSONB,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embeddings_email_summary (
    summary_id     TEXT PRIMARY KEY,
    person_id      TEXT REFERENCES embeddings_person(person_id) ON DELETE CASCADE,
    content        TEXT NOT NULL,
    embedding      vector(768),
    metadata       JSONB,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW indexes for cosine similarity (EmbeddingGemma outputs 768-dim vectors)
CREATE INDEX IF NOT EXISTS idx_person_embedding
    ON embeddings_person USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_entity_embedding
    ON embeddings_entity USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_email_summary_embedding
    ON embeddings_email_summary USING hnsw (embedding vector_cosine_ops);

-- Per-email embeddings + metadata + extracted signals (per-email pipeline).
-- One row per filtered Enron email after filter_corpus.py + embed_emails.py.
CREATE TABLE IF NOT EXISTS embeddings_email (
    email_id            TEXT PRIMARY KEY,
    thread_id           TEXT,
    sender_email        TEXT,
    sender_resolved     TEXT,
    recipients_emails   JSONB,
    recipients_resolved JSONB,
    date                TIMESTAMPTZ,
    subject             TEXT,
    body_chars          INTEGER,
    body_truncated      TEXT,
    topics              JSONB,
    intent              TEXT,
    sentiment           REAL,
    decision_carrying   BOOLEAN,
    mentions_money      BOOLEAN,
    mentions_regulator  BOOLEAN,
    entities_mentioned  JSONB,
    embedding           VECTOR(768) NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    signals_extracted_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_email_embedding
    ON embeddings_email USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_email_date
    ON embeddings_email (date);
CREATE INDEX IF NOT EXISTS idx_email_sender
    ON embeddings_email (sender_resolved);
CREATE INDEX IF NOT EXISTS idx_email_decision
    ON embeddings_email (decision_carrying) WHERE decision_carrying = true;

-- Stage 1.5 deterministic per-email metadata
ALTER TABLE embeddings_email
    ADD COLUMN IF NOT EXISTS body_word_count       INTEGER,
    ADD COLUMN IF NOT EXISTS n_questions           INTEGER,
    ADD COLUMN IF NOT EXISTS n_exclamations        INTEGER,
    ADD COLUMN IF NOT EXISTS n_imperatives         INTEGER,
    ADD COLUMN IF NOT EXISTS n_modals              JSONB,
    ADD COLUMN IF NOT EXISTS n_first_person        INTEGER,
    ADD COLUMN IF NOT EXISTS n_second_person       INTEGER,
    ADD COLUMN IF NOT EXISTS to_count              INTEGER,
    ADD COLUMN IF NOT EXISTS cc_count              INTEGER,
    ADD COLUMN IF NOT EXISTS bcc_count             INTEGER,
    ADD COLUMN IF NOT EXISTS unique_recipients     INTEGER,
    ADD COLUMN IF NOT EXISTS is_thread_initiator   BOOLEAN,
    ADD COLUMN IF NOT EXISTS is_thread_closer      BOOLEAN,
    ADD COLUMN IF NOT EXISTS thread_position       INTEGER,
    ADD COLUMN IF NOT EXISTS reply_latency_hours   REAL,
    ADD COLUMN IF NOT EXISTS is_off_hours          BOOLEAN,
    ADD COLUMN IF NOT EXISTS is_weekend            BOOLEAN,
    ADD COLUMN IF NOT EXISTS politeness_score      REAL,
    ADD COLUMN IF NOT EXISTS hedge_count           INTEGER,
    ADD COLUMN IF NOT EXISTS metadata_extracted_at TIMESTAMPTZ;

-- Stage 2a schema swap: drop single-valued intent, add speech_acts + boolean signals
ALTER TABLE embeddings_email
    ADD COLUMN IF NOT EXISTS speech_acts      JSONB,
    ADD COLUMN IF NOT EXISTS action_required  BOOLEAN,
    ADD COLUMN IF NOT EXISTS commitment_made  BOOLEAN;
-- intent column kept for backward compatibility; will be derived from speech_acts in Stage 3

CREATE INDEX IF NOT EXISTS idx_email_thread_pos
    ON embeddings_email (thread_id, thread_position);
CREATE INDEX IF NOT EXISTS idx_email_action_required
    ON embeddings_email (action_required) WHERE action_required = TRUE;

-- §3.B Dyadic pair signals — written by Stage 3
CREATE TABLE IF NOT EXISTS pair_signals (
    sender_id              TEXT NOT NULL,
    recipient_id           TEXT NOT NULL,
    n_emails               INTEGER NOT NULL DEFAULT 0,
    n_to                   INTEGER NOT NULL DEFAULT 0,
    n_cc                   INTEGER NOT NULL DEFAULT 0,
    n_request_sent         INTEGER NOT NULL DEFAULT 0,
    n_commit_sent          INTEGER NOT NULL DEFAULT 0,
    n_deliver_sent         INTEGER NOT NULL DEFAULT 0,
    n_propose_sent         INTEGER NOT NULL DEFAULT 0,
    n_decision             INTEGER NOT NULL DEFAULT 0,
    n_action_required      INTEGER NOT NULL DEFAULT 0,
    request_commit_ratio   REAL,
    mean_reply_latency_h   REAL,
    mean_sentiment         REAL,
    mean_body_words        REAL,
    length_asymmetry       REAL,
    first_email_date       TIMESTAMPTZ,
    last_email_date        TIMESTAMPTZ,
    aggregated_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (sender_id, recipient_id)
);
CREATE INDEX IF NOT EXISTS idx_pair_signals_sender
    ON pair_signals (sender_id);
CREATE INDEX IF NOT EXISTS idx_pair_signals_recipient
    ON pair_signals (recipient_id);

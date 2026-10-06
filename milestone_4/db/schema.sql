-- Milestone 4 schema. Mounted into the postgres container's
-- /docker-entrypoint-initdb.d/, so it runs once on a fresh volume.

-- One row per HTTP request to /predict, including the ones that failed
-- (422 validation errors, 500s), so error rate can be computed from it.
CREATE TABLE IF NOT EXISTS requests (
    id             UUID PRIMARY KEY,
    ts             TIMESTAMPTZ NOT NULL,
    endpoint       TEXT NOT NULL,
    status_code    SMALLINT NOT NULL,
    latency_ms     REAL NOT NULL,
    text           TEXT,
    prediction     SMALLINT,
    confidence     REAL,
    cached         BOOLEAN,
    model_version  TEXT,
    error          TEXT
);

CREATE INDEX IF NOT EXISTS requests_ts_idx ON requests (ts);
CREATE INDEX IF NOT EXISTS requests_model_version_ts_idx ON requests (model_version, ts);

-- Ground-truth labels that arrive after the fact (simulated by the workload
-- generator). No foreign key to requests: request rows are written in
-- batches, so a label can arrive before its request row has been flushed.
CREATE TABLE IF NOT EXISTS feedback (
    request_id   UUID PRIMARY KEY,
    true_label   SMALLINT NOT NULL CHECK (true_label IN (0, 1)),
    received_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_received_at_idx ON feedback (received_at);

-- Every labelled example. This is what retraining reads.
CREATE OR REPLACE VIEW labeled_examples AS
SELECT r.id AS request_id, r.ts, r.text, f.true_label, r.prediction, r.confidence,
       r.model_version, f.received_at
FROM requests r
JOIN feedback f ON f.request_id = r.id
WHERE r.text IS NOT NULL;

-- Retention: deletes requests older than keep_days that never got a label.
-- Labelled rows are the scarce, valuable data, so they are kept for
-- retraining. Returns the number of rows deleted (or, with dry_run, the
-- number that would be).
CREATE OR REPLACE FUNCTION apply_retention(keep_days int, dry_run boolean DEFAULT false)
RETURNS bigint
LANGUAGE plpgsql AS $$
DECLARE
    cutoff timestamptz := now() - make_interval(days => keep_days);
    n bigint;
BEGIN
    IF dry_run THEN
        SELECT count(*) INTO n FROM requests r
        WHERE r.ts < cutoff
          AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.request_id = r.id);
    ELSE
        DELETE FROM requests r
        WHERE r.ts < cutoff
          AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.request_id = r.id);
        GET DIAGNOSTICS n = ROW_COUNT;
    END IF;
    RETURN n;
END $$;

-- One row per retraining cycle, whether it deployed, was rejected, or failed.
CREATE TABLE IF NOT EXISTS retraining_runs (
    id                  SERIAL PRIMARY KEY,
    started_at          TIMESTAMPTZ NOT NULL,
    finished_at         TIMESTAMPTZ,
    trigger             TEXT NOT NULL,
    status              TEXT NOT NULL,
    base_version        TEXT,
    candidate_version   TEXT,
    data_cutoff         TIMESTAMPTZ,
    n_new_labeled       INTEGER,
    n_train             INTEGER,
    n_eval              INTEGER,
    base_f1             REAL,
    candidate_f1        REAL,
    base_guard_acc      REAL,
    candidate_guard_acc REAL,
    candidate_int8_f1   REAL,
    message             TEXT
);

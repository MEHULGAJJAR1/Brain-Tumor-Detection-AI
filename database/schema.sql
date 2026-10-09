PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL UNIQUE,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    analyzed_at TEXT,
    status TEXT NOT NULL DEFAULT 'awaiting_model'
        CHECK (status IN ('awaiting_model', 'completed', 'inference_failed')),
    prediction_class TEXT,
    confidence REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    probabilities_json TEXT,
    CHECK (
        (status = 'completed' AND prediction_class IS NOT NULL AND confidence IS NOT NULL)
        OR status <> 'completed'
    )
);

CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scans_status ON scans(status);

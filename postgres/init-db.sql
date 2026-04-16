CREATE DATABASE mlflow;
CREATE DATABASE airflow;
CREATE DATABASE mlops;

-- Connect to mlops and create the feedback table with S3 image references
\c mlops;

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    predicted_label INTEGER NOT NULL,
    correct_label INTEGER NOT NULL,
    is_correct BOOLEAN NOT NULL,
    image_key TEXT,
    used_for_training BOOLEAN NOT NULL DEFAULT FALSE
);

-- Fast lookup of feedback not yet used by training
CREATE INDEX idx_feedback_unused ON feedback (used_for_training) WHERE NOT used_for_training;

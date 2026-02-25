import os
import psycopg

DDL = r"""
-- =========================
-- 0) ENUMS (optional)
-- =========================
DO $$ BEGIN
  CREATE TYPE task_status AS ENUM ('planned','in_progress','done','partial','skipped','cancelled');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE rec_action AS ENUM ('accepted','rejected','ignored');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =========================
-- 1) USERS
-- =========================
CREATE TABLE IF NOT EXISTS users (
  user_id            UUID PRIMARY KEY,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  timezone           TEXT NOT NULL DEFAULT 'America/New_York',
  role               TEXT,
  grade_level        TEXT
);

-- =========================
-- 2) TASKS
-- =========================
CREATE TABLE IF NOT EXISTS tasks (
  task_id              UUID PRIMARY KEY,
  user_id              UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  title                TEXT NOT NULL,
  description          TEXT,

  category             TEXT,
  priority             SMALLINT CHECK (priority BETWEEN 1 AND 5),

  due_at               TIMESTAMPTZ,
  planned_start        TIMESTAMPTZ,
  planned_end          TIMESTAMPTZ,
  planned_duration_min INTEGER CHECK (planned_duration_min IS NULL OR planned_duration_min > 0),

  difficulty_self      SMALLINT CHECK (difficulty_self BETWEEN 1 AND 5),
  status               task_status NOT NULL DEFAULT 'planned'
);

CREATE INDEX IF NOT EXISTS idx_tasks_user_due ON tasks(user_id, due_at);
CREATE INDEX IF NOT EXISTS idx_tasks_user_status ON tasks(user_id, status);

-- =========================
-- 3) TASK_SESSIONS (actual logs)
-- =========================
CREATE TABLE IF NOT EXISTS task_sessions (
  session_id           UUID PRIMARY KEY,
  user_id              UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  task_id              UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,

  started_at           TIMESTAMPTZ NOT NULL,
  ended_at             TIMESTAMPTZ,
  actual_duration_min  INTEGER CHECK (actual_duration_min IS NULL OR actual_duration_min >= 0),

  status               task_status NOT NULL DEFAULT 'in_progress',
  interruptions_count  INTEGER NOT NULL DEFAULT 0 CHECK (interruptions_count >= 0),

  context_location     TEXT,
  notes                TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_time ON task_sessions(user_id, started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_task_time ON task_sessions(task_id, started_at);

-- =========================
-- 4) PRODUCTIVITY_SIGNALS
-- =========================
CREATE TABLE IF NOT EXISTS productivity_signals (
  signal_id            UUID PRIMARY KEY,
  user_id              UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  captured_at          TIMESTAMPTZ NOT NULL,

  day_of_week          SMALLINT CHECK (day_of_week BETWEEN 0 AND 6),
  hour_of_day          SMALLINT CHECK (hour_of_day BETWEEN 0 AND 23),

  screen_time_min      INTEGER CHECK (screen_time_min IS NULL OR screen_time_min >= 0),
  sleep_hours          NUMERIC(4,2) CHECK (sleep_hours IS NULL OR sleep_hours >= 0),

  mood_energy          SMALLINT CHECK (mood_energy IS NULL OR mood_energy BETWEEN 1 AND 5),
  focus_self_report    SMALLINT CHECK (focus_self_report IS NULL OR focus_self_report BETWEEN 1 AND 5),

  extra                JSONB
);

CREATE INDEX IF NOT EXISTS idx_signals_user_time ON productivity_signals(user_id, captured_at);

-- =========================
-- 5) TAGS + TASK_TAGS
-- =========================
CREATE TABLE IF NOT EXISTS tags (
  tag_id               UUID PRIMARY KEY,
  user_id              UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  name                 TEXT NOT NULL,
  UNIQUE(user_id, name)
);

CREATE TABLE IF NOT EXISTS task_tags (
  task_id              UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  tag_id               UUID NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
  PRIMARY KEY (task_id, tag_id)
);

-- =========================
-- 6) RECOMMENDATIONS
-- =========================
CREATE TABLE IF NOT EXISTS recommendations (
  rec_id                 UUID PRIMARY KEY,
  user_id                UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  task_id                UUID REFERENCES tasks(task_id) ON DELETE SET NULL,

  created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  model_version          TEXT NOT NULL,
  features_version       TEXT,

  recommended_start      TIMESTAMPTZ,
  recommended_duration_min INTEGER CHECK (recommended_duration_min IS NULL OR recommended_duration_min > 0),

  predicted_difficulty_level SMALLINT CHECK (predicted_difficulty_level BETWEEN 1 AND 3),
  rationale              TEXT,
  metadata               JSONB
);

CREATE INDEX IF NOT EXISTS idx_rec_user_time ON recommendations(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_rec_task ON recommendations(task_id);

-- =========================
-- 7) RECOMMENDATION_FEEDBACK
-- =========================
CREATE TABLE IF NOT EXISTS recommendation_feedback (
  rec_id                 UUID PRIMARY KEY REFERENCES recommendations(rec_id) ON DELETE CASCADE,
  action                 rec_action NOT NULL,
  responded_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  followed               BOOLEAN,
  actual_start           TIMESTAMPTZ,
  actual_duration_min    INTEGER CHECK (actual_duration_min IS NULL OR actual_duration_min >= 0),

  outcome_status         task_status,
  comment                TEXT
);

-- =========================
-- 8) TASK_TEXT_FEATURES (optional NLP)
-- =========================
CREATE TABLE IF NOT EXISTS task_text_features (
  task_id                UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
  cleaned_text           TEXT,
  keywords               TEXT[],
  topic_label            TEXT,
  embedding              REAL[],
  updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

def main():
  # Example: "postgresql://user:password@localhost:5432/cogpass"
  db_url = os.getenv("DATABASE_URL")
  if not db_url:
    raise RuntimeError("Set DATABASE_URL env var, e.g. postgresql://user:pass@localhost:5432/dbname")

  with psycopg.connect(db_url) as conn:
    conn.execute("SELECT 1;")  # quick connection test
    with conn.cursor() as cur:
      cur.execute(DDL)
    conn.commit()

  print("✅ Database schema created/updated successfully.")

if __name__ == "__main__":
  main()
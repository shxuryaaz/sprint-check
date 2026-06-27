"""Shared plan storage in Postgres (Neon) for multi-user editing.

A plan is one generated sprint doc with a stable id (its URL). Each person's
section is a row that saves independently, so two people editing DIFFERENT
people's goals never collide. Same-section edits are last-write-wins.

If DATABASE_URL is unset (local dev without a DB), every function is a safe
no-op and the app falls back to its file-based run logs.
"""
import json
import os

DATABASE_URL = os.environ.get("DATABASE_URL")


def enabled() -> bool:
    return bool(DATABASE_URL)


def _conn():
    import psycopg2  # imported lazily so the app runs without the DB locally
    return psycopg2.connect(DATABASE_URL, connect_timeout=10)


def init_db() -> None:
    if not enabled():
        return
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            """CREATE TABLE IF NOT EXISTS plans (
                id text PRIMARY KEY,
                sprint_label text,
                header text,
                process_improvement text,
                meeting_summary text,
                applied jsonb,
                created_at timestamptz DEFAULT now(),
                updated_at timestamptz DEFAULT now()
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS plan_sections (
                plan_id text REFERENCES plans(id) ON DELETE CASCADE,
                idx int,
                name text,
                body text,
                updated_at timestamptz DEFAULT now(),
                updated_by text,
                PRIMARY KEY (plan_id, idx)
            )"""
        )


def save_plan(plan_id: str, meta: dict, sections: list, by: "str | None" = None) -> None:
    """Create/replace a plan and its sections (called on generate)."""
    if not enabled():
        return
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            """INSERT INTO plans (id, sprint_label, header, process_improvement,
                                  meeting_summary, applied)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET
                 sprint_label=EXCLUDED.sprint_label, header=EXCLUDED.header,
                 process_improvement=EXCLUDED.process_improvement,
                 meeting_summary=EXCLUDED.meeting_summary,
                 applied=EXCLUDED.applied, updated_at=now()""",
            (plan_id, meta.get("sprint_label"), meta.get("header"),
             meta.get("process_improvement"), meta.get("meeting_summary"),
             json.dumps(meta.get("applied") or {})),
        )
        # Replace sections wholesale so a re-save can't leave stale rows behind.
        cur.execute("DELETE FROM plan_sections WHERE plan_id=%s", (plan_id,))
        for i, s in enumerate(sections):
            cur.execute(
                """INSERT INTO plan_sections (plan_id, idx, name, body, updated_by)
                   VALUES (%s, %s, %s, %s, %s)""",
                (plan_id, i, s.get("name"), s.get("text"), by),
            )


def update_section(plan_id: str, idx: int, name: str, body: str,
                   by: "str | None" = None) -> "dict | None":
    """Save one person's section (autosave). Last-write-wins. Returns the new
    {updated_at, updated_by} or None if the plan doesn't exist / DB is off."""
    if not enabled():
        return None
    with _conn() as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM plans WHERE id=%s", (plan_id,))
        if not cur.fetchone():
            return None
        cur.execute(
            """INSERT INTO plan_sections (plan_id, idx, name, body, updated_by)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (plan_id, idx) DO UPDATE SET
                 name=EXCLUDED.name, body=EXCLUDED.body,
                 updated_at=now(), updated_by=EXCLUDED.updated_by
               RETURNING updated_at, updated_by""",
            (plan_id, idx, name, body, by),
        )
        row = cur.fetchone()
        cur.execute("UPDATE plans SET updated_at=now() WHERE id=%s", (plan_id,))
        return {"updated_at": row[0].isoformat(), "updated_by": row[1]}


def get_plan(plan_id: str) -> "dict | None":
    """Load a plan + its sections (latest saved state)."""
    if not enabled():
        return None
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT sprint_label, header, process_improvement, meeting_summary, applied "
            "FROM plans WHERE id=%s", (plan_id,))
        p = cur.fetchone()
        if not p:
            return None
        cur.execute(
            "SELECT idx, name, body, updated_at, updated_by FROM plan_sections "
            "WHERE plan_id=%s ORDER BY idx", (plan_id,))
        rows = cur.fetchall()
    sections = [
        {"idx": r[0], "name": r[1], "text": r[2] or "",
         "updated_at": r[3].isoformat() if r[3] else None, "updated_by": r[4]}
        for r in rows
    ]
    return {
        "sprint_label": p[0], "header": p[1], "process_improvement": p[2],
        "meeting_summary": p[3], "applied": p[4] or {}, "sections": sections,
    }


def list_plans() -> list:
    if not enabled():
        return []
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            """SELECT p.id, p.sprint_label, p.meeting_summary, p.updated_at,
                      count(s.idx) AS people
               FROM plans p LEFT JOIN plan_sections s ON s.plan_id = p.id
               GROUP BY p.id ORDER BY p.updated_at DESC LIMIT 100""")
        rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1] or "Sprint Goals",
         "summary": r[2] or "", "date": r[3].date().isoformat() if r[3] else "",
         "people_count": r[4]}
        for r in rows
    ]

#!/usr/bin/env python3
"""Loop 1 web app: paste a Granola transcript, get a per-person sprint goals doc,
edit/approve/reject each person, and teach the pipeline from your edits.

Run:
    OPENAI_API_KEY=... python app.py
then open http://127.0.0.1:5000

Reuses the two-phase pipeline in loop1_pipeline.py. The contextual learning loop
has two memories, stored in memory/memory.json and injected into Phase 1 on every
future run:
  - semantic rules: per-person edits/rejections are distilled into general
    correction rules, then consolidated (deduped / contradictions resolved) into a
    bounded rule set you can prune in the UI.
  - episodic examples: each edited section is saved as a before→after example; the
    most relevant few (lexical retrieval vs the transcript) are injected as
    few-shot guidance.
"""

import datetime
import json
import os
import re
import sys
import uuid
from collections import defaultdict

from flask import Flask, jsonify, render_template, request

import loop1_pipeline as pipeline

try:
    import connectors  # Google Calendar/Gmail/Drive (optional — needs google libs)
except Exception:  # noqa: BLE001
    connectors = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
MEMORY_PATH = os.path.join(MEMORY_DIR, "memory.json")          # v2 structured store
LESSONS_PATH = os.path.join(MEMORY_DIR, "learned_lessons.md")  # legacy, migrated in
CONTEXT_PATH = os.path.join(BASE_DIR, "context", "project_context.md")
# Drop extra context files here (North Star export, client SOWs, a calendar
# dump, etc.) — every .md/.txt in this folder is injected alongside the main
# context. This is also the seam where live connectors (Calendar/Gmail/Drive)
# would write their fetched data.
SOURCES_DIR = os.path.join(BASE_DIR, "context", "sources")
# Every generate + every feedback is logged here so a run can be replayed/debugged
# end-to-end (inputs applied, the extraction JSON, the formatted doc, and edits).
RUNS_DIR = os.path.join(BASE_DIR, "runs")

# How many past correction examples (episodic memory) to inject per generation.
MAX_EXAMPLES_INJECTED = 3
# Cap each injected example so the prompt stays bounded.
EXAMPLE_CHAR_CAP = 700

DEFAULT_TEAM = ["Shaurya", "Shiv", "Antonio", "Cameron"]

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "their", "they",
    "them", "will", "have", "has", "are", "was", "were", "our", "out", "not",
    "but", "you", "your", "all", "can", "his", "her", "she", "him", "who", "how",
    "what", "when", "which", "each", "per", "any", "more", "most", "some", "than",
    "then", "also", "about", "would", "could", "should", "a", "an", "of", "to",
    "in", "on", "is", "it", "be", "as", "we", "i", "or", "by", "at", "so",
}

app = Flask(__name__)


# --------------------------------------------------------------------------- #
# Lessons / context file helpers
# --------------------------------------------------------------------------- #
def _empty_memory() -> dict:
    return {"rules": [], "examples": []}


def load_memory() -> dict:
    """Load the structured memory store. Migrates a legacy learned_lessons.md
    (bullet list of rules) into the new store on first load if present."""
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                mem = json.load(f)
            mem.setdefault("rules", [])
            mem.setdefault("examples", [])
            for r in mem["rules"]:
                r.setdefault("person", None)  # older rules are team-wide
            return mem
        except (json.JSONDecodeError, OSError):
            return _empty_memory()

    mem = _empty_memory()
    # One-time migration from the legacy markdown file.
    if os.path.exists(LESSONS_PATH):
        with open(LESSONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("- "):
                    text = line[2:].strip()
                    if text:
                        mem["rules"].append(
                            {"id": uuid.uuid4().hex[:8], "text": text, "person": None,
                             "date": datetime.date.today().isoformat()}
                        )
        if mem["rules"]:
            save_memory(mem)
    return mem


def save_memory(mem: dict) -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2)


def north_star_candidates(context_text: str) -> str:
    """Code (not the LLM) parses the North Star table and grabs the rows that are
    Behind / In progress this sprint, so they can't be missed. Returns an explicit
    candidate block to prepend to the context, or '' if none found.

    Expects lines like: '2. Maintain & groom product backlog — 25% — Wk Jun 22 — Behind'
    """
    if not context_text:
        return ""
    active = []
    for line in context_text.splitlines():
        if not re.match(r"^\s*\d+\.", line):
            continue
        parts = [p.strip() for p in line.split("—")]
        if len(parts) < 4:
            continue
        status = parts[-1].lower()
        if "behind" not in status and "in progress" not in status:
            continue
        item = re.sub(r"^\s*\d+\.\s*", "", parts[0])
        target = parts[-2]
        active.append(f"- {item} ({parts[-1]}, target {target})")
    if not active:
        return ""
    return (
        "ACTIVE JOURNEY NORTH STAR DELIVERABLES THIS SPRINT (code-selected — turn "
        "EACH one into a goal for Shiv or Antonio, do not skip any):\n"
        + "\n".join(active)
    )


def source_files() -> list:
    """Names of the extra context files currently in context/sources/."""
    if not os.path.isdir(SOURCES_DIR):
        return []
    return [
        f for f in sorted(os.listdir(SOURCES_DIR))
        if f.lower().endswith((".md", ".txt")) and not f.lower().startswith("readme")
    ]


def log_run(kind: str, data: dict) -> None:
    """Write a timestamped JSON record of a generate/feedback so runs can be
    inspected end-to-end. Never raises — logging must not break a request."""
    try:
        os.makedirs(RUNS_DIR, exist_ok=True)
        ts = datetime.datetime.now()
        data = {"timestamp": ts.isoformat(timespec="seconds"), **data}
        path = os.path.join(RUNS_DIR, f"{ts.strftime('%Y%m%d_%H%M%S')}_{kind}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:  # noqa: BLE001
        pass


def rules_text(mem: dict) -> str:
    """Render rules for the prompt, grouped into team-wide vs per-person so a
    preference learned from one person doesn't bleed onto everyone else."""
    rules = mem.get("rules", [])
    general = [r["text"] for r in rules if not r.get("person")]
    per_person = defaultdict(list)
    for r in rules:
        if r.get("person"):
            per_person[r["person"]].append(r["text"])
    lines = []
    if general:
        lines.append("Apply to everyone:")
        lines += [f"- {t}" for t in general]
    for person, texts in per_person.items():
        lines.append(f"For {person} specifically:")
        lines += [f"- {t}" for t in texts]
    return "\n".join(lines)


def _tokens(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def relevant_examples(mem: dict, transcript: str, k: int = MAX_EXAMPLES_INJECTED) -> list:
    """Lexical retrieval: score each stored example by word overlap with the
    transcript (plus a boost when the person's name appears), return the top k."""
    examples = mem.get("examples", [])
    if not examples:
        return []
    t_tokens = _tokens(transcript)
    t_lower = (transcript or "").lower()
    scored = []
    for i, ex in enumerate(examples):
        overlap = len(_tokens(ex.get("after", "")) & t_tokens)
        name = (ex.get("name") or "").lower()
        if name and name in t_lower:
            overlap += 5
        # Tie-break toward more recent examples (higher index = newer).
        scored.append((overlap, i, ex))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [ex for score, _, ex in scored[:k] if score > 0]


def format_examples(examples: list) -> str:
    """Render retrieved examples as BEFORE→AFTER blocks for the prompt."""
    blocks = []
    for ex in examples:
        before = (ex.get("before") or "").strip()[:EXAMPLE_CHAR_CAP]
        after = (ex.get("after") or "").strip()[:EXAMPLE_CHAR_CAP]
        blocks.append(
            f"# {ex.get('name', 'Unknown')}\n"
            f"BEFORE (draft the human corrected):\n{before}\n\n"
            f"AFTER (what the human wanted):\n{after}"
        )
    return "\n\n----\n\n".join(blocks)


def read_context() -> "str | None":
    """Assemble project context from the main file PLUS any extra source files
    dropped in context/sources/ (North Star export, client SOWs, calendar dump,
    connector output, ...). Returns the concatenation, or None if nothing exists."""
    parts = []
    if os.path.exists(CONTEXT_PATH):
        with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
            main = f.read().strip()
        if main:
            parts.append(main)
    if os.path.isdir(SOURCES_DIR):
        for fn in sorted(os.listdir(SOURCES_DIR)):
            if not fn.lower().endswith((".md", ".txt")):
                continue
            if fn.lower().startswith("readme"):
                continue
            try:
                with open(os.path.join(SOURCES_DIR, fn), "r", encoding="utf-8") as f:
                    content = f.read().strip()
            except OSError:
                continue
            if content:
                parts.append(f"----- SOURCE: {fn} -----\n{content}")
    return "\n\n".join(parts) if parts else None


# --------------------------------------------------------------------------- #
# Per-person section split
# --------------------------------------------------------------------------- #
def split_into_sections(formatted_doc: str, people: "list[dict]") -> "list[dict]":
    """Split the formatted sprint doc into one block per person.

    The formatted doc looks like:
        Sprint goals (LABEL)
        <person 1 block>
        ---
        <person 2 block>
        ...
    We split on lines that are exactly '---', drop the leading
    'Sprint goals (...)' header from the first chunk, and map chunks to people by
    order. Falls back to one section per person name if counts disagree.
    """
    lines = formatted_doc.splitlines()
    # Drop the leading "Sprint goals (...)" header and the team-wide
    # "Process Improvement for the Week:" line — neither belongs to a person card.
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start < len(lines) and lines[start].strip().lower().startswith("sprint goals"):
        start += 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start < len(lines) and lines[start].strip().lower().startswith("process improvement"):
        start += 1
    body = "\n".join(lines[start:]).strip()

    chunks = [c.strip() for c in body.split("\n---\n")]
    # Some models emit '---' with surrounding blank lines collapsed differently;
    # normalize by also splitting on a bare '---' line.
    if len(chunks) == 1:
        rebuilt, cur = [], []
        for ln in body.splitlines():
            if ln.strip() == "---":
                rebuilt.append("\n".join(cur).strip())
                cur = []
            else:
                cur.append(ln)
        rebuilt.append("\n".join(cur).strip())
        chunks = [c for c in rebuilt if c]

    names = [p.get("name", f"Person {i+1}") for i, p in enumerate(people)]
    sections = []
    if len(chunks) == len(names):
        for name, text in zip(names, chunks):
            sections.append({"name": name, "text": text})
    else:
        # Counts disagree (e.g. an empty person). Best-effort: label by the first
        # non-empty line of each chunk, falling back to ordinal.
        for i, text in enumerate(chunks):
            first = next((l.strip() for l in text.splitlines() if l.strip()), "")
            name = names[i] if i < len(names) else (first or f"Person {i+1}")
            sections.append({"name": name, "text": text})
    return sections


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template(
        "index.html", default_team=", ".join(DEFAULT_TEAM)
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True, silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    if not transcript:
        return jsonify({"error": "Transcript is empty."}), 400

    sprint_label = (data.get("sprint_label") or "Current Sprint").strip()
    input_mode = (data.get("input_mode") or "transcript").strip()
    team_raw = (data.get("team") or "").strip()
    team_members = (
        [t.strip() for t in team_raw.split(",") if t.strip()] if team_raw else None
    )

    context_text = read_context()
    # Prepend the code-selected North Star deliverables so neither pass can miss them.
    if context_text:
        ns = north_star_candidates(context_text)
        if ns:
            context_text = ns + "\n\n" + context_text
    mem = load_memory()
    lessons_text = rules_text(mem) or None
    examples = relevant_examples(mem, transcript)
    examples_text = format_examples(examples) if examples else None

    try:
        if input_mode == "draft":
            # Layer-on-Granola: the pasted text is already a sprint-goals draft.
            # Structure it faithfully; the sources pass below adds what Granola
            # can't see (North Star / calendar / carry-over / missing owners).
            extraction = pipeline.parse_sprint_draft(transcript)
        else:
            extraction = pipeline.extract_goals(
                transcript,
                team_members,
                context_text=context_text,
                lessons_text=lessons_text,
                examples_text=examples_text,
            )
        # Second pass: pull goals directly from the live sources (calendar / North
        # Star / email / carry-overs), then merge — a dedicated pass reliably
        # surfaces source goals the transcript pass under-extracts.
        if context_text:
            try:
                src = pipeline.extract_goals_from_sources(context_text, team_members)
                extraction = pipeline.merge_extractions(extraction, src)
            except Exception:  # noqa: BLE001 - sources pass is best-effort, never block
                pass
        formatted = pipeline.format_sprint_doc(extraction, sprint_label)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI
        return jsonify({"error": f"Unexpected failure: {e}"}), 500

    people = extraction.get("people", [])
    sections = split_into_sections(formatted, people)
    process_improvement = (extraction.get("process_improvement") or "").strip()
    header = f"Sprint goals ({sprint_label})"
    if process_improvement:
        header += f"\n\nProcess Improvement for the Week: {process_improvement}"

    log_run("generate", {
        "sprint_label": sprint_label,
        "input_mode": input_mode,
        "team": team_members,
        "transcript_chars": len(transcript),
        "context_applied": bool(context_text),
        "context_chars": len(context_text or ""),
        "sources_applied": source_files(),
        "lessons_applied": bool(lessons_text),
        "examples_applied": len(examples),
        "process_improvement": process_improvement,
        "meeting_summary": extraction.get("meeting_summary", ""),
        "people_count": len(people),
        "extraction": extraction,      # the Phase 1 JSON
        "formatted_doc": formatted,     # the exact doc shown to the user
    })

    return jsonify(
        {
            "sprint_label": sprint_label,
            "header": header,
            "process_improvement": process_improvement,
            "meeting_summary": extraction.get("meeting_summary", ""),
            "sections": sections,
            "lessons_applied": bool(lessons_text),
            "examples_applied": len(examples),
            "context_applied": bool(context_text),
        }
    )


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(force=True, silent=True) or {}
    items = data.get("items") or []
    meeting_summary = data.get("meeting_summary")
    if not isinstance(items, list) or not items:
        return jsonify({"error": "No feedback items provided."}), 400

    mem = load_memory()
    today = datetime.date.today().isoformat()

    # 1) Episodic memory: store every edited section as a before→after example.
    new_examples = 0
    for it in items:
        if it.get("status") == "edited":
            before = (it.get("original") or "").strip()
            after = (it.get("final") or "").strip()
            if before and after and before != after:
                mem["examples"].append(
                    {
                        "id": uuid.uuid4().hex[:8],
                        "name": it.get("name", "Unknown"),
                        "before": before,
                        "after": after,
                        "date": today,
                    }
                )
                new_examples += 1

    # 2) Semantic memory: distill candidate rules (each tagged with a person or
    #    None), then consolidate within each scope bucket so team-wide and
    #    per-person rules dedupe/merge independently and never cross-contaminate.
    try:
        candidates = pipeline.distill_feedback(items, meeting_summary=meeting_summary)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Unexpected failure: {e}"}), 500

    if candidates:
        buckets = defaultdict(lambda: {"existing": [], "candidate": []})
        for r in mem["rules"]:
            buckets[r.get("person")]["existing"].append(r["text"])
        for c in candidates:
            buckets[c.get("person")]["candidate"].append(c["text"])
        # Preserve ids of surviving rules so the UI's delete buttons keep working.
        id_lookup = {
            (r.get("person"), r["text"].strip().lower()): r["id"] for r in mem["rules"]
        }
        rebuilt = []
        for person, b in buckets.items():
            merged = (
                pipeline.consolidate_rules(b["existing"], b["candidate"])
                if b["candidate"] else b["existing"]
            )
            for text in merged:
                key = (person, text.strip().lower())
                rebuilt.append(
                    {
                        "id": id_lookup.get(key, uuid.uuid4().hex[:8]),
                        "text": text,
                        "person": person,
                        "date": today,
                    }
                )
        mem["rules"] = rebuilt

    save_memory(mem)

    log_run("feedback", {
        "items": items,                              # before/after/status/reason per person
        "added_rules": [c["text"] for c in candidates],
        "new_examples": new_examples,
        "rules_after": mem["rules"],
        "example_count": len(mem["examples"]),
    })

    return jsonify(
        {
            "added_rules": [c["text"] for c in candidates],
            "new_examples": new_examples,
            "rules": mem["rules"],
            "example_count": len(mem["examples"]),
        }
    )


@app.route("/api/lessons", methods=["GET"])
def api_lessons():
    mem = load_memory()
    return jsonify(
        {
            "rules": mem.get("rules", []),
            "example_count": len(mem.get("examples", [])),
        }
    )


@app.route("/api/lessons/delete", methods=["POST"])
def api_lessons_delete():
    data = request.get_json(force=True, silent=True) or {}
    rule_id = data.get("id")
    if not rule_id:
        return jsonify({"error": "No rule id provided."}), 400
    mem = load_memory()
    before = len(mem["rules"])
    mem["rules"] = [r for r in mem["rules"] if r.get("id") != rule_id]
    save_memory(mem)
    return jsonify(
        {
            "removed": before - len(mem["rules"]),
            "rules": mem["rules"],
            "example_count": len(mem.get("examples", [])),
        }
    )


@app.route("/api/connectors", methods=["GET"])
def api_connectors():
    """Status + a live preview of exactly what each Google source returns."""
    if connectors is None:
        return jsonify({"available": False, "connected": False,
                        "calendar": [], "gmail": [], "drive": []})
    try:
        data = connectors.preview()
        data["available"] = True
        return jsonify(data)
    except Exception as e:  # noqa: BLE001
        return jsonify({"available": True, "connected": False, "error": str(e),
                        "calendar": [], "gmail": [], "drive": []})


@app.route("/api/connectors/connect", methods=["POST"])
def api_connectors_connect():
    """Run the one-time Google OAuth consent (opens a browser on this machine)."""
    if connectors is None:
        return jsonify({"error": "Google libraries are not installed."}), 500
    try:
        connectors.authorize()
        return jsonify(connectors.preview() | {"available": True})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Could not connect: {e}"}), 500


@app.route("/api/connectors/disconnect", methods=["POST"])
def api_connectors_disconnect():
    if connectors is not None:
        connectors.disconnect()
    return jsonify({"available": connectors is not None, "connected": False,
                    "calendar": [], "gmail": [], "drive": []})


@app.route("/api/connectors/drives", methods=["GET"])
def api_connectors_drives():
    """List the Shared Drives the user can pick from, plus the current selection."""
    if connectors is None:
        return jsonify({"drives": [], "selected": {}})
    try:
        return jsonify({"drives": connectors.list_shared_drives(),
                        "selected": connectors.get_drive_source()})
    except Exception as e:  # noqa: BLE001
        return jsonify({"drives": [], "selected": connectors.get_drive_source(),
                        "error": str(e)})


@app.route("/api/connectors/drive-source", methods=["POST"])
def api_connectors_drive_source():
    """Choose which Drive to read: recent (my drive), a Shared Drive, or a folder."""
    if connectors is None:
        return jsonify({"error": "Google libraries are not installed."}), 500
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode", "recent")
    name = data.get("name", "")
    raw = (data.get("id") or "").strip()
    if mode == "folder":
        rid, name = connectors._extract_folder_id(raw), name or "Folder"
    elif mode == "drive":
        rid = raw
    else:
        mode, rid, name = "recent", "", "My Drive (recent)"
    cfg = connectors.set_drive_source(mode, rid, name)
    try:
        return jsonify({"selected": cfg, "drive": connectors.fetch_drive()})
    except Exception as e:  # noqa: BLE001
        return jsonify({"selected": cfg, "drive": [], "error": str(e)})


@app.route("/api/connectors/calendars", methods=["POST"])
def api_connectors_calendars():
    """Set whose calendars to read (the user's + teammates' emails)."""
    if connectors is None:
        return jsonify({"error": "Google libraries are not installed."}), 500
    data = request.get_json(force=True, silent=True) or {}
    emails = data.get("emails")
    if isinstance(emails, str):
        emails = [e.strip() for e in emails.split(",")]
    cals = connectors.set_team_calendars(emails or [])
    try:
        detail = connectors.fetch_calendar_detailed()
        return jsonify({"team_calendars": cals, "calendar": detail["events"],
                        "calendars": detail["calendars"]})
    except Exception as e:  # noqa: BLE001
        return jsonify({"team_calendars": cals, "calendar": [], "calendars": [],
                        "error": str(e)})


@app.route("/api/connectors/sync", methods=["POST"])
def api_connectors_sync():
    """Write the fetched data into context/sources/ so the next Generate uses it."""
    if connectors is None:
        return jsonify({"error": "Google libraries are not installed."}), 500
    try:
        written = connectors.sync_to_sources()
        return jsonify({"written": written})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    try:
        pipeline.init_client()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    # debug=True enables auto-reload: edits to app.py / templates / the pipeline
    # take effect without manually restarting the server.
    # Port is configurable via PORT env var (macOS uses 5000 for AirPlay).
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)

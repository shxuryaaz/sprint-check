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
import store  # shared plan storage (Neon/Postgres) for multi-user editing

try:
    import connectors  # Google Calendar/Gmail/Drive (optional — needs google libs)
except Exception:  # noqa: BLE001
    connectors = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Persistent state (learned memory, run logs, synced context) lives under DATA_DIR.
# Locally this defaults to the repo dir. On a host where the repo is wiped on each
# deploy (e.g. Render), point DATA_DIR at a persistent disk — DATA_DIR=/var/data —
# so memory and history survive deploys.
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
MEMORY_DIR = os.path.join(DATA_DIR, "memory")
MEMORY_PATH = os.path.join(MEMORY_DIR, "memory.json")          # v2 structured store
LESSONS_PATH = os.path.join(MEMORY_DIR, "learned_lessons.md")  # legacy, migrated in
CONTEXT_PATH = os.path.join(DATA_DIR, "context", "project_context.md")
# Drop extra context files here (North Star export, client SOWs, a calendar
# dump, etc.) — every .md/.txt in this folder is injected alongside the main
# context. This is also the seam where live connectors (Calendar/Gmail/Drive)
# would write their fetched data.
SOURCES_DIR = os.path.join(DATA_DIR, "context", "sources")
# Every generate + every feedback is logged here so a run can be replayed/debugged
# end-to-end (inputs applied, the extraction JSON, the formatted doc, and edits).
RUNS_DIR = os.path.join(DATA_DIR, "runs")

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
        "JOURNEY NORTH STAR ITEMS CURRENTLY BEHIND / IN PROGRESS (these are Shiv's & "
        "Antonio's core Journey deliverables — assign each ONLY to Shiv or Antonio, "
        "never to Cameron or Shaurya). Turn EACH item below that is overdue or due "
        "within the next ~2 weeks into a real goal for its owner — these make up the "
        "bulk of Shiv's and Antonio's sprint, so include them. SKIP only the far-future "
        "ones (target a month or more out). The North Star ranks below the meeting and "
        "calendar, so prefer the meeting's wording when an item was also discussed there, "
        "but do NOT drop these:\n"
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
@app.route("/ping")
def ping():
    """Liveness probe for Render health checks and the UptimeRobot keep-alive."""
    return "ok", 200


@app.route("/")
def index():
    return render_template(
        "index.html", default_team=", ".join(DEFAULT_TEAM)
    )


def _canonicalize_people(extraction: dict, team_members: "list[str] | None") -> dict:
    """Map source/calendar-derived name variants (e.g. 'Shauryajps' from a calendar
    account handle) to the canonical team name ('Shaurya') and merge duplicate
    person entries so a person never appears twice."""
    if not team_members:
        return extraction

    def canonical(name: str) -> str:
        low = (name or "").strip().lower()
        for t in team_members:
            tl = t.strip().lower()
            if low == tl or low.startswith(tl) or tl in low:
                return t.strip()
        return (name or "").strip()

    order, by_canon = [], {}
    for p in extraction.get("people", []):
        cname = canonical(p.get("name", ""))
        if cname not in by_canon:
            merged = dict(p)
            merged["name"] = cname
            by_canon[cname] = merged
            order.append(cname)
        else:
            tgt = by_canon[cname]
            existing = tgt.get("goals", []) or []
            for g in (p.get("goals", []) or []):
                # Skip goals whose title duplicates one already present for this person.
                if not any(pipeline._titles_similar(g.get("title", ""), e.get("title", ""))
                           for e in existing):
                    existing.append(g)
            tgt["goals"] = existing
            if not tgt.get("kaizen") and p.get("kaizen"):
                tgt["kaizen"] = p.get("kaizen")
    extraction["people"] = [by_canon[c] for c in order]
    return extraction


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
    ns_block = ""
    if context_text:
        ns_block = north_star_candidates(context_text)
        if ns_block:
            context_text = ns_block + "\n\n" + context_text
    mem = load_memory()
    lessons_text = rules_text(mem) or None
    examples = relevant_examples(mem, transcript)
    examples_text = format_examples(examples) if examples else None

    # The transparency trace: exactly what shaped this generation, split into the
    # team-wide preferences and each person's own preferences, so the UI can show
    # "here's what Agilow applied to <person>".
    general_prefs = [r["text"] for r in mem.get("rules", []) if not r.get("person")]
    person_prefs = defaultdict(list)
    for r in mem.get("rules", []):
        if r.get("person"):
            person_prefs[r["person"]].append(r["text"])
    applied = {
        "general_prefs": general_prefs,
        "person_prefs": person_prefs,
        "examples": [{"name": e.get("name"), "before": e.get("before"),
                      "after": e.get("after")} for e in examples],
        "context_applied": bool(context_text),
        "context_sources": source_files(),
        "north_star": ns_block,
    }

    try:
        if input_mode == "draft":
            # Layer-on-Granola (primary flow): the pasted text is the Granola
            # template's sprint-goals output. Structure it faithfully BUT apply
            # the learned corrections + past edit examples so it reflects what the
            # AI has learned from earlier human edits. The sources pass below then
            # adds what Granola can't see (North Star / calendar / carry-over).
            extraction = pipeline.parse_sprint_draft(
                transcript,
                lessons_text=lessons_text,
                examples_text=examples_text,
            )
        else:
            extraction = pipeline.extract_goals(
                transcript,
                team_members,
                context_text=context_text,
                lessons_text=lessons_text,
                examples_text=examples_text,
            )
        # Second pass: pull goals directly from the live sources (calendar / North
        # Star / email / carry-overs) for the FULL team — this is what generates
        # substantive goals for people the Granola draft doesn't cover (e.g. Shiv,
        # whose Journey/North-Star work isn't in the pasted template) — then merge.
        if context_text:
            try:
                src = pipeline.extract_goals_from_sources(context_text, team_members)
                extraction = pipeline.merge_extractions(extraction, src)
            except Exception:  # noqa: BLE001 - sources pass is best-effort, never block
                pass
        # Merge name variants (e.g. calendar handle 'Shauryajps' -> 'Shaurya').
        extraction = _canonicalize_people(extraction, team_members)
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

    # Persist as a shared, editable plan with its own id (its URL). The team opens
    # this id and edits sections concurrently; each section autosaves on its own.
    user = (data.get("user") or "").strip() or None
    plan_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    try:
        store.save_plan(plan_id, {
            "sprint_label": sprint_label, "header": header,
            "process_improvement": process_improvement,
            "meeting_summary": extraction.get("meeting_summary", ""),
            "applied": applied,
        }, sections, by=user)
    except Exception:  # noqa: BLE001 - never let storage break a generate
        pass

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
        "applied": applied,            # the transparency trace shown in the UI
        "extraction": extraction,      # the Phase 1 JSON
        "formatted_doc": formatted,     # the exact doc shown to the user
    })

    return jsonify(
        {
            "plan_id": plan_id,
            "shared": store.enabled(),
            "sprint_label": sprint_label,
            "header": header,
            "process_improvement": process_improvement,
            "meeting_summary": extraction.get("meeting_summary", ""),
            "sections": sections,
            "applied": applied,
            "lessons_applied": bool(lessons_text),
            "examples_applied": len(examples),
            "context_applied": bool(context_text),
        }
    )


def _next_goal_number(section_text: str) -> int:
    """The number to give the next goal (max existing leading 'N.' + 1)."""
    nums = [int(m.group(1)) for m in re.finditer(r"(?m)^\s*(\d+)\.\s", section_text or "")]
    return (max(nums) + 1) if nums else 1


def _existing_titles(section_text: str) -> list:
    """Existing goal titles in a section (so /api/add doesn't duplicate them)."""
    titles = []
    for m in re.finditer(r"(?m)^\s*\d+\.\s+(.*)$", section_text or ""):
        title = re.sub(r"\s*\(\d+(?:\.\d+)?\s*points?\)\s*$", "", m.group(1).strip())
        if title:
            titles.append(title)
    return titles


@app.route("/api/add", methods=["POST"])
def api_add():
    """Turn a person's free-form brain-dump (Wispr Flow voice-to-text) into one or
    more properly-formatted goal blocks to append to their section, applying the
    learned rules + examples so the additions match the team's style."""
    data = request.get_json(force=True, silent=True) or {}
    person = (data.get("name") or "this person").strip()
    dump = (data.get("dump") or "").strip()
    section = data.get("section") or ""
    if not dump:
        return jsonify({"error": "Nothing to add — type or dictate what to add first."}), 400

    mem = load_memory()
    # Scope learned rules to general + this person only, so one teammate's
    # preferences don't reshape another's additions.
    scoped = {"rules": [r for r in mem.get("rules", [])
                        if not r.get("person") or r.get("person") == person],
              "examples": mem.get("examples", [])}
    lessons_text = rules_text(scoped) or None
    examples = relevant_examples(mem, dump)
    examples_text = format_examples(examples) if examples else None

    try:
        added = pipeline.format_added_goals(
            dump,
            person,
            start_number=_next_goal_number(section),
            existing_titles=_existing_titles(section),
            lessons_text=lessons_text,
            examples_text=examples_text,
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Unexpected failure: {e}"}), 500

    log_run("add", {"name": person, "dump": dump, "added": added,
                    "lessons_applied": bool(lessons_text),
                    "examples_applied": len(examples)})
    return jsonify({"added": added})


@app.route("/api/change-section", methods=["POST"])
def api_change_section():
    """The single 'Tell Agilow what to change' action: apply one free-form
    instruction (remove / reword / add / redo) to a person's CURRENT section,
    in place. Optionally remember the instruction as a lasting preference."""
    data = request.get_json(force=True, silent=True) or {}
    person = (data.get("name") or "this person").strip()
    section = data.get("section") or ""
    instruction = (data.get("instruction") or "").strip()
    save_pref = bool(data.get("save_pref"))
    if not instruction:
        return jsonify({"error": "Say what to change first."}), 400
    if not section.strip():
        return jsonify({"error": "Nothing to change yet — generate a plan first."}), 400

    mem = load_memory()
    person_rules = [r["text"] for r in mem.get("rules", []) if r.get("person") == person]
    general_rules = [r["text"] for r in mem.get("rules", []) if not r.get("person")]
    scoped = {"rules": [r for r in mem.get("rules", [])
                        if not r.get("person") or r.get("person") == person],
              "examples": mem.get("examples", [])}
    lessons_text = rules_text(scoped) or None
    examples = relevant_examples(mem, instruction + " " + section)
    examples_text = format_examples(examples) if examples else None

    try:
        new_section = pipeline.refine_section(
            section, person, instruction,
            lessons_text=lessons_text, examples_text=examples_text)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Unexpected failure: {e}"}), 500

    saved_rule = None
    if save_pref:
        rule = {"id": uuid.uuid4().hex[:8], "text": instruction, "person": person,
                "date": datetime.date.today().isoformat()}
        mem["rules"].append(rule)
        save_memory(mem)
        saved_rule = rule
        person_rules = person_rules + [instruction]

    bits = [f"Applied your change to {person}'s goals: “{instruction}”."]
    if person_rules or general_rules:
        bits.append(f"Kept {person}'s preferences ({len(person_rules)}) and "
                    f"{len(general_rules)} team-wide in mind.")
    if saved_rule:
        bits.append("Saved it as a lasting preference.")
    reasoning = " ".join(bits)

    log_run("change", {"name": person, "instruction": instruction,
                       "saved_pref": bool(saved_rule), "section": new_section,
                       "reasoning": reasoning})
    return jsonify({"section": new_section, "reasoning": reasoning,
                    "saved_rule": saved_rule,
                    "rules": mem["rules"] if saved_rule else None})


@app.route("/api/regenerate-person", methods=["POST"])
def api_regenerate_person():
    """Regenerate ONE person's section from the transcript, applying their
    preferences + an optional one-off instruction ('modify the prompt for this
    person'). Returns the new section text plus a plain-language reasoning of what
    shaped it. Optionally saves the instruction as a durable preference."""
    data = request.get_json(force=True, silent=True) or {}
    person = (data.get("name") or "").strip()
    transcript = (data.get("transcript") or "").strip()
    instruction = (data.get("instruction") or "").strip()
    input_mode = (data.get("input_mode") or "transcript").strip()
    sprint_label = (data.get("sprint_label") or "Current Sprint").strip()
    save_pref = bool(data.get("save_pref"))
    if not person:
        return jsonify({"error": "No person specified."}), 400
    if not transcript:
        return jsonify({"error": "Paste the transcript above first — regenerate reads from it."}), 400

    context_text = read_context()
    ns_block = ""
    if context_text:
        ns_block = north_star_candidates(context_text)
        if ns_block:
            context_text = ns_block + "\n\n" + context_text

    mem = load_memory()
    # Only this person's preferences + the team-wide ones — never another teammate's.
    person_rules = [r["text"] for r in mem.get("rules", []) if r.get("person") == person]
    general_rules = [r["text"] for r in mem.get("rules", []) if not r.get("person")]
    scoped = {"rules": [r for r in mem.get("rules", [])
                        if not r.get("person") or r.get("person") == person],
              "examples": mem.get("examples", [])}
    lessons_text = rules_text(scoped) or ""
    if instruction:
        lessons_text += (
            f"\nFor {person} specifically — one-off instruction for THIS regeneration "
            f"(highest priority, override conflicting rules): {instruction}"
        )
    lessons_text = lessons_text or None
    examples = relevant_examples(mem, transcript)
    examples_text = format_examples(examples) if examples else None

    try:
        if input_mode == "draft":
            extraction = pipeline.parse_sprint_draft(
                transcript, lessons_text=lessons_text, examples_text=examples_text)
        else:
            extraction = pipeline.extract_goals(
                transcript, [person], context_text=context_text,
                lessons_text=lessons_text, examples_text=examples_text)
        if context_text:
            try:
                src = pipeline.extract_goals_from_sources(context_text, [person])
                extraction = pipeline.merge_extractions(extraction, src)
            except Exception:  # noqa: BLE001
                pass
        # Merge name variants (calendar handle -> canonical), then keep this person.
        extraction = _canonicalize_people(extraction, [person])
        people = extraction.get("people", [])
        match = [p for p in people if (p.get("name", "").strip().lower() == person.lower())]
        extraction["people"] = match or people[:1]
        if not extraction["people"]:
            return jsonify({"error": f"Could not extract goals for {person}."}), 500
        formatted = pipeline.format_sprint_doc(extraction, sprint_label)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Unexpected failure: {e}"}), 500

    sections = split_into_sections(formatted, extraction["people"])
    section = sections[0]["text"] if sections else ""

    saved_rule = None
    if save_pref and instruction:
        rule = {"id": uuid.uuid4().hex[:8], "text": instruction, "person": person,
                "date": datetime.date.today().isoformat()}
        mem["rules"].append(rule)
        save_memory(mem)
        saved_rule = rule
        person_rules = person_rules + [instruction]

    # Plain-language "thinking" of what shaped this regeneration.
    bits = [f"Applied {person}'s preferences ({len(person_rules)}) and {len(general_rules)} team-wide."]
    if instruction:
        bits.append(f"Followed your instruction: “{instruction}”.")
    if examples:
        bits.append(f"Drew on {len(examples)} past edit example{'s' if len(examples) != 1 else ''}.")
    if context_text:
        bits.append("Pulled from the transcript plus your synced context (calendar / North Star / email / Drive).")
    reasoning = " ".join(bits)

    log_run("regenerate", {"name": person, "instruction": instruction,
                           "input_mode": input_mode, "saved_pref": bool(saved_rule),
                           "section": section, "reasoning": reasoning})
    return jsonify({"section": section, "reasoning": reasoning,
                    "saved_rule": saved_rule,
                    "rules": mem["rules"] if saved_rule else None})


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
            "examples": mem.get("examples", []),
            "example_count": len(mem.get("examples", [])),
        }
    )


@app.route("/api/plan/<plan_id>", methods=["GET"])
def api_plan_get(plan_id):
    """Load a shared plan (latest saved sections) for collaborative editing."""
    if "/" in plan_id or "\\" in plan_id or ".." in plan_id:
        return jsonify({"error": "Bad plan id."}), 400
    if store.enabled():
        p = store.get_plan(plan_id)
        if not p:
            return jsonify({"error": "Plan not found."}), 404
        header = p.get("header") or f"Sprint goals ({p.get('sprint_label') or 'Sprint Goals'})"
        return jsonify({
            "plan_id": plan_id, "shared": True,
            "sprint_label": p.get("sprint_label"), "header": header,
            "process_improvement": p.get("process_improvement") or "",
            "meeting_summary": p.get("meeting_summary") or "",
            "sections": p.get("sections", []),
            "applied": p.get("applied") or {},
        })
    # No DB configured — fall back to the single-user file-based reopen.
    return api_plan_detail(plan_id)


@app.route("/api/plan/<plan_id>/section", methods=["POST"])
def api_plan_section(plan_id):
    """Autosave one person's section. Last-write-wins; different sections never
    collide. Returns {updated_at, updated_by}."""
    if "/" in plan_id or "\\" in plan_id or ".." in plan_id:
        return jsonify({"error": "Bad plan id."}), 400
    if not store.enabled():
        return jsonify({"error": "Shared editing isn't configured (no DATABASE_URL)."}), 400
    data = request.get_json(force=True, silent=True) or {}
    idx = data.get("idx")
    if idx is None:
        return jsonify({"error": "No section idx."}), 400
    try:
        res = store.update_section(
            plan_id, int(idx), (data.get("name") or "").strip(),
            data.get("text") or "", (data.get("user") or "").strip() or None)
    except (ValueError, TypeError):
        return jsonify({"error": "Bad section idx."}), 400
    if res is None:
        return jsonify({"error": "Plan not found."}), 404
    return jsonify(res)


@app.route("/api/plans", methods=["GET"])
def api_plans():
    """List past generated plans (newest first) — from the DB when shared editing
    is configured, otherwise from the local run logs."""
    if store.enabled():
        return jsonify({"plans": store.list_plans()})
    plans = []
    if os.path.isdir(RUNS_DIR):
        for fn in sorted(os.listdir(RUNS_DIR), reverse=True):
            if not fn.endswith("_generate.json"):
                continue
            try:
                with open(os.path.join(RUNS_DIR, fn), "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            plans.append({
                "id": fn[:-5],  # filename without .json
                "title": rec.get("sprint_label") or "Sprint Goals",
                "date": (rec.get("timestamp") or "")[:10],
                "people_count": rec.get("people_count", 0),
                "summary": rec.get("meeting_summary", ""),
            })
    return jsonify({"plans": plans})


@app.route("/api/plans/<plan_id>", methods=["GET"])
def api_plan_detail(plan_id):
    """Re-open a past plan: rebuild its per-person sections from the run log."""
    # Guard against path traversal — only a bare run-id basename is allowed.
    if "/" in plan_id or "\\" in plan_id or ".." in plan_id:
        return jsonify({"error": "Bad plan id."}), 400
    path = os.path.join(RUNS_DIR, plan_id + ".json")
    if not os.path.exists(path):
        return jsonify({"error": "Plan not found."}), 404
    try:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
    except (json.JSONDecodeError, OSError):
        return jsonify({"error": "Could not read plan."}), 500

    extraction = rec.get("extraction") or {}
    formatted = rec.get("formatted_doc") or ""
    people = extraction.get("people", [])
    sections = split_into_sections(formatted, people) if formatted else []
    sprint_label = rec.get("sprint_label") or "Sprint Goals"
    pi = (rec.get("process_improvement") or "").strip()
    header = f"Sprint goals ({sprint_label})"
    if pi:
        header += f"\n\nProcess Improvement for the Week: {pi}"
    return jsonify({
        "id": plan_id,
        "sprint_label": sprint_label,
        "header": header,
        "process_improvement": pi,
        "meeting_summary": rec.get("meeting_summary", ""),
        "sections": sections,
        "applied": rec.get("applied") or {},
        "date": (rec.get("timestamp") or "")[:10],
    })


@app.route("/api/lessons/add", methods=["POST"])
def api_lessons_add():
    """Add a preference manually (e.g. Shiv saying 'it's missing an instruction').
    person=None makes it a team-wide preference; a name scopes it to that member."""
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    person = (data.get("person") or "").strip() or None
    if not text:
        return jsonify({"error": "Preference text is empty."}), 400
    mem = load_memory()
    mem["rules"].append({
        "id": uuid.uuid4().hex[:8], "text": text, "person": person,
        "date": datetime.date.today().isoformat(),
    })
    save_memory(mem)
    return jsonify({"rules": mem["rules"], "example_count": len(mem.get("examples", []))})


@app.route("/api/lessons/update", methods=["POST"])
def api_lessons_update():
    """Edit an existing preference's text (e.g. 'this instruction is wrong')."""
    data = request.get_json(force=True, silent=True) or {}
    rule_id = data.get("id")
    text = (data.get("text") or "").strip()
    if not rule_id or not text:
        return jsonify({"error": "Need a rule id and non-empty text."}), 400
    mem = load_memory()
    found = False
    for r in mem["rules"]:
        if r.get("id") == rule_id:
            r["text"] = text
            r["date"] = datetime.date.today().isoformat()
            found = True
            break
    if not found:
        return jsonify({"error": "Preference not found."}), 404
    save_memory(mem)
    return jsonify({"rules": mem["rules"], "example_count": len(mem.get("examples", []))})


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
    store.init_db()  # create tables if DATABASE_URL is set (no-op otherwise)
    # debug enables auto-reload: edits to app.py / templates / the pipeline take
    # effect without manually restarting the server. Defaults on for local dev,
    # but MUST be off when the app is exposed via a public tunnel/deploy — the
    # Werkzeug debugger is a remote-code-execution vector. Set FLASK_DEBUG=0 then.
    # Port is configurable via PORT env var (macOS uses 5000 for AirPlay).
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=debug)

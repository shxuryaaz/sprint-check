# Loop 1 — Transcript → Sprint Goals Pipeline

Converts a Granola meeting transcript into Agilow's standard sprint goals
document via a two-phase LLM pipeline (extraction → formatting).

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
python loop1_pipeline.py transcript.txt --sprint-label "June 6th - June 13th" --team team.json --debug
```

Only `transcript_path` is required:

```bash
python loop1_pipeline.py transcript.txt
```

### Arguments

- `transcript_path` (required) — path to the transcript text file.
- `--sprint-label` (default `"Current Sprint"`) — e.g. `"June 6th - June 13th"`.
- `--team` (optional) — path to a JSON file containing a list of name strings,
  e.g. `["Shiv", "Antonio", "Shaurya", "Precious", "Cameron"]`.
- `--context` (optional) — path to a markdown/text file of project context
  (North Star deliverables, deadlines, key people). Injected into Phase 1 to add
  real detail to goals; it does **not** invent goals not in the transcript.
- `--debug` — print raw Phase 1 JSON and Phase 2 text to stderr.

The formatted document is printed to stdout and saved to
`sprint_goals_{sanitized_label}.txt` in the current directory.

## Web UI (paste-and-iterate)

A simple local web app to paste a Granola transcript, get the per-person sprint
goals document, and edit/approve/reject each person — your edits teach the next
run via a contextual learning loop.

```bash
pip install -r requirements.txt
OPENAI_API_KEY="sk-..." python app.py
# open http://127.0.0.1:5000
```

- Paste transcript → **Generate sprint goals** → one editable card per person.
- Edit or **Reject** sections, then **Approve & Learn**. Your per-person edits are
  distilled into reusable correction rules saved to `memory/learned_lessons.md`,
  which is injected into Phase 1 on every future generation.
- Optional static project context lives at `context/project_context.md` (created
  by you); it is auto-loaded if present.
- **Copy / Download .md** exports the final approved document.

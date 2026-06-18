#!/usr/bin/env python3
"""Loop 1 pipeline: Granola meeting transcript -> Agilow sprint goals document.

Two sequential OpenAI API calls:
  Phase 1 (extraction): transcript text -> structured JSON of per-person commitments
  Phase 2 (formatting): that JSON -> final plain-text sprint goals document

Usage:
    python loop1_pipeline.py transcript.txt \
        --sprint-label "June 6th - June 13th" \
        --team team.json \
        --debug
"""

import argparse
import json
import os
import re
import sys

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.2

# The OpenAI client is created via init_client() after the API key is validated.
_client = None


def init_client(api_key: "str | None" = None):
    """Initialize the module-level OpenAI client.

    Reads OPENAI_API_KEY from the environment if api_key is not supplied.
    Used by both the CLI (main) and the web server (app.py) so the client is
    set up exactly once per process before any API call.
    """
    global _client
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    _client = openai.OpenAI(api_key=key)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(
        (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        )
    ),
    reraise=True,
)
def _retrying_call(messages):
    """Make a chat completion call, retrying only on transient network errors."""
    return _client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=messages,
    )


def _call_api(messages):
    """Wrap the retrying call, translating auth/bad-request errors into clear messages."""
    try:
        return _retrying_call(messages)
    except openai.AuthenticationError as e:
        raise RuntimeError(
            f"OpenAI authentication failed. Check your OPENAI_API_KEY. Details: {e}"
        )
    except openai.BadRequestError as e:
        raise RuntimeError(
            f"OpenAI rejected the request as invalid (bad request). Details: {e}"
        )


def _strip_code_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences (```/```json/```text) if present."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        # Drop the opening fence line (which may carry a language tag).
        lines = lines[1:]
        # Drop the closing fence line if present.
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _build_phase1_system_prompt(
    team_members: "list[str] | None",
    context_text: "str | None" = None,
    lessons_text: "str | None" = None,
    examples_text: "str | None" = None,
) -> str:
    if team_members:
        team_section = (
            "The team is: "
            + ", ".join(team_members)
            + ". Include EVERY one of these people as an entry in the people "
            "array, in this order. This is a sprint planning meeting, so each "
            "member almost certainly took on real work — your PRIMARY job is to "
            "find and extract the concrete goals each person committed to or was "
            "assigned. Dig carefully through the whole transcript: planning "
            "meetings rarely contain crisp 'I will…' sentences, so treat work a "
            "person clearly agreed to do, was handed, or said they'd own as a "
            "goal even when phrased loosely. An empty goals array is a LAST "
            "RESORT, only when the transcript genuinely says nothing about what a "
            "listed person will do — do not default to empty, and never fabricate "
            "a goal to fill a gap. Someone NOT in this list should only be "
            "included if they clearly have a commitment."
        )
    else:
        team_section = "Identify team members by name from context in the transcript."

    if context_text and context_text.strip():
        context_section = (
            "\nPROJECT CONTEXT & LIVE SOURCES (authoritative — includes team roles, the "
            "Journey North Star, and live data pulled from the team's calendar, email, and "
            "Drive). Use this BOTH to enrich transcript goals AND to CREATE new goals:\n"
            "- CREATE goals directly from these sources, attributing each to the correct owner "
            "(use the team roles):\n"
            "  - A calendar event this week (meeting, demo, travel, networking) -> a 'prepare for / "
            "attend' goal for whoever owns it. Respect anyone marked unavailable.\n"
            "  - A North Star / SOW deliverable that is due, behind, or in progress this sprint -> a "
            "goal for its owner (Journey items belong to Shiv/Antonio only).\n"
            "  - A clear commitment, deadline, or follow-up found in email -> a goal for the relevant "
            "person.\n"
            "  - An unfinished item carried over from a previous sprint -> carry it forward as a goal.\n"
            "- Use the precise names/dates/details from the context (not vague paraphrases).\n"
            "- The ONLY limit: do not invent goals with no basis in EITHER the transcript OR these "
            "sources. Every goal must trace to the transcript or to an authoritative source here.\n\n"
            "----- BEGIN PROJECT CONTEXT & SOURCES -----\n"
            f"{context_text.strip()}\n"
            "----- END PROJECT CONTEXT & SOURCES -----\n"
        )
    else:
        context_section = ""

    if lessons_text and lessons_text.strip():
        lessons_section = (
            "\nLEARNED CORRECTIONS (distilled from past human edits to earlier "
            "drafts — apply these to avoid repeating the same mistakes):\n"
            "- These are behavioral rules about HOW to extract, not project facts. "
            "Follow them unless the transcript clearly contradicts them.\n\n"
            "----- BEGIN LEARNED CORRECTIONS -----\n"
            f"{lessons_text.strip()}\n"
            "----- END LEARNED CORRECTIONS -----\n"
        )
    else:
        lessons_section = ""

    if examples_text and examples_text.strip():
        examples_section = (
            "\nPAST CORRECTION EXAMPLES (real before→after edits a human made to "
            "earlier drafts for similar people/work — imitate the style and "
            "judgement shown in the AFTER versions):\n\n"
            "----- BEGIN EXAMPLES -----\n"
            f"{examples_text.strip()}\n"
            "----- END EXAMPLES -----\n"
        )
    else:
        examples_section = ""

    schema_example = """{
  "meeting_summary": "string, 2-3 sentences",
  "process_improvement": "string — ONE team-wide process improvement for the whole team this sprint",
  "people": [
    {
      "name": "string",
      "kaizen": "string — this person's single highest-priority deliverable (outcome kaizen)",
      "goals": [
        {
          "title": "string, under 15 words",
          "description": "string, one sentence",
          "points": 0.0,
          "points_is_estimated": false,
          "subtasks": ["string", "..."],
          "success_criteria": "string",
          "dependencies": [
            {"description": "string", "owner": "string or null"}
          ],
          "risks": [
            {"description": "string", "mitigation": "string or null"}
          ]
        }
      ]
    }
  ]
}"""

    return f"""You are an expert project management assistant analyzing a sprint planning meeting transcript for a technical consulting team called Agilow.

TRANSCRIPT HANDLING NOTES:
- The transcript may contain informal speech, filler words, interruptions, and crosstalk between speakers labeled 'Me:' and 'Them:'.
- The transcript may contain text in languages other than English, including code-switching within a single sentence (e.g. mixing English and Hindi). Extract the meaning regardless of language — do not skip or ignore non-English portions.
- Speaker labels 'Me:' and 'Them:' do not reliably map to specific named individuals — infer actual names from context (people referring to each other by name, self-introductions, etc.).

TEAM MEMBER HANDLING:
{team_section}
{context_section}{lessons_section}{examples_section}
EXTRACTION TASK:
Extract per-person sprint commitments and output a JSON object matching EXACTLY this schema:

{schema_example}

FIELD-BY-FIELD RULES:
- meeting_summary: 2-3 sentences, plain prose, what the meeting covered overall.
- process_improvement: ONE team-wide process improvement chosen for the whole team this sprint — a single shared commitment that applies to everyone (e.g. "Everyone reads each other's goals by Monday night and leaves one comment per person"). Same value regardless of person. Always provide one.
- name: the person's actual name as used in the transcript (e.g. "Shiv" not "Them").
- kaizen: this person's OUTCOME KAIZEN — their single highest-priority, non-negotiable deliverable for the sprint, written as one short line. It must correspond to that person's highest-point goal. Always provide one.
- goals: one object per distinct commitment or work item. A goal must be something the person agreed to do or was assigned. Enumerate at FINE granularity — a separate goal per distinct deliverable, never collapse several into one.
  GOAL COVERAGE: a person's goals should span the categories that apply to them, not just one area. The four categories are: (1) current paying-client work (e.g. Journey Robotics delivery), (2) company-building (operations, legal, finance), (3) outreach & networking events, (4) personal goals. Cover every category the transcript/context gives signal for.
  HOW TO REACH FULL COVERAGE: build each person's goal list by COMBINING all available sources, not just the transcript — (a) commitments in the transcript, (b) calendar events this week, (c) North Star / SOW deliverables due or behind this sprint, (d) commitments/follow-ups in email, (e) unfinished carry-overs. Each of these contributes goals. Real sprints have 6-12 goals per active person; if you produced only 1-3 for an active person, you UNDER-EXTRACTED — go back and add the goals implied by the calendar, North Star, email, and carry-overs before finishing.
- title: imperative phrasing, under 15 words, e.g. "Deliver Program Schedule for Pilot Phase 1 to JR".
- description: ONE sentence describing what the goal involves and why, grounded in the transcript/context. Always provide a description (never null).
- points: numeric (0.5, 1, 1.5, 2, ...), reflecting effort AND impact: ~2 for a major deliverable, ~0.5-1 for a supporting task. The person's outcome kaizen goal should carry the HIGHEST points among their goals. Use the value stated in the transcript if given; otherwise estimate and set points_is_estimated: true.
- subtasks: array of short imperative strings — the concrete steps. ALWAYS provide at least 2 per goal; infer reasonable steps if not spelled out. Never null/empty.
- success_criteria: ONE sentence that is BINARY and MEASURABLE (SMART: specific, measurable, attainable, relevant, time-bound) — it must be easy to say yes/no whether it was achieved (e.g. "Program Schedule is sent and acknowledged by Keith and Reeg by Friday"). Always provide one.
- dependencies: array of objects. ALWAYS provide at least one per goal. owner is the person/entity depended on (or null if unclear). Infer the most plausible dependency if none was stated.
- risks: array of objects, each with description + mitigation. ALWAYS provide at least one per goal. Name REAL challenges — knowledge gaps, scope creep, external dependencies — not "everything will be fine". The mitigation must be a concrete next step, not a vague hope. Never leave mitigation null.

OWNERSHIP AND SALIENCE RULES:
- Assign each goal to the person who actually OWNS it. When work is explicitly handed off or divided (e.g. "X will own this part, Y will support"), attribute the goal to the owner and record the supporter as a dependency — do not give the owner's goal to the wrong person.
- Weight commitments made or confirmed near the END of the meeting (the concrete planning/assignment portion) over offhand remarks made early in the conversation. An early casual aside ("this week I might look at X") is NOT a goal unless it is reaffirmed as a real commitment later.
- If two people discuss the same work, only the person assigned to do it gets the goal.

FABRICATION RULES:
- Do NOT invent GOALS or PEOPLE that have no basis in the transcript — the goals themselves and who owns them must come from what was actually discussed or assigned.
- HOWEVER, for each real goal you DO populate its supporting detail fields fully (subtasks, success_criteria, dependencies, risks+mitigation) and each person's kaizen — inferring reasonable, plausible values when they were not explicitly discussed. These detail fields must always be filled, never empty/null, even when that means making a sensible inference.
- A person who is NOT in the provided team list and has no concrete commitment should NOT be included. (Team members in the list are always included per the TEAM MEMBER HANDLING rule above — with an empty goals array if they have no commitment.) Either way, NEVER manufacture a goal (e.g. "take a break", "manage workload", "stay aligned") just to fill an empty entry.
- Do NOT convert something that merely happened or was reported (a past event, a status update, someone being absent or busy) into a forward-looking goal.
- A commitment does not require the exact words 'I will'. It includes explicit statements ('I'll do…', 'my goal is…'), tasks someone is assigned or hands off, and work a person clearly agreed to take on or own — even if phrased loosely or across several messy lines. The bar is "did they agree to do this work?", not "did they say a perfect commitment sentence?". Only exclude something when it is genuinely just a topic discussed with no one taking it on.

OUTPUT FORMAT RULES:
- Output ONLY the JSON object described above.
- Do NOT wrap the output in markdown code fences (no ```json).
- Do NOT include any explanatory text before or after the JSON.
- The output must be valid JSON parseable by Python's json.loads()."""


def _build_phase2_system_prompt() -> str:
    worked_example = """Sprint goals (May 23rd - May 30th)

Process Improvement for the Week: Everyone reads each other's goals by Monday night and leaves at least one comment per person.

Shiv: __ / 7
Kaizen: Serve Journey Robotics at a 6 out of 5 level

1. Journey Robotics: Unified view (2 points)
Create a unified view of goals, updates, and issues for the Journey Robotics team.
- Consolidate the three current sources (Word doc, async standups, GitHub issues) into one view with GitHub issues at the center
- Conduct a successful and short sprint planning meeting on Tuesday
Success criteria: one unified view of goals, updates, and issues is provided to the Journey Robotics team by next Friday's sprint planning

2. Journey Robotics: Engineering timeline (1 point)
Set up work streams and epics with a timeline that increases confidence in the schedule.
- Define the various work streams involved for Journey Robotics engineering
Success criteria: work streams and epics are set up, and the timeline increases team confidence in the schedule

3. Journey Robotics: Program Gantt view (1 point)
Create a Gantt chart for the pre-pilot and phase one.
- Review with Reeg
- Deliver to Nicole (IAG and HAL teams) by end of week
Success criteria: Gantt view reviewed by Reeg and delivered to Nicole

4. Support Shaurya: Package Agilow service (1 point)
Define the consulting service offering so Shaurya has materials for outreach.
- Define what we provide, deliverables, and the pitch
- Create materials Shaurya can use for outreach
Success criteria: service packaging documented and ready for outreach

5. VentureBridge Demo Day (1 point)
Decide on attendance for June 3rd in NYC and handle logistics.
- Respond to Stephen and Namrata about attendance
- If attending, book flights and hotel
Success criteria: decision made, communicated, and logistics booked if attending

---

Shaurya: __ / 5
Kaizen: Build the engine that finds us our next clients

1. Outreach research and plan (1 point)
Research how comparable consulting companies acquire clients, then present a plan.
- Research client acquisition, pricing, and positioning of comparable firms
- Build on Cameron's GTM framework rather than duplicating it
- Present a rough outreach plan to Shiv, Antonio, and the team by Wednesday
Success criteria: research plan presented by Wednesday, team aligned on next steps

2. Package the Agilow consulting service (2 points)
Define the consulting offering and build the pitch deck for outreach.
- Define what we provide, deliverables, and the pitch
- Build the Agilow pitch deck for outreach to robotics companies
Success criteria: service packaging and pitch deck drafted and ready for outreach

3. Build the qualified lead list (1 point)
Manually build a list of early-stage robotics companies as outreach targets.
- Filter by robotics focus, seed-Series B ($2-8M raised), 7-10 person engineering teams, active PM hiring
- Target 20-40 qualified leads
Success criteria: filtered list of 20-40 robotics company leads ready for outreach

4. Lead-generation automation - stretch (1 point)
Explore an automation to scrape LinkedIn/Indeed for robotics companies hiring a PM.
- Lower priority: pick up only after goals 1-3 are on track
Success criteria: progress made on a working scraper, even if not fully complete"""

    return f"""You are formatting structured sprint goal data into Agilow's standard sprint goals document format. Follow the format below EXACTLY, including spacing, numbering, and section ordering.

TARGET FORMAT — here is a complete realistic worked example showing exactly how the output must look:

{worked_example}

FORMATTING RULES:
1. First line is always exactly `Sprint goals ({{sprint_label}})` followed by a blank line.
1b. If the input JSON has a non-empty `process_improvement`, print `Process Improvement for the Week: {{process_improvement}}` on the next line, followed by a blank line. This is the team-wide process kaizen and appears ONCE, before any person.
2. For each person in the people array, in the order they appear in the input JSON:
   a. Print a person header line: `{{name}}: __ / {{total}}` where total is the sum of all that person's goal points. Format total as an integer if it has no fractional part (e.g. 7 not 7.0), otherwise as the decimal (e.g. 5.5). The `__` is a blank for the person to self-grade at the end of the sprint — always print it literally as two underscores.
   b. If kaizen is not null, print `Kaizen: {{kaizen}}` on the next line. If kaizen is null, skip this line entirely (do not print "Kaizen: null" or an empty Kaizen line).
   c. Print a blank line.
   d. For each goal, numbered starting at 1 (restart numbering for each person):
      - Print `{{number}}. {{title}} ({{points}} point{{s}})` — use "point" (singular) if points == 1, otherwise "points". NEVER append "(estimated)" or any other suffix to the points — just the plain `(N points)`.
      - On the next line, print the goal `description` exactly as given (one sentence, no label, no bullet). The description always appears, directly under the goal heading and before any subtasks.
      - For each subtask, print `- {{subtask}}` on its own line. If subtasks is an empty array, skip this block entirely.
      - If success_criteria is not null, print `Success criteria: {{success_criteria}}`. If null, skip this line.
      - If dependencies is non-empty, print `Dependencies:` then for each dependency print `- {{description}} (Owner: {{owner}})`, using `(Owner: unassigned)` if owner is null. If dependencies is empty, skip this entire block.
      - If risks is non-empty, print `Risks:` then for each risk print `- {{description}}` followed by `  Mitigation: {{mitigation}}` (two-space indent) on the next line, using `Not yet defined` if mitigation is null. If risks is empty, skip this entire block.
      - Print a blank line after each goal.
   e. Do NOT print any total/grade line at the bottom of a person's section — the grade lives only in the header line from step (a).
   f. Separate people with a line containing only `---` (with a blank line before and after it) — UNLESS this is the last person, in which case omit the `---`.
3. Output ONLY this formatted text. No markdown code fences, no commentary before or after.

If the input JSON for a person has an empty goals array, still print their header line `{{name}}: __ / 0` and their Kaizen if present, then move to the next person."""


def extract_goals(
    transcript_text: str,
    team_members: "list[str] | None",
    context_text: "str | None" = None,
    lessons_text: "str | None" = None,
    examples_text: "str | None" = None,
    debug: bool = False,
) -> dict:
    """
    Calls the OpenAI API to extract per-person sprint goals from a transcript.

    Returns a dict matching the schema in the Phase 1 system prompt.
    Raises RuntimeError if the response cannot be parsed as valid JSON
    after one retry, with the raw model output included in the error message.
    """
    system_prompt = _build_phase1_system_prompt(
        team_members, context_text, lessons_text, examples_text
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "TRANSCRIPT:\n\n" + transcript_text},
    ]

    response = _call_api(messages)
    raw_output = response.choices[0].message.content.strip()
    usage = response.usage
    cleaned = _strip_code_fences(raw_output)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        messages.append({"role": "assistant", "content": raw_output})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. Return ONLY the JSON "
                    "object described in the system prompt, with no markdown formatting, "
                    "no code fences, and no explanatory text."
                ),
            }
        )
        response = _call_api(messages)
        raw_output = response.choices[0].message.content.strip()
        usage = response.usage
        cleaned = _strip_code_fences(raw_output)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Phase 1 extraction failed to produce valid JSON after retry. "
                f"Raw output:\n{raw_output}"
            )

    if not isinstance(result.get("meeting_summary"), str) or not isinstance(
        result.get("people"), list
    ):
        raise RuntimeError(
            f"Phase 1 output missing required keys or wrong types. Got: {result}"
        )

    if debug:
        print("--- PHASE 1 RAW OUTPUT ---", file=sys.stderr)
        print(json.dumps(result, indent=2), file=sys.stderr)

    print(
        f"Phase 1: extracted goals for {len(result['people'])} people",
        file=sys.stderr,
    )
    print(result["meeting_summary"], file=sys.stderr)
    print(
        f"Phase 1 token usage: prompt={usage.prompt_tokens}, "
        f"completion={usage.completion_tokens}",
        file=sys.stderr,
    )

    return result


def extract_goals_from_sources(
    context_text: str, team_members: "list[str] | None", debug: bool = False
) -> dict:
    """Second extraction pass: produce goals PURELY from the project context and
    live sources (North Star, calendar, email, Drive, carry-overs) — no transcript.
    A dedicated pass reliably surfaces source goals that the transcript pass under-
    extracts. Returns {"people": [{"name", "goals": [...]}]} (no kaizen/summary)."""
    if not context_text or not context_text.strip():
        return {"people": []}
    team = ", ".join(team_members) if team_members else "the team"
    system_prompt = (
        "You extract sprint goals PURELY from the project context and live sources below "
        "(team roles, the Journey North Star, and live calendar/email/Drive data). There is "
        "NO meeting transcript — surface the goals these sources imply.\n\n"
        f"TEAM: {team}. Attribute every goal to the correct owner using the team roles. "
        "Journey / North-Star items belong to Shiv and Antonio only.\n\n"
        "CREATE one goal, for the right owner, from each of these:\n"
        "- each calendar event this week (meeting, demo, travel, networking) -> a 'prepare for / "
        "attend' goal; respect anyone marked unavailable\n"
        "- each North Star / SOW deliverable due, behind, or in progress this sprint\n"
        "- each clear commitment, deadline, or follow-up in email\n"
        "- each unfinished carry-over from a previous sprint\n"
        "Skip pure noise (newsletters, marketing email, verification codes, automated "
        "notifications). Do NOT invent goals with no basis in the sources.\n\n"
        'OUTPUT ONLY this JSON: {"people":[{"name":str,"goals":[GOAL,...]}]} where GOAL is '
        '{"title":str (<15 words, imperative),"description":str (one sentence),"points":number '
        '(2 major, 0.5-1 supporting),"points_is_estimated":true,"subtasks":[>=2 strings],'
        '"success_criteria":str (binary, measurable, time-bound),"dependencies":'
        '[{"description":str,"owner":str or null}],"risks":[{"description":str,"mitigation":str}]}. '
        "Populate every field. No markdown, no code fences — valid JSON for json.loads()."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "PROJECT CONTEXT & LIVE SOURCES:\n\n" + context_text},
    ]
    response = _call_api(messages)
    cleaned = _strip_code_fences(response.choices[0].message.content.strip())
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        messages.append({"role": "assistant", "content": cleaned})
        messages.append({"role": "user", "content": (
            "Your previous response was not valid JSON. Return ONLY the JSON object, no other text.")})
        response = _call_api(messages)
        cleaned = _strip_code_fences(response.choices[0].message.content.strip())
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            raise RuntimeError(f"Sources extraction failed to produce valid JSON. Raw:\n{cleaned}")
    if not isinstance(result.get("people"), list):
        return {"people": []}
    usage = response.usage
    print(
        f"Sources pass: goals for {len(result['people'])} people "
        f"(prompt={usage.prompt_tokens}, completion={usage.completion_tokens})",
        file=sys.stderr,
    )
    return result


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def _titles_similar(a: str, b: str) -> bool:
    a, b = _norm_title(a), _norm_title(b)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    aw, bw = set(a.split()), set(b.split())
    if not aw or not bw:
        return False
    return len(aw & bw) / min(len(aw), len(bw)) >= 0.6


def merge_extractions(primary: dict, secondary: dict) -> dict:
    """Merge the sources-pass goals into the transcript-pass extraction, per person,
    skipping near-duplicate goal titles. Keeps primary's kaizen / process_improvement."""
    people = primary.setdefault("people", [])
    by_name = {p.get("name", "").lower(): p for p in people}
    for sp in secondary.get("people", []):
        name = sp.get("name", "")
        key = name.lower()
        sgoals = sp.get("goals", []) or []
        if key in by_name:
            existing = by_name[key].setdefault("goals", [])
            for g in sgoals:
                if not any(_titles_similar(g.get("title"), e.get("title")) for e in existing):
                    existing.append(g)
        elif name:
            people.append({"name": name, "kaizen": sp.get("kaizen"), "goals": sgoals})
            by_name[key] = people[-1]
    return primary


def parse_sprint_draft(draft_text: str, debug: bool = False) -> dict:
    """Layer-on-Granola mode: take a sprint-goals draft that's already roughly in
    our format (e.g. produced by Granola's template) and structure it into our JSON
    FAITHFULLY — don't add, drop, merge, or invent goals. The off-transcript goals
    and learned fixes are added afterward by the sources pass + merge."""
    if not draft_text or not draft_text.strip():
        return {"meeting_summary": "", "process_improvement": "", "people": []}
    system_prompt = (
        "You convert a sprint-goals draft (already roughly in Agilow's format, e.g. "
        "produced by Granola) into structured JSON. Preserve content FAITHFULLY — do "
        "NOT add, drop, merge, or invent goals; just structure exactly what is there.\n\n"
        'Output ONLY this JSON: {"process_improvement": str, "people": [{"name": str, '
        '"kaizen": str, "goals": [{"title": str, "description": str, "points": number, '
        '"points_is_estimated": false, "subtasks": [str], "success_criteria": str, '
        '"dependencies": [{"description": str, "owner": str or null}], "risks": '
        '[{"description": str, "mitigation": str or null}]}]}]}. '
        "process_improvement = the team-wide 'Process Improvement for the Week' line if "
        "present, else empty string. Capture each person's kaizen line and every goal "
        "with its points, description, subtasks, success criteria, dependencies (with "
        "owner) and risks (with mitigation) exactly as written. No markdown, no fences."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "SPRINT GOALS DRAFT:\n\n" + draft_text},
    ]
    response = _call_api(messages)
    cleaned = _strip_code_fences(response.choices[0].message.content.strip())
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        messages.append({"role": "assistant", "content": cleaned})
        messages.append({"role": "user", "content": (
            "Your previous response was not valid JSON. Return ONLY the JSON object, no other text.")})
        response = _call_api(messages)
        cleaned = _strip_code_fences(response.choices[0].message.content.strip())
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            raise RuntimeError(f"Draft parse failed to produce valid JSON. Raw:\n{cleaned}")
    result.setdefault("meeting_summary", "")
    result.setdefault("process_improvement", "")
    if not isinstance(result.get("people"), list):
        result["people"] = []
    usage = response.usage
    print(
        f"Draft parse: {len(result['people'])} people "
        f"(prompt={usage.prompt_tokens}, completion={usage.completion_tokens})",
        file=sys.stderr,
    )
    return result


def format_sprint_doc(
    extraction: dict, sprint_label: str, debug: bool = False
) -> str:
    """
    Calls the OpenAI API to format extracted goal data into Agilow's
    standard sprint goals document text format.

    Returns the formatted document as a plain string.
    """
    system_prompt = _build_phase2_system_prompt()
    # Compact JSON (no indent/whitespace) — ~30% fewer tokens than pretty-printed
    # for the same data, with no loss of information the model needs.
    user_message = (
        f"SPRINT LABEL: {sprint_label}\n\n"
        f"EXTRACTED DATA (JSON):\n{json.dumps(extraction, separators=(',', ':'))}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = _call_api(messages)
    raw_output = response.choices[0].message.content.strip()
    cleaned = _strip_code_fences(raw_output)

    if debug:
        print("--- PHASE 2 RAW OUTPUT ---", file=sys.stderr)
        print(cleaned, file=sys.stderr)

    usage = response.usage
    print(
        f"Phase 2 token usage: prompt={usage.prompt_tokens}, "
        f"completion={usage.completion_tokens}",
        file=sys.stderr,
    )

    return cleaned


def distill_feedback(
    feedback_items: "list[dict]", meeting_summary: "str | None" = None
) -> "list[str]":
    """Turn per-person human edits into durable, reusable correction rules.

    feedback_items: list of dicts with keys:
      - name: the person whose section this is
      - original: the section text the pipeline generated
      - final: the section text the human approved (may equal original)
      - status: "approved" | "edited" | "rejected"
      - reason: optional free-text "why" for the change

    Only "edited" and "rejected" items carry a learning signal. Makes one LLM
    call that compares draft vs corrected text and returns a list of correction
    rules as dicts {"text": str, "person": str | None} — person is set only when a
    rule is an individual's preference, else None (applies to everyone). Returns an
    empty list if there is no signal to learn from.
    """
    signal_items = [
        it for it in feedback_items if it.get("status") in ("edited", "rejected")
    ]
    if not signal_items:
        return []

    blocks = []
    for it in signal_items:
        reason = (it.get("reason") or "").strip()
        reason_line = f"\nREASON THE HUMAN GAVE: {reason}" if reason else ""
        if it.get("status") == "rejected":
            blocks.append(
                f"### {it.get('name', 'Unknown')} — REJECTED ENTIRELY\n"
                f"The human rejected this whole section as not a valid set of "
                f"sprint goals:\n{it.get('original', '').strip()}{reason_line}"
            )
        else:
            blocks.append(
                f"### {it.get('name', 'Unknown')} — EDITED\n"
                f"DRAFT (what the pipeline produced):\n{it.get('original', '').strip()}\n\n"
                f"CORRECTED (what the human actually wanted):\n{it.get('final', '').strip()}"
                f"{reason_line}"
            )
    diff_text = "\n\n".join(blocks)

    summary_line = (
        f"\nMeeting summary for context: {meeting_summary}\n" if meeting_summary else ""
    )

    system_prompt = (
        "You are improving an automated sprint-goal extraction pipeline for a "
        "consulting team called Agilow. You are given the pipeline's DRAFT sprint "
        "goals and the human's CORRECTED version (or a flag that the human rejected "
        "a section entirely). Your job is to infer GENERAL, REUSABLE correction "
        "rules that, if followed next time, would make the pipeline's output match "
        "the human's intent.\n\n"
        "RULES FOR YOUR OUTPUT:\n"
        "- Output ONLY a JSON array of objects, each {\"text\": <short rule>, "
        "\"person\": <a single name or null>}. No prose, no code fences.\n"
        "- Set \"person\" to an individual's name ONLY when the rule is about THAT "
        "person's individual style/preference (e.g. 'Shiv prefers fewer, larger "
        "goals'). If the rule should apply to everyone, set \"person\": null.\n"
        "- Each rule must be GENERAL and reusable across future meetings — not a "
        "fact about this specific week (e.g. write 'Do not turn a person's absence "
        "or vacation into a goal', NOT 'Antonio went on vacation').\n"
        "- CRITICAL: many edits are ONE-OFF judgments with no transferable pattern "
        "— e.g. changing a single goal's points from 1.5 to 2, or rewording one "
        "title, with no stated reason. Do NOT invent a rule from these; they are "
        "kept separately as examples. Only emit a rule when a pattern would "
        "plausibly recur. It is correct and expected to return [] when the edits "
        "are just one-off corrections.\n"
        "- If the human gave a REASON for a change, you MAY turn it into a rule even "
        "for sizing/ownership when the reason generalizes (e.g. reason 'regulation "
        "deliverables always take longer' -> rule 'Size regulation/compliance "
        "deliverables at 2+ points'). No reason behind a pure number change -> no "
        "rule.\n"
        "- Focus on recurring failure patterns: which work counts as a goal, who "
        "OWNS each goal, goal granularity (too few/too many), goal titles and "
        "wording, point sizing, and kaizen phrasing.\n"
        "- IMPORTANT POLICY: the pipeline is REQUIRED to always populate every "
        "field (kaizen, subtasks, success_criteria, dependencies, risks with "
        "mitigation) for each goal, inferring sensible values when the transcript "
        "is silent. Therefore do NOT produce any rule that says to omit, skip, or "
        "avoid inferring these fields — that policy is fixed. Rules may only change "
        "HOW those fields are phrased or what content goes in them, never whether "
        "they appear.\n"
        "- Prefer 1-6 crisp rules. If there is nothing genuinely generalizable, "
        "return an empty array []."
    )
    user_message = (
        f"{summary_line}\nHere are the human corrections to learn from:\n\n{diff_text}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = _call_api(messages)
    raw_output = response.choices[0].message.content.strip()
    cleaned = _strip_code_fences(raw_output)
    try:
        rules = json.loads(cleaned)
    except json.JSONDecodeError:
        messages.append({"role": "assistant", "content": raw_output})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. Return ONLY a JSON "
                    "array of short rule strings, with no other text."
                ),
            }
        )
        response = _call_api(messages)
        cleaned = _strip_code_fences(response.choices[0].message.content.strip())
        try:
            rules = json.loads(cleaned)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Feedback distillation failed to produce valid JSON. "
                f"Raw output:\n{cleaned}"
            )

    if not isinstance(rules, list):
        return []
    out = []
    for r in rules:
        if isinstance(r, dict):
            text = str(r.get("text", "")).strip()
            person = r.get("person")
            person = str(person).strip() if person else None
        else:
            text, person = str(r).strip(), None
        if text:
            out.append({"text": text, "person": person})
    return out


def _parse_json_array(messages: "list[dict]") -> "list[str]":
    """Call the API expecting a JSON array of strings, with one retry. Returns a
    cleaned list of non-empty strings. Raises RuntimeError if unparseable twice."""
    response = _call_api(messages)
    cleaned = _strip_code_fences(response.choices[0].message.content.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        retry = messages + [
            {"role": "assistant", "content": cleaned},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. Return ONLY a JSON "
                    "array of strings, with no other text."
                ),
            },
        ]
        response = _call_api(retry)
        cleaned = _strip_code_fences(response.choices[0].message.content.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Expected a JSON array but could not parse it. Raw:\n{cleaned}"
            )
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def consolidate_rules(
    existing_rules: "list[str]", new_rules: "list[str]", max_rules: int = 25
) -> "list[str]":
    """Merge new correction rules into the existing set: dedupe near-duplicates,
    combine overlapping rules, resolve contradictions (newer guidance wins), and
    cap the total. Returns the cleaned full rule list. Falls back to a simple
    case-insensitive union if the LLM consolidation step fails.
    """
    existing = [str(r).strip() for r in (existing_rules or []) if str(r).strip()]
    new = [str(r).strip() for r in (new_rules or []) if str(r).strip()]
    if not new:
        return existing[:max_rules]
    if not existing:
        # Nothing to merge against; just dedupe and cap.
        out, seen = [], set()
        for r in new:
            if r.lower() not in seen:
                seen.add(r.lower())
                out.append(r)
        return out[:max_rules]

    system_prompt = (
        "You maintain a concise, non-redundant list of GENERAL correction rules "
        "for an automated sprint-goal extraction pipeline. Merge the NEW candidate "
        "rules into the EXISTING list. Remove exact and near-duplicate rules, "
        "combine overlapping rules into one clear rule, and when two rules "
        "genuinely conflict keep the one that reflects the newer guidance (the "
        "candidates). Every rule must stay general and reusable. Never exceed "
        f"{max_rules} rules — if you would, merge or drop the least important. "
        "Output ONLY a JSON array of the final rule strings, no other text."
    )
    user_message = (
        "EXISTING RULES:\n"
        + json.dumps(existing, indent=2)
        + "\n\nNEW CANDIDATE RULES:\n"
        + json.dumps(new, indent=2)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    try:
        merged = _parse_json_array(messages)
    except RuntimeError:
        # Fallback: naive union that still avoids exact-duplicate spam.
        merged = existing + [
            r for r in new if r.lower() not in {e.lower() for e in existing}
        ]

    out, seen = [], set()
    for r in merged:
        r = str(r).strip()
        if r and r.lower() not in seen:
            seen.add(r.lower())
            out.append(r)
    return out[:max_rules]


def _sanitize_label(label: str) -> str:
    """Lowercase, replace non [a-z0-9] with _, collapse repeats, strip ends."""
    s = label.lower()
    s = re.sub(r"[^a-z0-9]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Convert a meeting transcript into an Agilow sprint goals document."
    )
    parser.add_argument(
        "transcript_path",
        type=str,
        help="Path to the transcript text file.",
    )
    parser.add_argument(
        "--sprint-label",
        type=str,
        default="Current Sprint",
        help='Label for the sprint, e.g. "June 6th - June 13th".',
    )
    parser.add_argument(
        "--team",
        type=str,
        default=None,
        help="Path to a JSON file containing a list of team member name strings.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Path to a markdown/text file of project context (North Star "
        "deliverables, deadlines, key people) injected into Phase 1 extraction.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw Phase 1 JSON output to stderr before Phase 2 runs.",
    )
    args = parser.parse_args()

    try:
        # --- Validation (no API calls) ---
        if not os.path.exists(args.transcript_path):
            print(
                f"ERROR: Transcript file not found: {args.transcript_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(args.transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        stripped_len = len(transcript_text.strip())
        if stripped_len < 100:
            print(
                f"WARNING: Transcript is very short ({stripped_len} chars). "
                f"Output quality may be poor.",
                file=sys.stderr,
            )

        team_members = None
        if args.team is not None:
            try:
                with open(args.team, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(
                    "ERROR: --team file must be a JSON array of strings",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not isinstance(loaded, list) or not all(
                isinstance(x, str) for x in loaded
            ):
                print(
                    "ERROR: --team file must be a JSON array of strings",
                    file=sys.stderr,
                )
                sys.exit(1)
            team_members = loaded

        context_text = None
        if args.context is not None:
            if not os.path.exists(args.context):
                print(
                    f"ERROR: Context file not found: {args.context}",
                    file=sys.stderr,
                )
                sys.exit(1)
            with open(args.context, "r", encoding="utf-8") as f:
                context_text = f.read()
            print(
                f"Using project context from {args.context} "
                f"({len(context_text.strip())} chars)",
                file=sys.stderr,
            )

        # --- Set up the OpenAI client now that the key is known ---
        init_client(api_key)

        # --- Phase 1: extraction ---
        try:
            extraction = extract_goals(
                transcript_text,
                team_members,
                context_text=context_text,
                debug=args.debug,
            )
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        # --- Phase 2: formatting ---
        formatted = format_sprint_doc(
            extraction, args.sprint_label, debug=args.debug
        )

        # --- Output ---
        print("\n===== SPRINT GOALS DOCUMENT =====\n")
        print(formatted)
        print("\n===== END =====\n")

        sanitized = _sanitize_label(args.sprint_label)
        output_path = f"sprint_goals_{sanitized}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted)
        print(f"Saved to: {output_path}")

    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: Unexpected failure: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

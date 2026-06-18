#!/usr/bin/env python3
"""Live data connectors for Loop 1 — Google Calendar, Gmail, Drive (read-only).

OAuth uses the Desktop-app credentials in credentials.json; the token is cached
in token.json after the first consent. Each connector returns plain dicts (for
the UI to display exactly what's being pulled) and can be written into
context/sources/ so the sprint pipeline ingests it on the next Generate.

Nothing here is destructive — all scopes are read-only.
"""

import datetime
import json
import os
import re

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")
SOURCES_DIR = os.path.join(BASE_DIR, "context", "sources")
DRIVE_SOURCE_PATH = os.path.join(BASE_DIR, "drive_source.json")  # which Drive to read
TEAM_CAL_PATH = os.path.join(BASE_DIR, "team_calendars.json")    # whose calendars to read
DEFAULT_TEAM_CALENDARS = [
    "shaurya@agilow.ai", "shiv@agilow.ai", "antonio@agilow.ai", "cameron@agilow.ai",
]

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    # drive.readonly (not metadata-only) so we can list Shared Drives and, later,
    # read file contents — all still read-only.
    "https://www.googleapis.com/auth/drive.readonly",
]


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def _load_creds() -> "Credentials | None":
    """Return cached, valid credentials (refreshing if needed), or None."""
    if not os.path.exists(TOKEN_PATH):
        return None
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    return creds if creds and creds.valid else None


def is_connected() -> bool:
    try:
        return _load_creds() is not None
    except Exception:  # noqa: BLE001 - any token problem = not connected
        return False


def authorize() -> bool:
    """Run the one-time OAuth consent (opens a browser) and cache the token.
    Returns True on success. Raises if credentials.json is missing."""
    if not os.path.exists(CREDENTIALS_PATH):
        raise RuntimeError("credentials.json not found in the project folder.")
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    creds = flow.run_local_server(port=0)  # opens the browser for consent
    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    return True


def disconnect() -> None:
    if os.path.exists(TOKEN_PATH):
        os.remove(TOKEN_PATH)


# --------------------------------------------------------------------------- #
# Fetchers — each returns a list of plain dicts (what the UI shows)
# --------------------------------------------------------------------------- #
def get_team_calendars() -> list:
    """The calendar emails to read (the user's + teammates'). Editable in the UI."""
    if os.path.exists(TEAM_CAL_PATH):
        try:
            with open(TEAM_CAL_PATH, "r", encoding="utf-8") as f:
                cals = json.load(f)
            if isinstance(cals, list) and cals:
                return [str(c).strip() for c in cals if str(c).strip()]
        except (json.JSONDecodeError, OSError):
            pass
    return list(DEFAULT_TEAM_CALENDARS)


def set_team_calendars(emails: list) -> list:
    cleaned = [str(e).strip() for e in (emails or []) if str(e).strip()]
    cleaned = cleaned or list(DEFAULT_TEAM_CALENDARS)
    with open(TEAM_CAL_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f)
    return cleaned


def _short_err(ex: Exception) -> str:
    s = str(ex)
    if "404" in s or "notFound" in s:
        return "not shared / not visible to you"
    if "403" in s or "forbidden" in s.lower():
        return "no access"
    return s[:120]


def fetch_calendar_detailed(days: int = 7, per_calendar: int = 10) -> dict:
    """Read each team member's calendar (the user's login must have org-wide or
    shared visibility). Returns {'events': [...with 'who'...], 'calendars': [status]}."""
    creds = _load_creds()
    if not creds:
        return {"events": [], "calendars": []}
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    now = datetime.datetime.utcnow()
    time_min = now.isoformat() + "Z"
    time_max = (now + datetime.timedelta(days=days)).isoformat() + "Z"

    events, status = [], []
    for cal in get_team_calendars():
        who = cal.split("@")[0]
        try:
            items = (
                service.events().list(
                    calendarId=cal, timeMin=time_min, timeMax=time_max,
                    singleEvents=True, orderBy="startTime", maxResults=per_calendar)
                .execute().get("items", [])
            )
            for e in items:
                start = e.get("start", {})
                events.append({
                    "who": who,
                    "summary": e.get("summary", "(no title)"),
                    "start": start.get("dateTime", start.get("date", "")),
                    "location": e.get("location", ""),
                })
            status.append({"calendar": cal, "ok": True, "count": len(items)})
        except Exception as ex:  # noqa: BLE001 - one inaccessible calendar shouldn't fail the rest
            status.append({"calendar": cal, "ok": False, "error": _short_err(ex)})
    events.sort(key=lambda e: e["start"])
    # Code does the sorting: collapse repeated events (e.g. the daily standup x5)
    # into a single entry per person so they don't spam the goal list.
    seen, collapsed = {}, []
    for e in events:
        key = (e["who"], e["summary"].strip().lower())
        if key in seen:
            seen[key]["_count"] += 1
            continue
        e = {**e, "_count": 1}
        seen[key] = e
        collapsed.append(e)
    for e in collapsed:
        if e.pop("_count", 1) > 1:
            e["summary"] += " (recurring)"
    return {"events": collapsed, "calendars": status}


def fetch_calendar(days: int = 7, max_results: int = 40) -> list:
    return fetch_calendar_detailed(days)["events"][:max_results]


# Senders that are pure noise (verification codes, automated security, bounces).
_GMAIL_DENY = re.compile(
    r"(accountprotection|account-security|security-noreply|verification|"
    r"mailer-daemon|postmaster|do-?not-?reply@)",
    re.IGNORECASE,
)


def fetch_gmail(max_results: int = 12) -> list:
    creds = _load_creds()
    if not creds:
        return []
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    # Code does the sorting: exclude Gmail's marketing/auto categories at the query
    # level (kills newsletters/notifications), then deny obvious automated senders.
    q = ("newer_than:7d in:inbox "
         "-category:promotions -category:social -category:updates -category:forums")
    msgs = (
        service.users().messages()
        .list(userId="me", maxResults=max_results * 3, q=q)
        .execute().get("messages", [])
    )
    out = []
    for m in msgs:
        full = (
            service.users().messages()
            .get(userId="me", id=m["id"], format="metadata",
                 metadataHeaders=["Subject", "From", "Date"])
            .execute()
        )
        headers = {h["name"]: h["value"] for h in full.get("payload", {}).get("headers", [])}
        sender = headers.get("From", "")
        if _GMAIL_DENY.search(sender):
            continue
        out.append({
            "subject": headers.get("Subject", "(no subject)"),
            "from": sender,
            "date": headers.get("Date", ""),
            "snippet": full.get("snippet", "")[:200],
        })
        if len(out) >= max_results:
            break
    return out


# --- Drive source selection (which Drive/folder to read, not just "my recent") ---
def get_drive_source() -> dict:
    """Current Drive source: {'mode': 'recent'|'drive'|'folder', 'id', 'name'}."""
    if os.path.exists(DRIVE_SOURCE_PATH):
        try:
            with open(DRIVE_SOURCE_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg.setdefault("mode", "recent")
            return cfg
        except (json.JSONDecodeError, OSError):
            pass
    return {"mode": "recent", "id": "", "name": "My Drive (recent)"}


def set_drive_source(mode: str, source_id: str = "", name: str = "") -> dict:
    cfg = {"mode": mode, "id": (source_id or "").strip(), "name": name}
    with open(DRIVE_SOURCE_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return cfg


def _extract_folder_id(s: str) -> str:
    """Accept a Drive folder URL or a raw ID and return the ID."""
    s = (s or "").strip()
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", s) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", s)
    return m.group(1) if m else s


def list_shared_drives() -> list:
    """Shared Drives (Workspace Team Drives) the user can access."""
    creds = _load_creds()
    if not creds:
        return []
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    try:
        drives = (
            service.drives().list(pageSize=100, fields="drives(id,name)")
            .execute().get("drives", [])
        )
    except Exception:  # noqa: BLE001 - no shared drives / not a Workspace account
        return []
    return [{"id": d["id"], "name": d.get("name", "")} for d in drives]


def fetch_drive(max_results: int = 10) -> list:
    creds = _load_creds()
    if not creds:
        return []
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    cfg = get_drive_source()
    common = dict(
        orderBy="modifiedTime desc", pageSize=max_results,
        fields="files(name, modifiedTime, mimeType, webViewLink)",
    )
    if cfg.get("mode") == "drive" and cfg.get("id"):
        # Files inside a specific Shared Drive.
        files = service.files().list(
            driveId=cfg["id"], corpora="drive",
            includeItemsFromAllDrives=True, supportsAllDrives=True, **common,
        ).execute().get("files", [])
    elif cfg.get("mode") == "folder" and cfg.get("id"):
        fid = cfg["id"]
        if fid.startswith("0A"):
            # A Shared Drive ROOT id (pasting the Shared Drive link) — query the
            # drive itself, which needs corpora=drive + driveId.
            files = service.files().list(
                driveId=fid, corpora="drive",
                includeItemsFromAllDrives=True, supportsAllDrives=True, **common,
            ).execute().get("files", [])
        else:
            # A normal/shared folder — list its children.
            files = service.files().list(
                q=f"'{fid}' in parents and trashed=false",
                includeItemsFromAllDrives=True, supportsAllDrives=True, **common,
            ).execute().get("files", [])
    else:
        # Default: the user's own recently-modified files.
        files = service.files().list(**common).execute().get("files", [])
    return [
        {"name": f.get("name", ""), "modified": f.get("modifiedTime", ""),
         "type": f.get("mimeType", "").split(".")[-1], "link": f.get("webViewLink", "")}
        for f in files
    ]


def preview() -> dict:
    """Everything the UI needs to show what's being pulled from each source."""
    if not is_connected():
        return {"connected": False, "calendar": [], "calendars": [], "gmail": [],
                "drive": [], "drive_source": get_drive_source(),
                "team_calendars": get_team_calendars()}
    cal = fetch_calendar_detailed()
    return {
        "connected": True,
        "calendar": cal["events"],
        "calendars": cal["calendars"],          # per-person access status
        "team_calendars": get_team_calendars(),
        "gmail": fetch_gmail(),
        "drive": fetch_drive(),
        "drive_source": get_drive_source(),
    }


# --------------------------------------------------------------------------- #
# Write fetched data into context/sources/ so the pipeline ingests it
# --------------------------------------------------------------------------- #
def sync_to_sources() -> dict:
    """Fetch from each connected source and write a markdown file per source into
    context/sources/. Returns how many items were written per source."""
    if not is_connected():
        raise RuntimeError("Not connected to Google.")
    os.makedirs(SOURCES_DIR, exist_ok=True)
    written = {}

    cal = fetch_calendar()
    if cal:
        lines = ["# Team calendar — next 7 days", ""]
        for e in cal:
            who = e.get("who", "")
            loc = f" @ {e['location']}" if e["location"] else ""
            lines.append(f"- [{who}] {e['start']} — {e['summary']}{loc}")
        _write("calendar_this_week.md", "\n".join(lines))
    written["calendar"] = len(cal)

    mail = fetch_gmail()
    if mail:
        lines = ["# Recent email (last 7 days, inbox)", ""]
        for m in mail:
            lines.append(f"- {m['subject']} — from {m['from']}\n  {m['snippet']}")
        _write("email_recent.md", "\n".join(lines))
    written["gmail"] = len(mail)

    files = fetch_drive()
    if files:
        lines = ["# Recently modified Drive files", ""]
        for f in files:
            lines.append(f"- {f['name']} (modified {f['modified']})")
        _write("drive_recent.md", "\n".join(lines))
    written["drive"] = len(files)

    return written


def _write(filename: str, content: str) -> None:
    with open(os.path.join(SOURCES_DIR, filename), "w", encoding="utf-8") as f:
        f.write(content + "\n")

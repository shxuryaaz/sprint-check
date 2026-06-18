# Context sources

Drop extra context files here. **Every `.md` or `.txt` file in this folder is
automatically injected** into Phase 1 alongside `context/project_context.md` on
each Generate — no code changes, no restart.

Use it for the live/factual layer the transcript can't supply:

- `journey_north_star.md` — paste/export the Journey North Star tracker
- `client_sow.md` — a client Statement of Work
- `calendar_this_week.md` — upcoming events (networking, demos, travel)
- anything else worth grounding goals in (recent email commitments, etc.)

Keep each file focused; the whole folder is concatenated into the prompt, so very
large files cost tokens.

## Where live connectors plug in

This folder is also the **seam for automation**. A Google Calendar / Gmail / Drive
connector (e.g. via an MCP server or Claude Cowork) would simply **write its
fetched data into a file here** (e.g. `calendar_this_week.md`) before a run — and
it's picked up the same way, with no pipeline changes.

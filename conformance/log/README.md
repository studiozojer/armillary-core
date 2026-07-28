# log

Log replay and gap detection — constitution `instances.md` **A-1** (`subscribe(stream, from_seq)`: replay durable events past the cursor, then tail live) and **A-3** (resume re-instantiates from the log: durable events since the current compaction/epoch baseline). This suite is the runnable half of those two rules: given a stream's durable event log and a client's cursor, an implementation must produce the right replay set, and must say so — not guess, not silently truncate — when the cursor asks for history the log no longer holds.

## What it tests

Every durable event carries a `seq`, monotonic within its stream (`schema/event.schema.json`). A client's cursor is **the last `seq` it has already seen** — exclusive, so the first event it actually needs has `seq` one past the cursor. Given `{log, from}`, an implementation must report:

- **`replayed`** — the `id`s, in order, of every event with `seq` greater than `from`.
- **`gap`** — `null` when the log holds everything back to (and including) what the cursor needs; otherwise `{ requestedFrom, earliestAvailable }`, where `requestedFrom` echoes the cursor the client sent and `earliestAvailable` is the earliest `seq` the log actually still holds. A gap means events between those two numbers existed once and are no longer available (e.g. compacted elsewhere) — the client's history has a hole, and the implementation must say so rather than silently start replay from wherever the log happens to begin.

This is exactly A-1/A-3's contract: the log is a stream's durable record of what actually happened, `subscribe(stream, from_seq)` is the one primitive every client resumes through, and a resumed instance never gets a projection quietly missing events its cursor thought it would get.

## Fixture format

Each fixture is a triplet: `<stem>.jsonl` + `<stem>.expected.json`.

**`<stem>.jsonl`** — a stream's durable event log, one JSON object per line, oldest first. Every line MUST validate against `schema/event.schema.json` (see the schema-validation half of this suite, below) — these are not toy shapes, they are the same envelope any conforming log actually writes. All lines in one fixture share the same `stream`.

**`<stem>.expected.json`**:

```json
{
  "from": 2,
  "replayed": ["<id>", "..."],
  "gap": null
}
```

or, when the cursor asks for history the log no longer holds:

```json
{
  "from": 0,
  "replayed": ["<id>", "..."],
  "gap": { "requestedFrom": 0, "earliestAvailable": 5 }
}
```

`from` is the query — the client's cursor to run against `<stem>.jsonl`. `replayed` and `gap` are the operation's required output.

## Fixtures

- **`basic`** — a fresh cursor (`from: 0`) against a short, ordinary log: every event replays, no gap.
- **`from-cursor`** — a cursor partway through the log (`from: 2`): only events after it replay, no gap (the log holds everything the cursor could still want).
- **`gap`** — a log whose earliest event is `seq 5`, not `seq 1` (a stream that has been compacted or trimmed elsewhere, upstream of what this fixture represents). A cursor of `0` asks for history starting at `seq 1`, which the log no longer holds: `replayed` is still every event actually present, but `gap` reports the hole honestly instead of pretending the log started where the client last saw it.

## Running the suite

An implementation loads `<stem>.jsonl` as the stream's durable log, runs its replay operation with the query `from` named in `<stem>.expected.json`, and deep-compares its own `{replayed, gap}` against that file. A conforming implementation passes every fixture. This directory has no dependency on any particular storage engine or language — it names only the schema and the constitution.

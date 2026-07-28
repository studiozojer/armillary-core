# conformance

Runnable fixtures. An implementation **proves** it runs the armillary by producing each fixture's `expected` output from its input — pass/fail, no prose claims. This directory is the standard's teeth: the survey (Q5, claim 3) found no harness anywhere that ships one, so the convention here is deliberately simple enough that every engine can afford it.

**Runner contract:** each suite directory holds fixture pairs. The implementation under test reads the input file, performs the named operation, and emits JSON; the result must deep-equal the `*.expected.json`. A conforming implementation passes every fixture in every suite it claims.

## Suites

- **`manifest/`** — manifest parsing (constitution/composition.md C-1..C-4). Input: TOML manifest text. Operation: parse to a composition. Output shape: `{operators: [{name, path}], commons: [...], repos: [...], protocols: [{name, source, load, when?}]}`. Covers: active vs commented entries, legacy `[[models]]`/`[[agents]]` normalization, `when` propagation, and overlay merge (C-6).

  **Two-file fixtures.** A fixture named `<stem>.toml` MAY be accompanied by `<stem>.local.toml`, in which case the operation is *parse both and merge* (C-6) rather than parse one. A runner discovers the overlay by stem; `*.local.toml` is never itself a base fixture.

  **Error fixtures.** A fixture carrying `<stem>.expected-error.json` instead of `<stem>.expected.json` MUST fail, and the emitted error must deep-equal that file. Shape: `{error, section?, name?}`, where `error` is a stable machine-readable code (`name_collision`, `parse_error`, `io_error`). Prose messages are for humans and may change freely; only the code and identifying fields are part of the contract.
- **`summon/`** — summon detection (constitution/composition.md B-1/B-3). Input: `{message, operators}`. Operation: detect summons. Output: ordered, deduped, canonical-cased operator names. Covers: word boundaries, unknown names, case folding, dedup.
- **`log/`** — log replay & gap detection (constitution/instances.md A-1/A-3). Input: a stream's durable event log (one `event.schema.json`-shaped envelope per line) plus a client cursor `from` (the last seq already seen, exclusive). Operation: replay every event past the cursor; report a gap when the log's earliest available event is later than what the cursor needs. Output shape: `{from, replayed: [id, ...], gap: null | {requestedFrom, earliestAvailable}}`. Covers: full replay from a fresh cursor, resuming from a partial cursor, and detecting a gap where earlier history is no longer available (e.g. compacted elsewhere). Every fixture line is itself schema-validated against `schema/event.schema.json` as part of running the suite.

## Status

Seeded 2026-07-24 from the pi `armillary-boot` extension's test cases — that extension is the first implementation expected to run these (its unit tests already assert the same behavior; pointing them at these files closes the loop). The log suite (`log/`, above) and event-envelope schema validation arrived with the first engine that has a log and a turn loop (2026-07-27/28). Wanted next: reducer-totality (instances.md P-3) and multi-writer/ownership fixtures — those follow once a reducer and a claim protocol exist to test against.

# conformance

Runnable fixtures. An implementation **proves** it runs the armillary by producing
each fixture's `expected` output from its input — pass/fail, no prose claims. This
directory is the standard's teeth: the survey (Q5, claim 3) found no harness
anywhere that ships one, so the convention here is deliberately simple enough that
every engine can afford it.

**Runner contract:** each suite directory holds fixture pairs. The implementation
under test reads the input file, performs the named operation, and emits JSON; the
result must deep-equal the `*.expected.json`. A conforming implementation passes
every fixture in every suite it claims.

## Suites

- **`manifest/`** — manifest parsing (constitution/composition.md C-1..C-4).
  Input: TOML manifest text. Operation: parse to a composition. Output shape:
  `{operators: [{name, path}], commons: [...], repos: [...], protocols: [{name,
  source, load, when?}]}`. Covers: active vs commented entries, legacy
  `[[models]]`/`[[agents]]` normalization, `when` propagation.
- **`summon/`** — summon detection (constitution/composition.md B-1/B-3).
  Input: `{message, operators}`. Operation: detect summons. Output: ordered,
  deduped, canonical-cased operator names. Covers: word boundaries, unknown
  names, case folding, dedup.

## Status

Seeded 2026-07-24 from the pi `armillary-boot` extension's test cases — that
extension is the first implementation expected to run these (its unit tests
already assert the same behavior; pointing them at these files closes the loop).
Wanted next: event-envelope validation against `schema/event.schema.json`,
reducer-totality (instances.md P-3), and log replay/resume suites — those arrive
with the first engine that has a log.

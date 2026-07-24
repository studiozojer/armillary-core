# armillary-core

**The armillary composition standard.** This repo holds the *constitution*: every rule
that holds **regardless of which modules are composed** — the manifest schema, load
timings, dispatch, summon-boot, the protocol interface, and the instance/log model —
as a normative spec with runnable conformance fixtures. An engine implements this
standard; a workspace instantiates it; the router hosts it. **Nothing else belongs
here.** Domain content goes back to the module that lives it — that scope guardrail
is this repo's first law, written here so the repo can enforce it against itself.

- `constitution/` — the normative documents. RFC-2119 keywords (MUST / SHOULD / MAY).
- `schema/` — machine-readable shapes (event envelope, manifest).
- `conformance/` — fixtures an independent implementation runs to *prove* it
  implements the standard, rather than approximately doing so.

## Vocabulary (normative)

Ratified 2026-07-24; the whole standard speaks these four nouns:

- **operator** — a composed identity: its files, graph, and protocols. Lives at
  `operators/<name>/` in a workspace. Model-agnostic.
- **model** — the engine piloting an operator in a given session. Recorded as
  provenance per turn; never part of identity.
- **instance** — an operator (or the bare dispatcher) instantiated in a live
  session window: the *running* thing.
- **log** — an instance's durable, typed, append-only record. All views —
  the context window, a client transcript, a summary — are **projections** of it.

## Status

**v0.1 — seeded 2026-07-24.** Documents: `constitution/instances.md` (the
instance/log model; the survey-hardened spine) and `constitution/composition.md`
(manifests, load timings, summon-boot — extracted from the router's lived
protocol). First fixtures cover manifest parsing and summon detection; the pi
`armillary-boot` extension is the first implementation expected to run them.

Known deferrals, deliberate: transport choice (Connect-RPC vs SSE — parked),
multi-writer/ownership (single-writer is v0 law), context paging/compaction
rules (stage 2), protocol-kind taxonomy (unearned; see the harness project).

## Provenance

Distilled from `zojercommons/projects/harness/` — the north star (direction),
`research/findings.md` (five-question prior-art survey; adopt/adapt/avoid
verdicts), and `research/ycc-architecture.md` (single-specimen deep read). The
decisions cited as "ratified" carry dates and live in the survey's decision
sheet. This repo states the rules; the harness project holds the reasoning.

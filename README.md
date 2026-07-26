# armillary-core

**The armillary harness: a composition standard, and an engine that obeys it.**

Two tiers, deliberately in one repo, with a hard seam between them.

**The standard** is every rule that holds **regardless of which modules are composed** — the manifest schema, load timings, dispatch, summon-boot, the protocol interface, and the instance/log model — as a normative spec with runnable conformance fixtures.

**The engine** is one implementation of it: a read-only files service over a composed workspace, born without a loop on purpose. A harness is roughly 5% loop and 95% edge-of-the-world plumbing, so the plumbing comes first and is useful on its own while the loop does not exist.

```
constitution/   the normative documents. RFC-2119 keywords (MUST / SHOULD / MAY).
schema/         machine-readable shapes (event envelope).
conformance/    fixtures an implementation runs to PROVE it conforms, rather
                than approximately doing so.
crates/
  armillary-composition/   manifest → composition. The conformance target.
  armillary-engine/        axum binary: /composition /tree /file /health.
```

## The seam

**`constitution/` and `conformance/` MUST NOT reference `crates/`.** The fixtures are consumed as a black box: inputs in, expected output compared, no imports.

This matters because a spec shipping beside its only implementation is the standard failure mode — the spec stops being tested independently and quietly becomes documentation of what the code does. The defense is not a repository boundary. It is **an implementation that is not this one running the same fixtures**, and that exists: the pi `armillary-boot` extension is written in TypeScript, lives in a different repo, and passes `conformance/` from there. When a fixture is added here, it has to pass there too — and when it does not, that is information about the spec rather than a bug in the extension.

The capacity being protected is concrete. In one evening the constitution refused two things: **G-1** declined a studio prose rule as workspace-tier rather than composition-agnostic law, and **C-5** froze non-collapse of protocol *kind* as normative. A standard that cannot refuse its implementation is not one.

## Scope

**No domain content belongs here.** A rule specific to an operator, a commons, a repo, or a person belongs to that module's own protocols at its own tier (workspace → operator → collaborator). That guardrail is **G-1**, written into the constitution so the repo can enforce it against itself.

## Vocabulary (normative)

Ratified 2026-07-24; the whole standard speaks these four nouns:

- **operator** — a composed identity: its files, graph, and protocols. Lives at `operators/<name>/` in a workspace. Model-agnostic.
- **model** — what pilots an operator in a given session. Recorded as provenance per turn; never part of identity. Distinct from an **engine**: an engine is a harness implementing this standard, a model is the weights it calls.
- **instance** — an operator (or the bare dispatcher) instantiated in a live session window: the *running* thing.
- **log** — an instance's durable, typed, append-only record. All views — the context window, a client transcript, a summary — are **projections** of it.

## Running the engine

```bash
cargo run -p armillary-engine -- --root /path/to/workspace
curl -s http://127.0.0.1:7778/health
```

Binds loopback by default and **refuses `--bind 0.0.0.0`**: it serves unauthenticated reads of an entire workspace, so it must bind loopback or a specific tailnet address. `.env*` is never listed and never served; `node_modules`, `target`, `build`, `.next` and `.git` are never listed. The engine serves the *disk*, not git, so `.gitignore` filters nothing — the denylist is what stands between a tailnet and a credential file.

Deployment recipe: `zojercommons/setup/armillary-engine-deploy.md` (studio-local).

```bash
cargo test            # conformance fixtures, guard, routes
```

## Status

**v0.1 — standard seeded 2026-07-24, machinery added 2026-07-26, public 2026-07-26.**

`constitution/instances.md` (the instance/log model; the survey-hardened spine) and `constitution/composition.md` (manifests, load timings, summon-boot, the overlay merge — extracted from the router's lived protocol). Fixtures cover manifest parsing, legacy section normalization, overlay merge, name collision, and summon detection. Two implementations run them: this repo's `armillary-composition`, and the pi `armillary-boot` extension.

Known deferrals, deliberate: transport choice (Connect-RPC vs SSE — parked), multi-writer/ownership (single-writer is v0 law), context paging/compaction rules, protocol-kind taxonomy (unearned — see C-5), and the loop itself.

## Provenance

Distilled from `zojercommons/projects/harness/` — the north star (direction), `research/findings.md` (five-question prior-art survey; adopt/adapt/avoid verdicts), and `research/ycc-architecture.md` (single-specimen deep read). Decisions cited as "ratified" carry dates. **This repo states the rules; the harness project holds the reasoning.**

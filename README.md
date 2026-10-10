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

Set `ANTHROPIC_API_KEY` in the environment, or drop a key at `~/.config/armillary/anthropic-key` (env wins if both are present) — never a flag, so a key never lands in shell history or `ps`. OpenCode Zen models (`zen/<slug>`) work the same way, via `OPENCODE_ZEN_API_KEY` or `~/.config/armillary/zen-key`. Without a key the engine still serves the Explorer, and every `send` on a model whose provider has no key fails with the named `no_api_key` error instead of the engine refusing to start.

Zen's `zen/deepseek-v4.1-flash` uses Chat Completions. `zen/gpt-6-astra`, `zen/gpt-6-sol`, and `zen/gpt-6-luna` use the Responses API, including its function-call stream and replay format. Add the desired IDs as `[[model]]` entries in the host's `models.toml` to offer them in the Expo picker; the app reads `GET /models` rather than keeping its own list. Access and billing for these models follow the host's Zen key.

`--model` sets the process-wide default only — the model a session pilots with when its instance names none of its own; an instance's own recorded model always wins (`POST /instances`' `model` field, pinned at creation). Absent `--model`, the default comes from `~/.config/armillary/models.toml`'s `default` line, then `claude-sonnet-5`. That same file's `[[model]]` entries — an `id` and optional `label` each — are the host's declared catalog, served (alongside the resolved default and each entry's provider/key-presence) at `GET /models` for a client-side picker; a host with no file still boots and pilots, it just has nothing to list.

Binds loopback by default and **refuses `--bind 0.0.0.0`**: it serves unauthenticated reads of an entire workspace, so it must bind loopback or a specific tailnet address. `.env*` is never listed and never served; `node_modules`, `target`, `build`, `.next` and `.git` are never listed. The engine serves the *disk*, not git, so `.gitignore` filters nothing — the denylist is what stands between a tailnet and a credential file.

Deployment recipe: `zojercommons/setup/armillary-engine-deploy.md` (studio-local).

```bash
cargo test            # conformance fixtures, guard, routes
```

## Reply latency and diagnostics

Session titles are best-effort background work after a successful foreground turn, not a model call the reply must wait for. Each stream allows one title request at a time, with a 15-second provider timeout. A new foreground turn cancels an active title request. Before renaming, the engine checks under the session write lock that neither the latest user message nor the current title has changed. Existing titles are checked at most once per 60 seconds; untitled sessions can retry on the next eligible successful turn. This is turn-triggered work, not a periodic scheduler, and restart resets the in-memory throttle.

Engine stderr includes JSON records prefixed `turn_timing`, `round_timing`, `tool_timing`, and `title_timing`. They record elapsed milliseconds, stream/generation identifiers, round counts, projection/provider/tool time, first nonempty streamed text, and context size. Turn-level first-text time starts at foreground turn entry; round-level first-text time starts at the provider call and includes relay scheduling. A missing first-text value is `null`, not zero. Provider time includes draining the stream relay; tool time includes recording its events. These measurements do not cover client/network delay before turn entry, and stage totals need not sum to the whole turn duration.

`context_content_bytes` counts UTF-8 system/message content and serialized tool inputs, not tokens or total request size; tool definitions and wire framing are excluded. The new timing records contain no prompts, replies, tool arguments, tool results, or credentials. Existing diagnostic logging is unchanged. Background title time is reported separately rather than charged to foreground reply latency. These measurements provide a baseline for subsequent context and tool-loop optimization; this change does not enable remote shell execution or broaden grants.

## Status

**v0.1 — standard seeded 2026-07-24, machinery added 2026-07-26, public 2026-07-26.**

`constitution/instances.md` (the instance/log model; the survey-hardened spine) and `constitution/composition.md` (manifests, load timings, summon-boot, the overlay merge — extracted from the router's lived protocol). Fixtures cover manifest parsing, legacy section normalization, overlay merge, name collision, summon detection, and (as of 2026-07-27/28) log replay/gap detection with event-envelope schema validation. Two implementations run the composition fixtures: this repo's `armillary-composition`, and the pi `armillary-boot` extension.

The engine has a loop now: `send` records the user message, runs one turn against a pluggable model provider, streams transient deltas and the durable outcome to every subscriber, and supports mid-turn interrupt and context eviction — proven by `cargo test` and, live, by a `curl` turn against a running engine.

Known deferrals, deliberate: transport choice (Connect-RPC vs SSE — parked), multi-writer/ownership (single-writer is v0 law), context paging/compaction rules, and protocol-kind taxonomy (unearned — see C-5).

## Provenance

Distilled from `zojercommons/projects/harness/` — the north star (direction), `research/findings.md` (five-question prior-art survey; adopt/adapt/avoid verdicts), and `research/ycc-architecture.md` (single-specimen deep read). Decisions cited as "ratified" carry dates. **This repo states the rules; the harness project holds the reasoning.**

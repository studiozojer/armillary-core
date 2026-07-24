# composition — manifests, load timings, summon

How a workspace declares what it is, and what a conforming engine does with the
declaration. Extracted from the router's lived dispatcher protocol (the prose form
remains in the router's `CLAUDE.md`; this is the normative form an engine
implements).

## 1 · The manifest

- **C-1** A workspace's composition is **declared, not discovered**: the engine
  knows a module or protocol exists because `modules.toml` (+ the private overlay
  `modules.local.toml`) names it. Engines MUST NOT scan for modules.
- **C-2** Module kinds: `[[operators]]`, `[[commons]]`, `[[repos]]` — each entry
  `{name, path, repo?, note?}` — and `[[protocols]]` entries
  `{name, source, load, when?, requires?}`. Legacy section names `[[models]]` and
  `[[agents]]` MUST parse as `[[operators]]` during the 2026-07-24 rename
  migration; declared legacy `models/` paths are honored as written.
- **C-3** Commented-out manifest entries are examples and MUST NOT be treated as
  declarations. Any composition summary shown or injected MUST be **byte-derived**
  from the manifests — an engine never asks the model to re-derive the
  composition. *(The gemma confabulation, made structurally impossible.)*
- **C-4** Everything is presence-gated: a bare clone declares nothing, composes
  nothing, and is a working host — not an error. A missing `requires:` dependency
  means *skip the protocol*, not fail.

## 2 · Load timings

- **L-1** `load = "boot"` — the body is read at session start and composes the
  starting context.
- **L-2** `load = "on-demand"` — registered as a pointer (`name` + `when`); the
  body MUST NOT be read until the work matches `when`. Lazy by design: anything
  auto-loaded becomes a write-attractor.
- **L-3** `load = "session-end"` — best-effort at close, and **best-effort is
  normative**: a clean close is not promised, so nothing that must persist may
  rely solely on a session-end protocol. Durability requires a threshold-fired or
  boot-time path beside the close-time one.

## 3 · Summon

- **B-1** An explicit `@<operator>` summon boot-gates the turn: the summoned
  operator's boot protocol runs **first**, before any repo, skill, or tool touch
  except the dispatcher itself — even when the same message names a destination.
  The destination is material to orient *through*, not a reason to skip orienting.
- **B-2** Where the engine can enforce (hooks, system-prompt injection, pre-tool
  gates), summon-boot is **deterministic**: the engine reads the operator's boot
  doc itself and injects it with an authoritative framing, re-asserted per turn so
  it survives compaction. Boot-gating that depends on the model choosing to comply
  is advisory and does not satisfy this rule. *(instances.md S-3 applies.)*
- **B-3** A summon of a name not declared in the manifest is a no-op, not an
  error. One boot per (session, operator).
- **B-4** Injected boot content MUST state its **path anchor** — the operator's
  root, and how bare relative paths resolve under it — rather than leaving path
  resolution to model inference. *(The anchor finding from the weak-pilot bench.)*

## 4 · The scope guardrail

- **G-1** This constitution holds only rules that are true **regardless of which
  modules are composed**. A rule specific to an operator, a commons, a repo, or a
  person belongs to that module's own protocols, at its own tier (workspace →
  operator → collaborator). An engine MUST NOT require a workspace to reorganize
  around it — each stage lands in the household.

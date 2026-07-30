//! The tool surface — what a session may do, and the switch that does it.
//!
//! Three things live here and nowhere else: the JSON definitions sent to the
//! provider, the name→function dispatch, and the mapping from a failure to the
//! machine code a `tool_result` event records.
//!
//! **This module is the join that had no owner.** The design that preceded it
//! specified three tool *bodies* and a path *gate* and never the switch between
//! them — the same shape as a boot event nobody wrote and a front door that
//! belonged to no task. It is written first, before the loop that calls it, so
//! that gap cannot reopen.
//!
//! **Verbs are engine-owned.** Which tools exist is not a composition question:
//! C-1 governs modules and protocols, and reading a file is a primitive, not a
//! composed module. What a workspace declares shapes what those verbs *reach*,
//! not which verbs exist. That is a recorded debt, not an oversight — a second
//! engine could ship a different surface and both would pass conformance today.

use std::path::Path;

/// A tool call that did not succeed.
///
/// `status` is the machine code, and it is the sovereign half of S-1: the
/// engine reads it, the log records it typed, and loop control keys on it. The
/// model never sees this struct — it sees `is_error` plus whatever the
/// projection renders, which is all the provider channel has room for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolError {
    /// A stable code, never prose. Guard codes pass through verbatim so the
    /// transcript and the log say the same word.
    pub status: &'static str,
    /// Detail for the log and for rendering. May be empty.
    pub detail: String,
}

impl ToolError {
    fn new(status: &'static str, detail: impl Into<String>) -> Self {
        ToolError {
            status,
            detail: detail.into(),
        }
    }
}

/// The tool definitions sent with every request.
///
/// Order is fixed and deliberate. Tool definitions render at the front of the
/// prompt, ahead of the system prompt and messages, so a set that reorders
/// between requests invalidates the whole cached prefix. It costs nothing to
/// fix that now with one tool and is awkward to retrofit later.
pub fn definitions() -> Vec<serde_json::Value> {
    vec![serde_json::json!({
        "name": "get_composition",
        "description": "Return what this workspace is composed of: the operators, \
                        the commons, the repos, and the protocols declared in its \
                        manifests, along with which protocol sources are actually \
                        present on disk. Call this to find out what exists before \
                        reasoning about it — the answer is derived from the \
                        manifest files, not from memory.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    })]
}

/// Execute one tool call.
///
/// **An unknown name is an error result, never nothing.** Both surveyed
/// harnesses that hit this case guarantee a result rather than letting the pair
/// dangle, and they are right for a reason measured here: a `tool_use` with no
/// `tool_result` is a 400 that kills every later turn. A stale or misspelled
/// name must come back as a refusal the model can read.
pub fn dispatch(name: &str, _input: &serde_json::Value, root: &Path) -> Result<String, ToolError> {
    match name {
        "get_composition" => get_composition(root),
        other => Err(ToolError::new(
            "unknown_tool",
            format!("no tool named {other}"),
        )),
    }
}

/// The full composition payload — the single implementation, shared by the
/// `/composition` route and by the tool below.
///
/// C-3 as running code: byte-derived from the manifests by a TOML parser, never
/// re-derived by a model. The rule exists because a local model once read
/// commented-out examples as a live composition, and a parser makes that
/// structurally impossible — comments are not data.
///
/// Carries the sha256 of every manifest and every resolved protocol source.
/// Nothing consumes them yet; they cost one hash over bytes already in memory
/// and mean "which bytes were in that window" stays answerable.
pub fn composition_payload(root: &Path) -> Result<serde_json::Value, ToolError> {
    let unreadable = |e: String| ToolError::new("composition_unreadable", e);

    let composition = armillary_composition::parse_workspace(root)
        .map_err(|e| unreadable(e.to_string()))?;

    let mut manifests = Vec::new();
    for name in ["modules.toml", "modules.local.toml"] {
        if let Ok(bytes) = std::fs::read(root.join(name)) {
            manifests.push(serde_json::json!({
                "path": name,
                "sha256": crate::hash::sha256_hex(&bytes),
            }));
        }
    }

    // C-4: a protocol whose source is not present is reported absent, not an
    // error. Through the guard, not `root.join` — an absolute `source` would
    // otherwise be read verbatim, since `Path::join` with an absolute argument
    // discards the base.
    let protocol_sources: Vec<serde_json::Value> = composition
        .protocols
        .iter()
        .map(|p| {
            match crate::guard::resolve(root, &p.source)
                .and_then(|path| std::fs::read(&path).map_err(|_| crate::guard::GuardError::NotFound))
            {
                Ok(bytes) => serde_json::json!({
                    "name": p.name, "path": p.source, "present": true,
                    "sha256": crate::hash::sha256_hex(&bytes),
                }),
                Err(_) => serde_json::json!({
                    "name": p.name, "path": p.source, "present": false,
                }),
            }
        })
        .collect();

    let mut body = serde_json::to_value(&composition).map_err(|e| unreadable(e.to_string()))?;
    body["manifests"] = serde_json::json!(manifests);
    body["protocol_sources"] = serde_json::json!(protocol_sources);
    Ok(body)
}

/// The model-facing composition: `composition_payload` with the digests removed.
///
/// **The sha256 values are stripped deliberately, and the strip is real** — this
/// calls the same builder the `/composition` route does, so the two cannot drift
/// into disagreeing about what is composed. Roughly a quarter of that payload is
/// hex a model can neither verify nor act on; it exists for drift detection,
/// which is an engine concern.
///
/// `present` survives, because presence is the C-4 question a model actually
/// needs answered: a protocol whose source is missing is skipped, not an error.
fn get_composition(root: &Path) -> Result<String, ToolError> {
    let mut body = composition_payload(root)?;

    if let Some(manifests) = body.get_mut("manifests").and_then(|m| m.as_array_mut()) {
        for entry in manifests {
            entry.as_object_mut().map(|o| o.remove("sha256"));
        }
    }
    if let Some(sources) = body
        .get_mut("protocol_sources")
        .and_then(|m| m.as_array_mut())
    {
        for entry in sources {
            entry.as_object_mut().map(|o| o.remove("sha256"));
        }
    }

    serde_json::to_string_pretty(&body)
        .map_err(|e| ToolError::new("composition_unreadable", e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A workspace shaped like the real one: a public manifest, a private
    /// overlay, one protocol whose source exists and one whose source does not.
    fn workspace() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("modules.toml"),
            "# a commented-out example that MUST NOT read as a declaration\n\
             # [[repos]]\n\
             # name = \"ghost\"\n\
             # path = \"repos/ghost\"\n\
             [router]\n\
             contains = [\"CLAUDE.md\"]\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("modules.local.toml"),
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\n\
             [[repos]]\nname = \"kairos-engine\"\npath = \"repos/kairos-engine\"\n\n\
             [[protocols]]\nname = \"board\"\nsource = \"present.md\"\nload = \"boot\"\n\n\
             [[protocols]]\nname = \"athanor\"\nsource = \"absent.md\"\nload = \"on-demand\"\n",
        )
        .unwrap();
        fs::write(dir.path().join("present.md"), "# board").unwrap();
        dir
    }

    #[test]
    fn get_composition_reports_what_the_manifests_declare() {
        let dir = workspace();
        let out = dispatch("get_composition", &serde_json::json!({}), dir.path()).unwrap();

        assert!(out.contains("tycho"), "{out}");
        assert!(out.contains("kairos-engine"), "{out}");
        assert!(out.contains("board"), "{out}");
    }

    #[test]
    fn a_commented_out_entry_is_not_a_declaration() {
        // C-3, and the reason it exists: a local model once read commented-out
        // examples as a live composition. A TOML parser makes it impossible.
        let dir = workspace();
        let out = dispatch("get_composition", &serde_json::json!({}), dir.path()).unwrap();

        assert!(
            !out.contains("ghost"),
            "a commented-out repo reached the model: {out}"
        );
    }

    #[test]
    fn protocol_presence_is_reported_because_a_missing_source_is_skipped_not_an_error() {
        let dir = workspace();
        let out = dispatch("get_composition", &serde_json::json!({}), dir.path()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();

        let by_name = |n: &str| -> bool {
            parsed["protocol_sources"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["name"] == n)
                .unwrap()["present"]
                .as_bool()
                .unwrap()
        };

        assert!(by_name("board"), "present.md exists: {out}");
        assert!(!by_name("athanor"), "absent.md does not exist: {out}");
    }

    #[test]
    fn the_sha256_digests_stay_out_of_the_model_facing_payload() {
        // A quarter of `/composition`'s bytes are hex a model cannot verify and
        // cannot act on. Drift detection is an engine concern.
        let dir = workspace();
        let out = dispatch("get_composition", &serde_json::json!({}), dir.path()).unwrap();

        assert!(!out.contains("sha256"), "{out}");
    }

    #[test]
    fn the_shared_builder_carries_the_digests_the_tool_strips() {
        // Without this, "no sha256 in the tool payload" is true by construction
        // rather than by design — a test that passes identically whether or not
        // the strip exists. Mutation-checked: removing the strip reddens the
        // pair below, not neither.
        let dir = workspace();
        let full = composition_payload(dir.path()).unwrap();

        assert!(
            full["manifests"][0]["sha256"].is_string(),
            "the route's payload must carry manifest digests: {full}"
        );
        assert!(
            full["protocol_sources"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["sha256"].is_string()),
            "a present protocol source must be hashed: {full}"
        );
    }

    #[test]
    fn a_bare_clone_composes_nothing_and_that_is_not_an_error() {
        // C-4: presence-gated throughout. A bare clone is a working host.
        let dir = tempfile::tempdir().unwrap();
        let out = dispatch("get_composition", &serde_json::json!({}), dir.path())
            .expect("a bare clone is a working host, not a failure");

        assert!(!out.contains("tycho"), "{out}");
    }

    #[test]
    fn an_unknown_tool_name_returns_an_error_result_rather_than_nothing() {
        // A tool_use with no tool_result is a 400 that kills every later turn,
        // so a stale or misspelled name must come back as something the model
        // can read.
        let dir = workspace();
        let err = dispatch("read_the_future", &serde_json::json!({}), dir.path()).unwrap_err();

        assert_eq!(err.status, "unknown_tool");
        assert!(err.detail.contains("read_the_future"));
    }

    #[test]
    fn the_definition_set_is_ordered_and_schema_shaped() {
        let defs = definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0]["name"], "get_composition");
        assert_eq!(defs[0]["input_schema"]["type"], "object");
        assert!(
            defs[0]["description"].as_str().unwrap().len() > 40,
            "the description is what decides whether a model calls it"
        );
    }
}

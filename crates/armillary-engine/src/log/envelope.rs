//! The durable event envelope — mirrors `schema/event.schema.json` v0.1
//! exactly (I-2 names each field this struct carries; this is that
//! agreement rendered as a Rust type, not a redesign of it).
//!
//! Transient events reuse this same shape with `seq == 0` and are never
//! persisted (I-4) — `store::LogStore::append` is where that is enforced.

use serde::{Deserialize, Serialize};

/// Event types I-1/I-4 name as durable — the fact side of the durable/
/// transient split. Not exhaustive of every type the system will ever emit;
/// P-3 requires a *reducer* to be total over durable types, which is a
/// separate, later concern from this list existing at all.
pub const DURABLE_TYPES: &[&str] = &[
    "instance_created",
    "boot",
    "composition",
    "user_message",
    "assistant_message",
    "interrupt",
    "context_evict",
    "dispatch",
    "return",
    "tool_use",
    "tool_result",
    // A write's EFFECT, distinct from the `tool_use` that requested it (D-1).
    // `tool_use` records what the model ASKED FOR and says nothing about what
    // reached the disk; the intent/effect gap is not hypothetical (the sync
    // work shipped a `verdict()` returning `current` for twenty-four repos
    // having contacted nothing).
    "file_changed",
    // A repo verb's EFFECT, in the `file_changed` tradition: past tense,
    // because these record what happened rather than what was asked for. They
    // live in the `workspace` stream, which no model context projects over —
    // see `projection.rs`'s arm for why that is stated rather than achieved by
    // leaving them out.
    "repo_fetched",
    "repo_pulled",
    "repo_pushed",
    "repo_committed",
];

/// Who REQUESTED an action, when that differs from what performed it.
///
/// Deliberately a *different* type from `principals::Principal` and
/// deliberately smaller: that one carries a token hash and a grant list,
/// neither of which belongs in a durable event. A log records identity, not
/// credentials — and a struct that could serialize a hash into a stream is
/// one refactor away from doing it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActorPrincipal {
    pub name: String,
}

/// `{role, instance, principal}` — I-2 requires actor be structured, never a
/// free string, so a summarizing model can't blur "who did this" into prose.
///
/// `principal` extends that argument rather than departing from it: for an
/// action a device requested and the engine performed, "who did this" has
/// two honest answers, and burying the requester in each event type's `data`
/// would have every type invent its own shape for it. Optional and skipped
/// when absent, so every event recorded before this field existed serializes
/// and replays byte-identically.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Actor {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub principal: Option<ActorPrincipal>,
}

/// The closed set of actor roles the schema enumerates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Operator,
    System,
    Tool,
    Machine,
}

/// SHOULD-carry size/cost (I-2) so budgets and eviction can be computed
/// without re-reading payloads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cost {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u64>,
}

/// The envelope every durable event carries (I-2). `type` is a reserved
/// word in Rust, so the field is named `event_type` here and renamed to
/// `type` on the wire — the schema's name wins on disk, not Rust's.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub stream: String,
    pub id: String,
    /// Monotonic within its stream (I-2 invariant iii). `0` marks a
    /// transient event — never persisted, never cursor-advancing (I-4).
    pub seq: u64,
    /// RFC 3339.
    pub ts: String,
    pub actor: Actor,
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    pub version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
    pub data: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_actor_without_a_principal_serializes_exactly_as_before() {
        // The compatibility claim this whole decision rests on. Every event
        // in every existing stream has no principal; if this shape changes,
        // the conformance fixtures and every stored log drift at once.
        let a = Actor { role: Role::User, instance: None, principal: None };
        assert_eq!(serde_json::to_string(&a).unwrap(), r#"{"role":"user"}"#);
    }

    #[test]
    fn an_actor_with_a_principal_names_it_in_one_place() {
        let a = Actor {
            role: Role::Machine,
            instance: None,
            principal: Some(ActorPrincipal { name: "iphone".to_string() }),
        };
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            r#"{"role":"machine","principal":{"name":"iphone"}}"#
        );
    }

    #[test]
    fn an_actor_recorded_before_this_field_existed_still_deserializes() {
        // Replay of a stored log must not break. This is the read half of
        // the same compatibility claim.
        let a: Actor = serde_json::from_str(r#"{"role":"operator","instance":"s1"}"#).unwrap();
        assert_eq!(a.role, Role::Operator);
        assert!(a.principal.is_none());
    }
}

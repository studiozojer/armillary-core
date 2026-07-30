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
];

/// `{role, instance}` — I-2 requires actor be structured, never a free
/// string, so a summarizing model can't blur "who did this" into prose.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Actor {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
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

//! The event log: envelope types and the append-only per-stream store.
//!
//! Constitution I-1: every instance writes to exactly one durable,
//! append-only, typed event log (its stream). This module is that log —
//! `envelope` names the wire shape, `store` is the file-backed writer/reader.

pub mod envelope;
pub mod store;

//! `Sessions`: the one writer per stream (A-4), and the live fanout over it.
//!
//! `Sessions` owns the durable `LogStore` and, per stream, a broadcast
//! channel created lazily on first append or subscribe. `append` assigns
//! `seq`/`id`/`ts`, writes durably FIRST (I-5: a failed write surfaces as
//! `SessionError::Log`, never swallowed, and never broadcast), THEN
//! broadcasts to live subscribers — so a subscriber never observes an event
//! that did not make it to disk.

use crate::log::envelope::EventEnvelope;
use crate::log::store::LogStore;
use axum::http::StatusCode;
use std::collections::HashMap;
use std::io;
use std::sync::Mutex;
use std::time::SystemTime;
use tokio::sync::broadcast;

/// Capacity of each stream's live broadcast channel (A-2: a subscriber that
/// falls this far behind drops itself via `RecvError::Lagged` rather than
/// blocking the writer).
const CHANNEL_CAPACITY: usize = 256;

/// A not-yet-enriched event: the caller supplies the parts I-2 leaves to the
/// writer's intent (`actor`, `type`, `data`); `Sessions::append` assigns the
/// parts I-2 requires the *store* to guarantee (`stream`, `id`, `seq`, `ts`).
pub struct NewEvent {
    pub actor: crate::log::envelope::Actor,
    pub event_type: String,
    pub data: serde_json::Value,
}

/// Placeholder for the loop's turn-ownership handle (Task 10+). `Sessions`
/// only needs a place to hold it — never to interpret it — so it stays
/// opaque here rather than pulling a dependency this crate doesn't have yet.
#[allow(dead_code)]
pub struct TurnHandle;

/// Per-stream live state.
struct StreamState {
    tx: broadcast::Sender<EventEnvelope>,
    /// Unused until the loop lands (Task 10+); the field exists now so A-4's
    /// one-writer-per-stream claim has somewhere to be recorded without a
    /// second registry appearing later.
    #[allow(dead_code)]
    turn: Option<TurnHandle>,
}

impl StreamState {
    fn new() -> Self {
        StreamState {
            tx: broadcast::channel(CHANNEL_CAPACITY).0,
            turn: None,
        }
    }
}

/// Errors `Sessions`/the instance routes can surface. `Log` wraps a durable
/// write failure (I-5); `UnknownInstance` is raised by route-level lookups
/// (attach) against a stream nothing ever created.
#[derive(Debug)]
pub enum SessionError {
    UnknownInstance,
    Log(io::Error),
}

impl From<io::Error> for SessionError {
    fn from(e: io::Error) -> Self {
        SessionError::Log(e)
    }
}

impl SessionError {
    /// Stable machine-readable `(status, code)` pair for a route to return —
    /// never the `Debug` rendering, matching `guard::GuardError::code`'s
    /// posture: a client should not be handed a Rust enum's shape as an API.
    pub fn into_response(self) -> (StatusCode, String) {
        match self {
            SessionError::UnknownInstance => {
                (StatusCode::NOT_FOUND, "unknown_instance".to_string())
            }
            SessionError::Log(e) => {
                eprintln!("log_write_failed: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, "log_write_failed".to_string())
            }
        }
    }
}

/// Owns the durable log and the live fanout over it.
///
/// A single coarse `Mutex` over the whole `HashMap` serializes every append
/// across every stream. At v0 that costs nothing worth optimizing: A-4 makes
/// one writer per stream a construction invariant, so two appends to the
/// SAME stream never legitimately race, and a global lock rather than a
/// per-stream one is simply the cheapest thing that is still correct until a
/// second writer is a real problem to solve.
pub struct Sessions {
    store: LogStore,
    inner: Mutex<HashMap<String, StreamState>>,
}

impl std::fmt::Debug for Sessions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sessions").finish_non_exhaustive()
    }
}

impl Sessions {
    pub fn new(store: LogStore) -> Self {
        Sessions {
            store,
            inner: Mutex::new(HashMap::new()),
        }
    }

    pub fn store(&self) -> &LogStore {
        &self.store
    }

    /// Assigns `seq` (the stream's head, plus one), `id` (`uuid` v4), and
    /// `ts` (RFC 3339 with millisecond precision, now); appends durably
    /// (I-5: a failed write propagates as `SessionError::Log` — it is never
    /// swallowed, and this event is never broadcast); THEN broadcasts to any
    /// live subscribers.
    ///
    /// The lock is held for the whole operation (read current head, write,
    /// broadcast) — see the struct doc for why a coarse lock is fine here.
    pub fn append(&self, stream: &str, partial: NewEvent) -> Result<EventEnvelope, SessionError> {
        let mut inner = self.inner.lock().unwrap();

        let seq = self.store.head_seq(stream)? + 1;
        let ev = EventEnvelope {
            stream: stream.to_string(),
            id: uuid::Uuid::new_v4().to_string(),
            seq,
            ts: humantime::format_rfc3339_millis(SystemTime::now()).to_string(),
            actor: partial.actor,
            event_type: partial.event_type,
            thread: None,
            parent: None,
            version: 1,
            cost: None,
            data: partial.data,
        };

        self.store.append(stream, &ev)?;

        let state = inner
            .entry(stream.to_string())
            .or_insert_with(StreamState::new);
        // `send` errors only when there are zero receivers — not an error
        // condition here (A-2): nobody currently listening is the common
        // case, not a fault, so the result is deliberately discarded.
        let _ = state.tx.send(ev.clone());

        Ok(ev)
    }

    /// Broadcasts a transient hint — `ev.seq` MUST be `0` (I-4: a hint is
    /// never persisted, so it never reaches `LogStore` at all, which is the
    /// one place that would otherwise enforce this). Debug-asserted rather
    /// than checked in release: this is an internal-caller contract, not
    /// user input arriving over HTTP.
    pub fn broadcast_transient(&self, stream: &str, ev: EventEnvelope) {
        debug_assert_eq!(ev.seq, 0, "broadcast_transient: seq must be 0 (I-4)");
        let mut inner = self.inner.lock().unwrap();
        let state = inner
            .entry(stream.to_string())
            .or_insert_with(StreamState::new);
        let _ = state.tx.send(ev);
    }

    /// A-1's live half: replay is `LogStore::read_from`, and this is the
    /// "then tail live" continuation. Capacity 256 (A-2): a receiver that
    /// falls behind gets `RecvError::Lagged` and drops itself rather than
    /// blocking this writer.
    pub fn subscribe_live(&self, stream: &str) -> broadcast::Receiver<EventEnvelope> {
        let mut inner = self.inner.lock().unwrap();
        let state = inner
            .entry(stream.to_string())
            .or_insert_with(StreamState::new);
        state.tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::envelope::{Actor, Role};
    use tokio::sync::broadcast::error::RecvError;

    fn actor() -> Actor {
        Actor {
            role: Role::User,
            instance: None,
        }
    }

    fn new_event() -> NewEvent {
        NewEvent {
            actor: actor(),
            event_type: "user_message".to_string(),
            data: serde_json::json!({"text": "hi"}),
        }
    }

    fn sessions() -> (tempfile::TempDir, Sessions) {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        (dir, Sessions::new(store))
    }

    #[tokio::test]
    async fn append_assigns_monotonic_seq_and_returns_the_enriched_envelope() {
        let (_dir, sessions) = sessions();

        let e1 = sessions.append("s1", new_event()).unwrap();
        let e2 = sessions.append("s1", new_event()).unwrap();

        assert_eq!(e1.seq, 1);
        assert_eq!(e2.seq, 2);
        assert_eq!(e1.stream, "s1");
        assert!(!e1.id.is_empty());
        assert_ne!(e1.id, e2.id, "each event gets its own id");
        assert!(!e1.ts.is_empty());
    }

    #[tokio::test]
    async fn append_persists_to_the_store_it_wraps() {
        let (_dir, sessions) = sessions();
        sessions.append("s1", new_event()).unwrap();

        assert_eq!(sessions.store().head_seq("s1").unwrap(), 1);
    }

    #[tokio::test]
    async fn broadcast_delivers_to_two_subscribers() {
        let (_dir, sessions) = sessions();
        let mut rx1 = sessions.subscribe_live("s1");
        let mut rx2 = sessions.subscribe_live("s1");

        let ev = sessions.append("s1", new_event()).unwrap();

        assert_eq!(rx1.recv().await.unwrap().id, ev.id);
        assert_eq!(rx2.recv().await.unwrap().id, ev.id);
    }

    #[tokio::test]
    async fn a_receiver_that_never_polls_does_not_block_a_fast_one() {
        let (_dir, sessions) = sessions();
        let mut fast = sessions.subscribe_live("s1");
        let mut laggard = sessions.subscribe_live("s1");

        // `fast` drains every append as it happens; `laggard` never polls
        // until after the fill. Filling well past capacity must never make
        // `append` (and therefore the writer's `send`) error.
        for _ in 0..(CHANNEL_CAPACITY + 10) {
            sessions.append("s1", new_event()).unwrap();
            assert!(
                fast.try_recv().is_ok(),
                "a receiver that keeps up must never lag"
            );
        }

        // The laggard, having never polled, has fallen behind the ring
        // buffer and must be told so rather than silently missing events or
        // blocking the writer above.
        let err = loop {
            match laggard.recv().await {
                Err(e) => break e,
                Ok(_) => continue,
            }
        };
        assert!(matches!(err, RecvError::Lagged(_)));
    }

    #[tokio::test]
    #[should_panic(expected = "seq must be 0")]
    async fn broadcast_transient_panics_in_debug_if_seq_is_not_zero() {
        let (_dir, sessions) = sessions();
        let ev = EventEnvelope {
            stream: "s1".to_string(),
            id: "hint-1".to_string(),
            seq: 1,
            ts: "2026-07-27T00:00:00Z".to_string(),
            actor: actor(),
            event_type: "typing".to_string(),
            thread: None,
            parent: None,
            version: 1,
            cost: None,
            data: serde_json::json!({}),
        };
        sessions.broadcast_transient("s1", ev);
    }

    #[tokio::test]
    async fn broadcast_transient_is_never_persisted() {
        let (_dir, sessions) = sessions();
        let ev = EventEnvelope {
            stream: "s1".to_string(),
            id: "hint-1".to_string(),
            seq: 0,
            ts: "2026-07-27T00:00:00Z".to_string(),
            actor: actor(),
            event_type: "typing".to_string(),
            thread: None,
            parent: None,
            version: 1,
            cost: None,
            data: serde_json::json!({}),
        };
        let mut rx = sessions.subscribe_live("s1");
        sessions.broadcast_transient("s1", ev);

        assert_eq!(rx.recv().await.unwrap().event_type, "typing");
        assert_eq!(sessions.store().head_seq("s1").unwrap(), 0, "never stored");
    }
}

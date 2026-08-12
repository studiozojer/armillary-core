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
use tokio::sync::{broadcast, watch};

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

/// The loop's turn-ownership handle (Task 11). A-4 extended to the loop: one
/// turn in flight per stream, enforced by `Sessions::begin_turn` refusing a
/// second claim while this is `Some` in `StreamState`. `cancel` is kept alive
/// here for the turn's whole duration (Task 10's documented semantics: the
/// provider treats a dropped sender the same as an explicit `true`, but this
/// handle holds the sender so `interrupt` has somewhere to send `true` *to*,
/// rather than relying on drop-as-cancel by accident). `generation` is the one
/// uuid minted per turn — not read by `Sessions` itself, just carried so a
/// caller can find out which generation is currently in flight for a stream.
pub struct TurnHandle {
    pub cancel: watch::Sender<bool>,
    #[allow(dead_code)] // read by callers via a future accessor; not yet needed internally
    pub generation: String,
}

/// Per-stream live state.
struct StreamState {
    tx: broadcast::Sender<EventEnvelope>,
    /// `Some` for exactly the lifetime of one turn (A-4 extended to the
    /// loop) — installed by `begin_turn`, cleared by `end_turn`. `interrupt`
    /// is a no-op when this is `None`: no turn running is still 204.
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
/// (attach) against a stream nothing ever created. `TurnInProgress` is A-4
/// extended to the loop (a second `send` while one turn is in flight);
/// `UnknownEvent` is `evict`'s target-not-found case.
#[derive(Debug)]
pub enum SessionError {
    UnknownInstance,
    Log(io::Error),
    TurnInProgress,
    UnknownEvent,
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
            // A-4: a second turn while one is in flight is a conflict, not a
            // server error — the caller can retry once the first completes.
            SessionError::TurnInProgress => {
                (StatusCode::CONFLICT, "turn_in_progress".to_string())
            }
            SessionError::UnknownEvent => {
                (StatusCode::NOT_FOUND, "unknown_event".to_string())
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
        self.append_inner(stream, partial, None)
    }

    /// Append an event that belongs to another event — a tool call or its
    /// answer, linked to the assistant event that owns the batch.
    ///
    /// `parent` has been on the envelope since v0.1, serialized and documented
    /// as a reserved seam, and constructed nowhere. It is what makes batch
    /// membership a *filter*: eviction has to take a whole tool batch or the
    /// stream dies (both halves of a split pair are a measured 400), and a
    /// positional walk from the assistant event breaks as soon as anything in
    /// the batch has already been evicted — which is exactly when the rule is
    /// being asked to work.
    pub fn append_child(
        &self,
        stream: &str,
        parent: &str,
        partial: NewEvent,
    ) -> Result<EventEnvelope, SessionError> {
        self.append_inner(stream, partial, Some(parent.to_string()))
    }

    fn append_inner(
        &self,
        stream: &str,
        partial: NewEvent,
        parent: Option<String>,
    ) -> Result<EventEnvelope, SessionError> {
        // Everything through here is durable by construction — `seq` is
        // `head + 1`, never 0 — so its type belongs on the durable list. The
        // exhaustiveness guard in `projection.rs` compares two hand-maintained
        // lists to each other and therefore cannot see a type absent from
        // both; this is the half that can. `debug_assert` on purpose: loud in
        // development and every test run, compiled out in release, where the
        // projection's `[unhandled event type: …]` remains the honest fallback
        // rather than a lost event.
        debug_assert!(
            crate::log::envelope::DURABLE_TYPES.contains(&partial.event_type.as_str()),
            "appending {:?}, which is not in DURABLE_TYPES — declare it there and give \
             `project_context` an arm, or it reaches the model as an unhandled marker",
            partial.event_type
        );

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
            parent,
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

    /// Claims the turn slot for `stream`, installing `handle`. Fails with
    /// `SessionError::TurnInProgress` (409, A-4) if one is already claimed —
    /// checked and set under the same lock `append` uses, so two concurrent
    /// `send`s against the same stream can never both win this race.
    pub fn begin_turn(&self, stream: &str, handle: TurnHandle) -> Result<(), SessionError> {
        let mut inner = self.inner.lock().unwrap();
        let state = inner
            .entry(stream.to_string())
            .or_insert_with(StreamState::new);
        if state.turn.is_some() {
            return Err(SessionError::TurnInProgress);
        }
        state.turn = Some(handle);
        Ok(())
    }

    /// Clears the turn slot unconditionally. Called once — always, success,
    /// interruption, or failure alike (the loop's `EndTurnGuard` is what
    /// makes "always" hold even across a panic) — so `interrupt` after
    /// completion finds nothing and is a 204 no-op, and a subsequent `send`
    /// is never blocked by a turn that has already ended.
    pub fn end_turn(&self, stream: &str) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(state) = inner.get_mut(stream) {
            state.turn = None;
        }
    }

    /// Whether a turn is currently claimed for `stream`.
    ///
    /// **Live process state, not a log fact** — unlike every other field on the
    /// instance payload this feeds, it is reconstructed from nothing and
    /// survives no restart. That is correct: an engine that restarted mid-turn
    /// has no turn, and reporting `false` is the honest answer rather than a
    /// gap.
    ///
    /// Uses `get`, never `entry`: being asked about a stream must not create a
    /// `StreamState` for it. `begin_turn` and `broadcast_transient` legitimately
    /// create one because they are writing, but a read that allocates would grow the
    /// map by one entry per probe of a nonexistent instance.
    pub fn turn_in_progress(&self, stream: &str) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.get(stream).is_some_and(|state| state.turn.is_some())
    }

    /// Sends the interrupt signal if a turn is currently claimed for
    /// `stream`; otherwise does nothing. Always safe to call — the route
    /// this backs returns 204 either way (interrupt is idempotent).
    ///
    /// S-3: enforceable beats advisory. This `watch` channel is the
    /// enforcement mechanism — the harness halts the loop directly (the
    /// provider's `select!` on `cancel.changed()`, or its next between-
    /// fragment check) rather than asking the model to stop nicely. The
    /// model's only channel is the result handed back (a truncated
    /// `TurnOutcome`, then the durable `interrupt`/`assistant_message`
    /// pair); it does not get a vote on whether the stop happens.
    pub fn interrupt(&self, stream: &str) {
        let inner = self.inner.lock().unwrap();
        if let Some(state) = inner.get(stream) {
            if let Some(handle) = &state.turn {
                // No receiver left (the turn already ended) is not an error
                // here — the same idempotence `interrupt`'s caller relies on.
                let _ = handle.cancel.send(true);
            }
        }
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
            principal: None,
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

    #[test]
    #[should_panic(expected = "not in DURABLE_TYPES")]
    fn appending_a_type_that_was_never_declared_durable_is_a_wiring_bug() {
        // The gap `handled_types_cover_every_durable_type` cannot see. That
        // guard compares two hand-maintained lists to each other, so a type
        // absent from BOTH satisfies it — which is exactly what happened when
        // `composition` was written, appended, and projected while appearing
        // on neither list. The projection's honest degradation meant nothing
        // vanished; it just rendered as `[unhandled event type: composition]`
        // in the model's window, silently, for as long as nobody looked.
        //
        // `debug_assert`, deliberately: loud here and in development, compiled
        // out in release, so a mis-declared type can never cost a real session
        // an event on the phone — there it still degrades honestly.
        let (_dir, sessions) = sessions();
        sessions
            .append(
                "s1",
                NewEvent {
                    actor: actor(),
                    event_type: "sudden_inspiration".to_string(),
                    data: serde_json::json!({}),
                },
            )
            .ok();
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

    fn handle(generation: &str) -> (TurnHandle, watch::Receiver<bool>) {
        let (tx, rx) = watch::channel(false);
        (
            TurnHandle {
                cancel: tx,
                generation: generation.to_string(),
            },
            rx,
        )
    }

    #[tokio::test]
    async fn begin_turn_then_begin_turn_again_is_turn_in_progress() {
        let (_dir, sessions) = sessions();
        let (h1, _rx1) = handle("g1");
        sessions.begin_turn("s1", h1).unwrap();

        let (h2, _rx2) = handle("g2");
        let err = sessions.begin_turn("s1", h2).unwrap_err();
        assert!(matches!(err, SessionError::TurnInProgress));
    }

    #[tokio::test]
    async fn end_turn_clears_the_slot_so_begin_turn_can_claim_it_again() {
        let (_dir, sessions) = sessions();
        let (h1, _rx1) = handle("g1");
        sessions.begin_turn("s1", h1).unwrap();
        sessions.end_turn("s1");

        let (h2, _rx2) = handle("g2");
        assert!(sessions.begin_turn("s1", h2).is_ok());
    }

    #[tokio::test]
    async fn interrupt_with_no_turn_running_is_a_silent_no_op() {
        let (_dir, sessions) = sessions();
        // No panic, no error return type at all — the route this backs is
        // unconditionally 204 whether or not a turn is running.
        sessions.interrupt("s1");
    }

    #[tokio::test]
    async fn interrupt_sends_true_on_the_claimed_handles_watch() {
        let (_dir, sessions) = sessions();
        let (h1, mut rx1) = handle("g1");
        sessions.begin_turn("s1", h1).unwrap();

        sessions.interrupt("s1");

        assert!(rx1.changed().await.is_ok());
        assert!(*rx1.borrow());
    }

    #[tokio::test]
    async fn interrupt_after_end_turn_is_a_no_op_not_a_reused_stale_handle() {
        let (_dir, sessions) = sessions();
        let (h1, mut rx1) = handle("g1");
        sessions.begin_turn("s1", h1).unwrap();
        sessions.end_turn("s1");

        sessions.interrupt("s1");

        // The old handle's cancel sender was dropped along with the cleared
        // slot, so its receiver observes closure, never a stray `true`.
        assert!(rx1.changed().await.is_err());
    }

    #[tokio::test]
    async fn turn_in_progress_tracks_the_claim_and_an_unknown_stream_is_false() {
        let (_dir, sessions) = sessions();
        assert!(!sessions.turn_in_progress("s1"), "no turn claimed yet");

        let (tx, _rx) = tokio::sync::watch::channel(false);
        let h1 = TurnHandle { cancel: tx, generation: "g1".to_string() };
        sessions.begin_turn("s1", h1).unwrap();
        assert!(sessions.turn_in_progress("s1"));

        sessions.end_turn("s1");
        assert!(!sessions.turn_in_progress("s1"));

        // A stream nothing has ever touched must answer false rather than
        // creating a StreamState as a side effect of being asked about.
        assert!(!sessions.turn_in_progress("never-seen"));
    }
}

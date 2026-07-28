//! `GET /streams/{stream}/events?from=N` — constitution A-1's
//! `subscribe(stream, from_seq)` made real over HTTP: replay durable events
//! past the client's cursor, then tail live, with the replay/tail race
//! closed by a dedup at the seam.
//!
//! SSE is this engine's transport BINDING, not the standard's — constitution
//! §5 explicitly defers the transport question (Connect-RPC vs SSE-split);
//! nothing here resolves that, it just picks one for this engine.
//!
//! **The correctness order (this is the design, not an implementation
//! detail):**
//! 1. `subscribe_live` FIRST — broadcast buffering begins before the durable
//!    read below, so no event appended in between is ever missed by the live
//!    channel.
//! 2. Read the replay batch (`read_from`) via `blocking::run` (filesystem
//!    work).
//! 3. Emit `gap` if the client's cursor is behind what the store still has,
//!    then the replay envelopes, then `caught-up`.
//! 4. Drain/tail the receiver, skipping any DURABLE event with
//!    `seq <= last_replayed` — the dedup that closes the window opened by
//!    doing (1) before (2): an event appended in that window lands in BOTH
//!    the replay batch and the live channel, and this is where the second
//!    copy is dropped. A transient (`seq == 0`) always passes through. This
//!    also means a transient broadcast during that same window (before the
//!    tail starts polling — i.e. anywhere from `subscribe_live` through the
//!    end of emitting `caught-up`) is simply delivered *later*, once the
//!    tail starts draining, rather than lost or reordered ahead of the
//!    replay: safe precisely because I-4 requires a transient's payload be a
//!    snapshot, not an increment, so a delayed delivery carries the same
//!    information a prompt one would have.
//! 5. Tail forever; on `RecvError::Lagged`, END the stream (A-2: the slow
//!    client drops itself — its cursor plus a fresh replay on reconnect
//!    heals whatever the channel dropped, so ending here is not data loss,
//!    it is where the loss gets healed).

use crate::blocking;
use crate::log::envelope::EventEnvelope;
use crate::log::store::LogStore;
use crate::sessions::SessionError;
use crate::state::SharedState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
};
use futures_util::stream::{self, Stream, StreamExt};
use serde::Deserialize;
use std::convert::Infallible;
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;

#[derive(Deserialize)]
pub struct SubscribeQuery {
    from: Option<String>,
}

/// Renders a durable-or-transient envelope as a named `envelope` SSE frame.
/// `EventEnvelope` is plain data (strings, numbers, an already-parsed
/// `serde_json::Value`) — serialization failing here would mean the type
/// itself is broken, not that this request did anything wrong.
fn envelope_event(ev: &EventEnvelope) -> Event {
    Event::default()
        .event("envelope")
        .data(serde_json::to_string(ev).expect("EventEnvelope always serializes"))
}

/// All filesystem work for one subscribe's replay half: the store's
/// earliest/head seq (to decide `gap` and to report `caught-up`'s
/// `headSeq`), plus the batch of durable events at or past `replay_from`.
fn replay_snapshot(
    store: &LogStore,
    stream: &str,
    replay_from: u64,
) -> Result<(u64, u64, Vec<EventEnvelope>), SessionError> {
    let earliest_seq = store.earliest_seq(stream)?;
    let head_seq = store.head_seq(stream)?;
    let replay = store.read_from(stream, replay_from)?;
    Ok((earliest_seq, head_seq, replay))
}

/// The tail half of subscribe: drains `rx` forever, skipping any DURABLE
/// event (`seq != 0`) with `seq <= last_replayed` — the dedup that closes
/// the window opened by running `subscribe_live` (buffering begins) before
/// the durable replay read (what `last_replayed` summarizes): an event
/// appended in that window lands in BOTH the replay batch and the broadcast
/// channel, and this is where the second copy is dropped rather than
/// re-delivered. A transient (`seq == 0`) always passes through untouched.
/// Ends the stream on `RecvError::Lagged` (A-2: the slow client drops
/// itself here — its cursor plus a fresh replay on reconnect heals whatever
/// the channel dropped) or `RecvError::Closed`.
///
/// Kept separate from `envelope_event`'s `Event`-rendering step, and
/// yielding the domain type (`EventEnvelope`) rather than `axum::Event`, on
/// purpose: `Event` has no public accessor to read a name/payload back out
/// once built, so a dedup expressed over `EventEnvelope` is what makes this
/// the race-closing logic — not the whole handler — unit-testable directly,
/// without going through axum's wire format or a real socket at all.
fn tail_envelopes(
    rx: broadcast::Receiver<EventEnvelope>,
    last_replayed: u64,
) -> impl Stream<Item = EventEnvelope> {
    stream::unfold((rx, last_replayed), |(mut rx, last_replayed)| async move {
        loop {
            match rx.recv().await {
                Ok(ev) if ev.seq != 0 && ev.seq <= last_replayed => continue,
                Ok(ev) => return Some((ev, (rx, last_replayed))),
                Err(RecvError::Lagged(_)) => return None,
                Err(RecvError::Closed) => return None,
            }
        }
    })
}

pub async fn subscribe(
    State(state): State<SharedState>,
    Path(stream): Path<String>,
    Query(query): Query<SubscribeQuery>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    // `from` is the client's cursor — the last seq it has already seen, 0
    // meaning "nothing yet" — so the first seq it actually needs is `from + 1`.
    let from: u64 = match query.from.as_deref() {
        None => 0,
        Some(s) => s
            .parse()
            .map_err(|_| (StatusCode::BAD_REQUEST, "bad_from".to_string()))?,
    };

    // Unknown stream is 404 BEFORE the SSE response starts, and checked
    // before `subscribe_live` runs at all — a typo'd stream name should never
    // leave a phantom broadcast channel behind in `Sessions`. A stream exists
    // iff the store has it (v0: nothing subscribes before something durable
    // has been appended), so `store().streams()` is the whole check.
    let sessions = state.sessions.clone();
    let known_stream = stream.clone();
    let known = blocking::run(move || {
        sessions
            .store()
            .streams()
            .map(|names| names.contains(&known_stream))
            .map_err(|e| SessionError::Log(e).into_response())
    })
    .await?;
    if !known {
        return Err((StatusCode::NOT_FOUND, "unknown_stream".to_string()));
    }

    // (1) `subscribe_live` FIRST — buffering begins now, before the durable
    // read below, so nothing appended in between this line and that read is
    // ever missed by the live channel (it may also land in the replay batch
    // — step 4's dedup is what makes that safe).
    let rx = state.sessions.subscribe_live(&stream);

    // (2) the replay batch, via `blocking::run` (filesystem work).
    let sessions = state.sessions.clone();
    let replay_stream = stream.clone();
    // `saturating_add`: an adversarial `from=18446744073709551615` (u64::MAX)
    // must fall out as an ordinary empty replay, not panic the request task.
    let replay_from = from.saturating_add(1);
    let (earliest_seq, head_seq, replay) = blocking::run(move || {
        replay_snapshot(sessions.store(), &replay_stream, replay_from)
            .map_err(SessionError::into_response)
    })
    .await?;

    // (3) `gap` (if the cursor is behind what the store still has), then the
    // replay envelopes, then `caught-up`.
    let mut prefix: Vec<Result<Event, Infallible>> = Vec::new();
    if replay_from < earliest_seq {
        let gap = serde_json::json!({
            "requestedFrom": from,
            "earliestAvailable": earliest_seq,
        });
        prefix.push(Ok(Event::default().event("gap").data(gap.to_string())));
    }

    let mut last_replayed = from;
    for ev in &replay {
        last_replayed = ev.seq;
        prefix.push(Ok(envelope_event(ev)));
    }

    let caught_up = serde_json::json!({ "headSeq": head_seq });
    prefix.push(Ok(Event::default()
        .event("caught-up")
        .data(caught_up.to_string())));

    // (4) + (5): the dedup-then-tail-forever half — see `tail_envelopes`.
    let tail = tail_envelopes(rx, last_replayed).map(|ev| Ok(envelope_event(&ev)));

    Ok(Sse::new(stream::iter(prefix).chain(tail)).keep_alive(KeepAlive::default()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::envelope::{Actor, Role};
    use futures_util::StreamExt;

    fn envelope(seq: u64, event_type: &str) -> EventEnvelope {
        EventEnvelope {
            stream: "s1".to_string(),
            id: format!("id-{seq}"),
            seq,
            ts: "2026-07-27T00:00:00Z".to_string(),
            actor: Actor {
                role: Role::User,
                instance: None,
            },
            event_type: event_type.to_string(),
            thread: None,
            parent: None,
            version: 1,
            cost: None,
            data: serde_json::json!({}),
        }
    }

    /// The exact scenario the dedup exists for: the replay batch already
    /// covered seq 1..=4 (`last_replayed = 4`), but — because
    /// `subscribe_live` starts buffering before that durable read even runs
    /// — the broadcast channel ALSO holds 3 and 4, delivered a second time
    /// by the writer's `append`. `tail_envelopes` must drop the two
    /// already-replayed duplicates, deliver the genuinely-new seq 5, and let
    /// a transient (seq 0) pass through untouched, in arrival order.
    #[tokio::test]
    async fn skips_durable_duplicates_already_covered_by_replay_but_passes_new_ones_and_transients()
    {
        let (tx, rx) = broadcast::channel(16);
        tx.send(envelope(3, "user_message")).unwrap();
        tx.send(envelope(4, "user_message")).unwrap();
        tx.send(envelope(5, "user_message")).unwrap();
        tx.send(envelope(0, "typing")).unwrap();
        // Dropping the sender does not truncate what's already buffered —
        // `recv` still drains 5 and the transient before it sees `Closed`
        // and the stream ends, which is what lets a plain `collect` work
        // here without a separate "stop after N" signal.
        drop(tx);

        let out: Vec<EventEnvelope> = tail_envelopes(rx, 4).collect().await;

        let seqs: Vec<u64> = out.iter().map(|e| e.seq).collect();
        assert_eq!(
            seqs,
            vec![5, 0],
            "3 and 4 must be skipped as already-replayed duplicates; 5 and the transient pass"
        );
        assert_eq!(out[1].event_type, "typing", "the transient passes through untouched");
    }

    #[tokio::test]
    async fn a_transient_is_never_skipped_even_at_seq_zero_against_a_zero_last_replayed() {
        // Guards against a subtly wrong condition like `seq < last_replayed`
        // that would happen to also pass 0 through when last_replayed is 0 —
        // this pins that the pass-through is because `seq == 0`, not because
        // of what `last_replayed` happens to be.
        let (tx, rx) = broadcast::channel(4);
        tx.send(envelope(0, "typing")).unwrap();
        drop(tx);

        let out: Vec<EventEnvelope> = tail_envelopes(rx, 0).collect().await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].seq, 0);
    }

    #[tokio::test]
    async fn ends_the_stream_on_lagged_rather_than_skipping_past_it() {
        let (tx, rx) = broadcast::channel(2);
        // Overflow the tiny buffer before anything ever polls `rx`, so the
        // first `recv` sees `Lagged`, not the events themselves.
        for seq in 1..=5u64 {
            tx.send(envelope(seq, "user_message")).unwrap();
        }

        let out: Vec<EventEnvelope> = tail_envelopes(rx, 0).collect().await;
        assert!(
            out.is_empty(),
            "A-2: Lagged ends the stream immediately rather than replaying what's left"
        );
    }
}

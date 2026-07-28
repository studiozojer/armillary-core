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
//!    copy is dropped. A transient (`seq == 0`) always passes through.
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
    let replay_from = from + 1;
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

    // (4) + (5): tail forever, skipping any durable event this subscriber
    // already saw in replay (`seq <= last_replayed` — the dedup that closes
    // the race); a transient (`seq == 0`) always passes through. On
    // `RecvError::Lagged`, end the stream — A-2: the slow client drops
    // itself here, and its cursor plus a fresh replay on reconnect heals
    // whatever the channel dropped, so this is not data loss, it's where the
    // loss gets healed. `RecvError::Closed` cannot happen while `Sessions`
    // holds the sender, but is handled the same way defensively.
    let tail = stream::unfold((rx, last_replayed), |(mut rx, last_replayed)| async move {
        loop {
            match rx.recv().await {
                Ok(ev) if ev.seq != 0 && ev.seq <= last_replayed => continue,
                Ok(ev) => return Some((Ok(envelope_event(&ev)), (rx, last_replayed))),
                Err(RecvError::Lagged(_)) => return None,
                Err(RecvError::Closed) => return None,
            }
        }
    });

    Ok(Sse::new(stream::iter(prefix).chain(tail)).keep_alive(KeepAlive::default()))
}

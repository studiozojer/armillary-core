use axum::http::StatusCode;

/// Run filesystem work on a thread that is allowed to block.
///
/// # Why this exists
///
/// The engine is deliberately loopless today, and every route is a short read —
/// so `std::fs` called straight from an `async fn` costs nothing measurable and
/// nothing breaks. That changes the moment the roadmap lands.
///
/// `subscribe(stream, from_seq)` (constitution A-1) holds a connection open for
/// the life of a session. Those connections are parked on tokio's worker
/// threads, of which there are roughly one per core. A blocking `std::fs` call
/// does not yield that thread — it freezes it until the disk answers. So one
/// `/tree` over a directory whose entries symlink to a disconnected volume
/// blocks for the mount timeout, and **every live subscriber sharing that
/// worker stops receiving events**, silently, along with `/health` — the one
/// signal the deploy runbook names as trustworthy.
///
/// `spawn_blocking` moves the work to a pool sized for exactly this. Preferred
/// over `tokio::fs`, which spawns per syscall and is worse for directory walks.
///
/// Done now rather than when it hurts, because the habit is what propagates:
/// four handlers today, copied into the log routes tomorrow, and by then the
/// bug only appears when a disk is slow and someone happens to be listening.
pub async fn run<T, F>(work: F) -> Result<T, (StatusCode, String)>
where
    F: FnOnce() -> Result<T, (StatusCode, String)> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(work).await {
        Ok(result) => result,
        // The closure panicked. Surface it as a server error rather than
        // letting the connection hang.
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error".to_string(),
        )),
    }
}

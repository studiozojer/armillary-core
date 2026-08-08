//! Integration tests for the CLI binary ITSELF — spawns the real, compiled
//! `armillary-engine` executable (`CARGO_BIN_EXE_armillary-engine`, which
//! Cargo sets for every integration test binary) rather than calling any
//! function inside it.
//!
//! # Why these exist
//!
//! `main()` decides, in three lines, whether the bare `--root <path>` form
//! serves or runs a subcommand:
//!
//! ```ignore
//! if let Some(cmd) = &args.command {
//!     return run_command(cmd);
//! }
//! ```
//!
//! Nothing in the unit-test suite can reach those three lines — `main()`
//! is not a function any test calls, by construction (it is `async fn
//! main()`, wired to `#[tokio::main]`). `main.rs`'s own
//! `the_bare_form_still_serves` only exercises `Args::try_parse_from`,
//! which is clap's job, one layer up from the dispatch this file guards.
//! A mutation that inverted the dispatch above — routing the BARE form to
//! a subcommand instead of serve, which on the two launchd hosts means the
//! service silently stops serving — left the entire unit suite green,
//! including that test, because none of it ever runs the compiled binary
//! and watches what it actually does. These two tests close exactly that
//! gap.
//!
//! # Why a fake `HOME`
//!
//! The bare form's `main()`, on a successful bind, mints the `host`
//! principal via `ensure_host(&default_registry_dir(), ...)` —
//! `default_registry_dir()` reads `$HOME` directly. Every spawn below sets
//! `HOME` to a fresh tempdir so these tests never touch this machine's
//! real `~/.config/armillary/devices` or `~/.config/armillary/host-token`
//! (the same reason Task 4's brief Step 5 hand-walk was skipped: those
//! paths are real, shared, machine state, not a test fixture).

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

fn armillary_bin() -> &'static str {
    env!("CARGO_BIN_EXE_armillary-engine")
}

/// Forward `reader`'s lines onto a channel from a background thread, so the
/// test thread can wait for a specific line with a bounded timeout instead
/// of blocking forever on `read_line` against a child that never writes
/// one (a hang in the child must fail this test, not hang the suite).
fn stream_lines(mut reader: impl BufRead + Send + 'static) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break, // EOF or the pipe broke — child closed stdout.
                Ok(_) => {
                    if tx.send(line.trim_end().to_string()).is_err() {
                        break; // Receiver dropped (test already returned) — stop reading.
                    }
                }
            }
        }
    });
    rx
}

/// Poll `child` for exit with a bounded total wait, never `wait()`'s
/// unbounded block. `None` means still running when the deadline passed.
fn wait_bounded(child: &mut Child, timeout: Duration) -> Option<std::process::ExitStatus> {
    let start = Instant::now();
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Some(status);
        }
        if start.elapsed() >= timeout {
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Owns a `Child` and kills + reaps it on drop, ignoring errors — runs on
/// EVERY exit path (success, a failed `assert!`, or any other panic,
/// because `drop` still runs while a panic unwinds) so a failed assertion
/// in one test never leaks a server process holding a port, which would
/// make the NEXT run fail for an unrelated-looking reason.
struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn the_bare_form_serves() {
    let workspace = tempfile::tempdir().unwrap();
    let fake_home = tempfile::tempdir().unwrap();

    let mut child = KillOnDrop(
        Command::new(armillary_bin())
            .arg("--root")
            .arg(workspace.path())
            .arg("--bind")
            .arg("127.0.0.1")
            .arg("--port")
            .arg("0")
            .env("HOME", fake_home.path())
            .env_remove("ANTHROPIC_API_KEY")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn armillary-engine"),
    );

    let stdout = BufReader::new(child.0.stdout.take().unwrap());
    let rx = stream_lines(stdout);

    // main.rs: `println!("armillary-engine serving {} on http://{}", ...)`,
    // printed only after `TcpListener::bind` succeeds — the one line that
    // means "the bare form actually started serving," not just "parsed."
    let deadline = Duration::from_secs(10);
    let start = Instant::now();
    let mut saw_serving = false;
    while start.elapsed() < deadline {
        match rx.recv_timeout(deadline.saturating_sub(start.elapsed())) {
            Ok(line) => {
                if line.starts_with("armillary-engine serving ") && line.contains("on http://") {
                    saw_serving = true;
                    break;
                }
            }
            Err(_) => break, // timed out, or the channel closed (child exited without serving)
        }
    }

    assert!(
        saw_serving,
        "expected the \"armillary-engine serving ...\" line on stdout within {deadline:?} \
         — the bare form must serve, not do something else"
    );
    // Confirm it's still up, not that it printed the line and then died.
    assert!(
        child.0.try_wait().unwrap().is_none(),
        "the bare form must still be running after printing the serving line"
    );
}

#[test]
fn a_subcommand_does_not_serve() {
    let workspace = tempfile::tempdir().unwrap();
    let fake_home = tempfile::tempdir().unwrap();

    let mut child = KillOnDrop(
        Command::new(armillary_bin())
            .arg("--root")
            .arg(workspace.path())
            .arg("principals")
            .env("HOME", fake_home.path())
            .env_remove("ANTHROPIC_API_KEY")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn armillary-engine principals"),
    );

    let stdout = BufReader::new(child.0.stdout.take().unwrap());
    let rx = stream_lines(stdout);

    let deadline = Duration::from_secs(10);
    let start = Instant::now();
    let mut lines = Vec::new();
    // Drains every line until EOF (child closed stdout) or the wait budget
    // runs out — once `deadline` has elapsed, `saturating_sub` yields zero
    // and `recv_timeout` returns immediately, ending the loop.
    while let Ok(line) = rx.recv_timeout(deadline.saturating_sub(start.elapsed())) {
        lines.push(line);
    }

    let status = wait_bounded(&mut child.0, Duration::from_secs(5))
        .unwrap_or_else(|| panic!("`principals` did not exit within the bound — a subcommand must not hang, let alone serve; lines so far: {lines:?}"));

    assert!(status.success(), "principals should exit 0, got {status:?}; lines: {lines:?}");
    assert!(
        !lines.iter().any(|l| l.starts_with("armillary-engine serving ")),
        "a subcommand must not print the serving line: {lines:?}"
    );
}

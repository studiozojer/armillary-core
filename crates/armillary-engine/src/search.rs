//! Finding things — the two verbs that answer "where is it" rather than
//! "what does it say".
//!
//! **The domain is declared, not discovered** (SD-2). Roots come from the
//! manifest, so `repos/external/` — 69% of this workspace's files, five other
//! harnesses' source — is excluded by nobody declaring it rather than by an
//! exclusion rule someone maintains. That makes this the first substrate verb
//! whose reach is decided by the composition.

use crate::tools::ToolError;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use walkdir::WalkDir;

/// The roots a default-domain search walks, with the two kinds counted apart.
///
/// **The counts are separate because the footer names them separately.** A
/// root is either a declared module (operator, commons, repo) or a file named
/// by `[router] contains`, and `get_composition` reports only the first kind.
/// Rendering one total under the word "modules" gave a model two
/// engine-authored answers to the same question — 23 from `get_composition`,
/// 30 from `search` — with nothing to reconcile them.
pub(crate) struct Roots {
    pub paths: Vec<PathBuf>,
    /// How many of `paths` came from a declared operator, commons, or repo.
    pub modules: usize,
    /// How many came from `[router] contains`.
    pub router_files: usize,
}

/// The directories and files a default-domain search walks.
///
/// **C-1 as running code:** declared, not discovered. Nothing here enumerates
/// a directory to find out what a workspace contains — it asks the same parser
/// `get_composition` asks, so the two cannot drift into disagreeing about what
/// this workspace is.
///
/// Each declared path goes through the guard, which canonicalises it and
/// judges the canonical result. A path that escapes, is denied, or is simply
/// not on this machine is **skipped rather than fatal** (C-4): a manifest
/// naming a repo that was never cloned here is the ordinary case.
pub(crate) fn search_roots(root: &Path) -> Result<Roots, ToolError> {
    let composition = armillary_composition::parse_workspace(root)
        .map_err(|e| ToolError::new("composition_unreadable", e.to_string()))?;

    let mut out = Roots { paths: Vec::new(), modules: 0, router_files: 0 };

    for m in composition
        .operators
        .iter()
        .chain(&composition.commons)
        .chain(&composition.repos)
    {
        if push_root(root, &m.path, &mut out.paths) {
            out.modules += 1;
        }
    }
    for f in &composition.router.contains {
        if push_root(root, f, &mut out.paths) {
            out.router_files += 1;
        }
    }

    Ok(out)
}

/// Resolve one declared path and add it if it survives the guard. Returns
/// whether it was actually added, so the caller's tally counts roots that
/// exist rather than declarations that were made.
///
/// A free function rather than a closure over `out` on purpose: a closure
/// holding `&mut out` across both loops and then returning `out` is a
/// borrow-checker argument nobody needs to have.
fn push_root(root: &Path, rel: &str, out: &mut Vec<PathBuf>) -> bool {
    if let Ok(p) = crate::guard::resolve(root, rel) {
        // Canonical, so two declarations reaching the same directory through
        // different spellings collapse to one root rather than producing
        // every match twice.
        if !out.contains(&p) {
            out.push(p);
            return true;
        }
    }
    false
}

/// How a call's domain was arrived at — which is what its footer must say.
#[derive(Debug)]
pub(crate) enum Scope {
    /// The caller named a path; the domain is exactly that one.
    Explicit,
    /// The default domain: what the manifest declares.
    Composed { modules: usize, router_files: usize },
}

#[derive(Debug)]
pub(crate) struct Domain {
    pub paths: Vec<PathBuf>,
    pub scope: Scope,
}

/// The domain for one call: the composed roots, or the one path the caller named.
///
/// **SD-3.** An explicit `path` overrides the default domain and reaches
/// anywhere `read_file` can, which is what makes SD-2's narrowing safe — a
/// file you can read but not search would otherwise be a trap.
///
/// An **empty** `path` is treated as absent, deliberately. `list_directory`
/// uses `""` for the workspace root, so a model has learned that spelling and
/// will send it here meaning "no scope"; resolving it to the root would
/// silently search every external clone — the opposite of the request.
pub(crate) fn resolve_domain(root: &Path, path: Option<&str>) -> Result<Domain, ToolError> {
    match path {
        Some(p) if !p.is_empty() => Ok(Domain {
            paths: vec![crate::guard::resolve(root, p)?],
            scope: Scope::Explicit,
        }),
        _ => {
            let roots = search_roots(root)?;
            Ok(Domain {
                paths: roots.paths,
                scope: Scope::Composed {
                    modules: roots.modules,
                    router_files: roots.router_files,
                },
            })
        }
    }
}

/// The wall-clock cap on one search.
///
/// The `regex` crate is linear in input length, so this does not guard the
/// pattern — it guards the **disk**. The expected failure is a slow or
/// sleeping volume, not an exotic one, and an uncapped walk would hold a
/// blocking-pool thread until the filesystem answered on its own schedule.
/// Same reasoning as `git.rs`'s `DEFAULT_TIMEOUT`.
pub(crate) const SEARCH_BUDGET: Duration = Duration::from_secs(10);

pub(crate) struct WalkStats {
    /// Files handed to `visit`: everything under the roots that survived the
    /// **safety** gate. Deliberately not narrowed by `is_openable` — see the
    /// note on `walk` — so a caller that opens files must subtract its own
    /// refusals before printing this next to the word "searched".
    pub files: usize,
    pub timed_out: bool,
    /// The visitor ended the walk before the roots were exhausted.
    pub stopped_short: bool,
}

impl WalkStats {
    /// Whether `files` is the whole domain or merely how far we got.
    ///
    /// A count taken from an interrupted walk is not a denominator, and a
    /// footer that prints it under the word "searched" invites it to be read
    /// as one.
    pub fn complete(&self) -> bool {
        !self.timed_out && !self.stopped_short
    }
}

/// Walk the search roots, handing every file the safety gate allows to `visit`.
///
/// `visit` receives the absolute path and the workspace-relative path, and
/// returns `false` to stop the walk immediately — that is how a cap ends a
/// search rather than filling a 50-match budget and then walking 7,700 more
/// files to no purpose.
///
/// **Two gates, and only one of them is here.** `is_hidden_from_listings`
/// (credentials, `node_modules`, `target`, `.build`, `.worktrees`,
/// `.armillary`) is a safety and noise rule that governs whether a path is
/// served at all, so it prunes the walk. `is_openable` is a **content** gate —
/// it decides whether bytes may be handed out — and it lives in the visitor of
/// whichever caller actually reads a file. `guard::is_openable`'s own doc says
/// this: it governs opening, and hiding what cannot be opened from a *listing*
/// reintroduces the projection that gate was changed to remove. `find_files`
/// is a listing; applying the content gate here made `find_files("**/*.png")`
/// answer "there are none" about files sitting in plain view.
///
/// **Entries are visited in sorted order** (`sort_by_file_name`), roots in
/// declaration order. Without it the filesystem decides, so two identical
/// calls could return different results and — once a cap bites — a different
/// *set*.
///
/// **Symlinks are not followed.** Following one would risk a cycle, and — for
/// the common case, a symlink whose target already sits inside a declared
/// root — would return a second copy of a file already walked at its real
/// path.
///
/// That justification does not cover every case, and this is the part worth
/// being honest about: nothing here checks that a symlink's target is
/// actually reachable some other way. A symlink inside a declared root whose
/// target lies **outside every declared root** is simply skipped, and its
/// content drops out of search silently — no error, no note, nothing to
/// distinguish it from content that was never there. That is a known
/// limitation of scoping the domain to the manifest, not a property this
/// function guarantees.
pub(crate) fn walk(
    root: &Path,
    roots: &[PathBuf],
    deadline: Instant,
    visit: &mut dyn FnMut(&Path, &str) -> bool,
) -> WalkStats {
    let mut stats = WalkStats { files: 0, timed_out: false, stopped_short: false };

    // Entries arrive from canonical roots, so the prefix stripped from them
    // must be canonical too — otherwise every relative path silently keeps a
    // `/private` or a symlinked segment and no glob written by a model matches.
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());

    for start in roots {
        let walker = WalkDir::new(start)
            .follow_links(false)
            // Determinism, and the reason it is not cosmetic: a cap keeps the
            // first N *visited*, so an unordered walk makes the surviving set
            // itself vary between two identical calls.
            .sort_by_file_name()
            .into_iter()
            // `filter_entry` skips a whole subtree on false, so `node_modules`
            // costs one stat rather than a descent into it. Depth 0 is the
            // declared root itself and is always entered — a module could
            // legitimately be named something the guard would otherwise judge.
            .filter_entry(|e| {
                e.depth() == 0
                    || !crate::guard::is_hidden_from_listings(&e.file_name().to_string_lossy())
            });

        for entry in walker.flatten() {
            if Instant::now() >= deadline {
                stats.timed_out = true;
                return stats;
            }
            // With `follow_links(false)` a symlink is neither file nor dir
            // here, so it is skipped — see the note above.
            if !entry.file_type().is_file() {
                continue;
            }
            let Ok(rel) = entry.path().strip_prefix(&canonical_root) else {
                continue;
            };
            let rel = rel.to_string_lossy().to_string();
            stats.files += 1;
            if !visit(entry.path(), &rel) {
                stats.stopped_short = true;
                return stats;
            }
        }
    }

    stats
}

/// The most paths one `find_files` result may carry.
///
/// Generous next to the match caps because a path is cheap — roughly 60 bytes
/// against a 200-character match window.
pub(crate) const MAX_PATHS: usize = 100;

/// Exclusions no `path` argument can lift.
///
/// **The distinction this draws is the only one that matters to the reader.**
/// Undeclared content is *reachable if you name it*; these are *refused
/// however you name them* — `guard::resolve` denies them at any depth, so
/// `search("x", path="repos/engine/.worktrees/feat")` comes back
/// `denied_noise`. The sentence this replaced lumped worktrees in with
/// undeclared content and offered `path` as the recovery for both, which
/// promised a repair the guard refuses. Worktrees are not undeclared at all:
/// they sit *inside* declared roots and are excluded by a different mechanism.
///
/// Every name here is pinned to that behaviour by
/// `the_note_promises_a_recovery_only_where_one_exists`.
const NEVER_SEARCHED: &str = "Credentials, build and dependency trees \
     (node_modules, target, .build), git worktrees (.worktrees), and the \
     engine's own .armillary are never searched at any path — naming one is \
     refused, not widened.";

fn plural(n: usize, one: &str, many: &str) -> String {
    if n == 1 {
        format!("{n} {one}")
    } else {
        format!("{n} {many}")
    }
}

/// The sentence every result ends with, naming what was and was not searched.
///
/// **The whole absence-vs-refusal lesson lives here.** SD-2 deliberately
/// narrows the domain; without this, that narrowing reads exactly like an
/// empty workspace, and the model has no way to learn otherwise. The 2026-08-01
/// sync work shipped the opposite of this — `verdict()` returned `current` for
/// twenty-four repos having contacted nothing.
///
/// Three things it must get right, each of which it once got wrong:
///
/// - **The noun matches what is counted** — roots, split into modules and
///   router files, because `get_composition` counts only the former.
/// - **The verb matches how the walk ended.** `files` from an interrupted
///   walk is how far we got, not the size of the domain.
/// - **Both kinds of exclusion are named, and on the scoped path too.** Under
///   an explicit `path` the guard still prunes credentials, `node_modules`,
///   worktrees and the rest, so a scoped result that named nothing let
///   `search("apiKey", path="repos/app")` render "not there" about an answer
///   sitting in `node_modules`.
fn domain_note(scope: &Scope, files: usize, complete: bool) -> String {
    let where_ = match scope {
        Scope::Explicit => "1 path named explicitly".to_string(),
        Scope::Composed { modules, router_files } => format!(
            "{} ({}, {})",
            plural(modules + router_files, "declared root", "declared roots"),
            plural(*modules, "module", "modules"),
            plural(*router_files, "router file", "router files"),
        ),
    };

    let head = if complete {
        format!("searched {where_}, {}", plural(files, "file", "files"))
    } else {
        format!(
            "stopped early — {} walked so far across {where_}, which is not the size of the domain",
            plural(files, "file", "files")
        )
    };

    match scope {
        Scope::Explicit => format!("[{head}. {NEVER_SEARCHED}]\n"),
        Scope::Composed { .. } => format!(
            "[{head}. Content the manifest does not declare — repos/external/, for one — \
             was not searched; pass `path` to search it directly. {NEVER_SEARCHED}]\n"
        ),
    }
}

/// Find files whose workspace-relative path matches a glob.
///
/// **A listing, not a read.** It opens nothing, so the content gate
/// (`is_openable`) does not apply: a `.png` or a `.se1` matching the pattern is
/// reported. The alternative — inheriting `search`'s gate because the two share
/// a walker — made this verb answer "no files matching `**/*.png`" under a
/// footer claiming thousands of files were examined.
pub(crate) fn find_files(
    root: &Path,
    pattern: &str,
    path: Option<&str>,
) -> Result<String, ToolError> {
    let glob = globset::Glob::new(pattern)
        .map_err(|e| ToolError::new("invalid_pattern", e.to_string()))?
        .compile_matcher();

    let domain = resolve_domain(root, path)?;
    let deadline = Instant::now() + SEARCH_BUDGET;

    let mut found: Vec<String> = Vec::new();
    let mut capped = false;
    let stats = walk(root, &domain.paths, deadline, &mut |_abs, rel| {
        if glob.is_match(rel) {
            found.push(rel.to_string());
            if found.len() >= MAX_PATHS {
                capped = true;
                return false;
            }
        }
        true
    });

    found.sort();

    let mut out = String::new();
    if found.is_empty() {
        out.push_str(&format!("no files matching `{pattern}`\n"));
    } else {
        out.push_str(&format!("{} files matching `{pattern}`\n", found.len()));
        for f in &found {
            out.push_str(f);
            out.push('\n');
        }
    }
    if capped {
        // Naming *which* paths survived, not just how many. They are the first
        // hundred the walk reached — declared roots in manifest order, entries
        // sorted within each — and the list is then sorted for display. A
        // sorted list running `a…` to `z…` otherwise reads as complete.
        out.push_str(&format!(
            "[stopped at {MAX_PATHS} paths: the first {MAX_PATHS} the walk reached, \
             sorted for display — not the alphabetically first {MAX_PATHS}, and not all \
             that match. Narrow with `path` or a tighter pattern.]\n"
        ));
    }
    if stats.timed_out {
        out.push_str("[the search budget expired; results are partial]\n");
    }
    // `stats.files` is exactly what this verb examined: every walked path was
    // tested against the glob, because nothing here refuses a file for its type.
    out.push_str(&domain_note(&domain.scope, stats.files, stats.complete()));
    Ok(out)
}

/// The most characters one match line may carry.
pub(crate) const MATCH_WINDOW_CHARS: usize = 200;
/// The most matches one result may carry, across all files.
pub(crate) const MAX_MATCHES: usize = 50;
/// The most matches any single file may contribute.
pub(crate) const MAX_MATCHES_PER_FILE: usize = 5;

/// A window of at most `MATCH_WINDOW_CHARS` centred on the match.
///
/// **Characters, never bytes.** `.爻` is three bytes per character and this
/// workspace writes node files in it, so a byte-indexed cut lands
/// mid-character and the line renders `U+FFFD`. Everything below counts and
/// slices by `char`.
///
/// The measured reason this exists: prose here has no hard wrapping, so a
/// "line" is a paragraph — p99 594 characters, max 5,798 in one `BOARD.md`
/// entry. Returning whole lines would put no ceiling on a durable, re-projected
/// tool result.
fn window(line: &str, match_start_byte: usize, match_end_byte: usize) -> String {
    let total = line.chars().count();
    if total <= MATCH_WINDOW_CHARS {
        return line.to_string();
    }

    let m_start = line[..match_start_byte].chars().count();
    let m_end = line[..match_end_byte].chars().count();

    // Centre on the match. When the match is itself longer than the window,
    // `saturating_sub` collapses the context to zero and the head is kept —
    // the head is where the model's own query is.
    let context = MATCH_WINDOW_CHARS.saturating_sub(m_end - m_start) / 2;
    let mut start = m_start.saturating_sub(context);
    let end = (start + MATCH_WINDOW_CHARS).min(total);
    // Re-anchor so a match near the end of the line still fills the window.
    start = end.saturating_sub(MATCH_WINDOW_CHARS);

    let body: String = line.chars().skip(start).take(end - start).collect();
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.push_str(body.trim());
    if end < total {
        out.push('…');
    }
    out
}

/// Search file contents for a regex, returning windowed match lines.
///
/// The pattern is a regex via the `regex` crate, whose linear-time guarantee
/// is the only reason a model-supplied pattern is safe to compile at all —
/// there is no backtracking to blow up.
pub(crate) fn search(
    root: &Path,
    query: &str,
    path: Option<&str>,
    case_sensitive: bool,
) -> Result<String, ToolError> {
    let re = regex::RegexBuilder::new(query)
        .case_insensitive(!case_sensitive)
        .build()
        .map_err(|e| ToolError::new("invalid_pattern", e.to_string()))?;

    let domain = resolve_domain(root, path)?;
    let deadline = Instant::now() + SEARCH_BUDGET;

    let mut hits: Vec<(String, Vec<(usize, String)>)> = Vec::new();
    let mut total = 0usize;
    let mut capped = false;
    let mut skipped = 0usize;
    let mut not_text = 0usize;
    let mut per_file_capped = 0usize;

    let stats = walk(root, &domain.paths, deadline, &mut |abs, rel| {
        // **The content gate, and this is the only place it belongs.** `walk`
        // hands over every file the safety gate allows, because `find_files`
        // must list what it cannot open. This verb opens, so it refuses here —
        // and counts the refusals, so the footer can say how many files it
        // declined to read rather than dropping them into silence.
        if !crate::guard::is_openable(&abs.file_name().unwrap_or_default().to_string_lossy()) {
            not_text += 1;
            return true;
        }
        // Failing to open an already-`is_openable`, already-walked file
        // (permissions, a transient I/O error) is the same event as a bad
        // byte inside one that did open: a file that was in the domain and
        // could not be read. Both fold into the one `skipped` tally rather
        // than one counted and the other silently dropped.
        let Ok(file) = std::fs::File::open(abs) else {
            skipped += 1;
            return true;
        };
        let mut reader = std::io::BufReader::new(file);
        let mut per_file: Vec<(usize, String)> = Vec::new();
        let mut lineno = 0usize;

        loop {
            match crate::tools::next_line(&mut reader, rel) {
                Ok(Some(line)) => {
                    lineno += 1;
                    if let Some(m) = re.find(&line.text) {
                        // **The per-file cap is detected one match late, on
                        // purpose.** It was the one cap that set nothing: a
                        // file with thirty matches rendered five lines and was
                        // byte-identical to a file containing exactly five. But
                        // a notice raised at the fifth match would fire for a
                        // file that has exactly five, making a complete result
                        // read as a truncated one — the same collapse pointing
                        // the other way. So the file is read until a *sixth*
                        // match proves something is being withheld; that match
                        // is neither rendered nor charged to the total.
                        if per_file.len() == MAX_MATCHES_PER_FILE {
                            per_file_capped += 1;
                            break;
                        }
                        per_file.push((lineno, window(&line.text, m.start(), m.end())));
                        total += 1;
                        if total >= MAX_MATCHES {
                            capped = true;
                            break;
                        }
                    }
                }
                Ok(None) => break,
                // `is_openable` gates extensions, not contents. One bad byte in
                // one file must not fail a 7,788-file search — it is tallied
                // and the walk continues. The matches already found in this
                // file are withdrawn along with the lines that reported them:
                // a file reported as skipped must not still be charged against
                // the 50-match budget for lines nobody will see.
                Err(_) => {
                    skipped += 1;
                    total -= per_file.len();
                    per_file.clear();
                    break;
                }
            }
        }

        if !per_file.is_empty() {
            hits.push((rel.to_string(), per_file));
        }
        !capped
    });

    let mut out = String::new();
    if hits.is_empty() {
        out.push_str(&format!("no matches for `{query}`\n"));
    } else {
        out.push_str(&format!(
            "{total} matches for `{query}` in {} files\n\n",
            hits.len()
        ));
        for (path, matches) in &hits {
            out.push_str(path);
            out.push('\n');
            for (lineno, text) in matches {
                out.push_str(&format!("  {lineno}: {text}\n"));
            }
        }
        out.push('\n');
    }
    if capped {
        out.push_str(&format!(
            "[stopped at {MAX_MATCHES} matches; narrow with `path` or a tighter query]\n"
        ));
    }
    if per_file_capped > 0 {
        out.push_str(&format!(
            "[{} had more than {MAX_MATCHES_PER_FILE} matches; only the first \
             {MAX_MATCHES_PER_FILE} of each are shown — `read_file` the path, or narrow \
             the query, to see the rest]\n",
            plural(per_file_capped, "file", "files"),
        ));
    }
    if not_text > 0 {
        out.push_str(&format!(
            "[{} not searched: the type is not served as text — `find_files` lists them]\n",
            plural(not_text, "file", "files"),
        ));
    }
    if skipped > 0 {
        out.push_str(&format!("[{skipped} files skipped: unreadable or not valid UTF-8]\n"));
    }
    if stats.timed_out {
        out.push_str("[the search budget expired; results are partial]\n");
    }
    // Not `stats.files`: the walk offers files this verb never opens. The
    // number printed beside "searched" is the number actually read — the two
    // exclusions it subtracts are each announced above, so the three tallies
    // add back up to what was walked.
    let searched = stats.files.saturating_sub(not_text + skipped);
    out.push_str(&domain_note(&domain.scope, searched, stats.complete()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A workspace shaped like the real one: two declared modules, one
    /// declared-but-absent module, and one directory that exists and is
    /// declared by nothing.
    fn workspace() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("modules.toml"),
            "[router]\ncontains = [\"CLAUDE.md\"]\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("modules.local.toml"),
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\n\
             [[repos]]\nname = \"engine\"\npath = \"repos/engine\"\n\n\
             [[repos]]\nname = \"never-cloned\"\npath = \"repos/never-cloned\"\n",
        )
        .unwrap();
        fs::write(dir.path().join("CLAUDE.md"), "# router\n").unwrap();

        fs::create_dir_all(dir.path().join("operators/tycho")).unwrap();
        fs::write(dir.path().join("operators/tycho/self.md"), "needle\n").unwrap();
        fs::create_dir_all(dir.path().join("repos/engine")).unwrap();
        fs::write(dir.path().join("repos/engine/lib.rs"), "// needle\n").unwrap();

        // Declared by nothing. This is `repos/external/` in miniature.
        fs::create_dir_all(dir.path().join("repos/external/opencode")).unwrap();
        fs::write(dir.path().join("repos/external/opencode/x.ts"), "needle\n").unwrap();

        dir
    }

    fn rels(root: &Path, roots: &[PathBuf]) -> Vec<String> {
        let canon = root.canonicalize().unwrap();
        let mut out: Vec<String> = roots
            .iter()
            .map(|p| p.strip_prefix(&canon).unwrap().to_string_lossy().to_string())
            .collect();
        out.sort();
        out
    }

    #[test]
    fn roots_are_the_declared_modules_and_the_router_files() {
        let dir = workspace();
        let roots = search_roots(dir.path()).unwrap();

        assert_eq!(
            rels(dir.path(), &roots.paths),
            vec!["CLAUDE.md", "operators/tycho", "repos/engine"]
        );
    }

    #[test]
    fn modules_and_router_files_are_counted_apart() {
        // The footer used to print this total under the word "modules", which
        // on the real workspace rendered "30 composed modules" while
        // `get_composition` said 23 — two engine-authored answers to one
        // question, with nothing to reconcile them. The counts are separate so
        // the sentence can be true.
        let dir = workspace();
        let roots = search_roots(dir.path()).unwrap();

        assert_eq!(roots.modules, 2, "tycho and engine; never-cloned is absent");
        assert_eq!(roots.router_files, 1, "CLAUDE.md");
        assert_eq!(roots.modules + roots.router_files, roots.paths.len());

        let out = find_files(dir.path(), "**/*.nothing", None).unwrap();
        assert!(
            out.contains("3 declared roots (2 modules, 1 router file)"),
            "the noun must match what is counted: {out}"
        );
    }

    #[test]
    fn a_declared_root_that_is_not_on_disk_is_skipped_not_an_error() {
        // C-4: presence-gated throughout. A manifest naming a repo this
        // machine has never cloned is the normal case, not a malformed
        // workspace.
        let dir = workspace();
        let roots = search_roots(dir.path()).expect("an absent module must not fail enumeration");

        assert!(!rels(dir.path(), &roots.paths).iter().any(|r| r.contains("never-cloned")));
    }

    #[test]
    fn undeclared_content_is_not_a_root_and_declaring_it_makes_it_one() {
        // MUTATION-CHECKED, and the mutation is the whole test: asserting
        // only that `repos/external` is absent would pass identically if
        // `search_roots` returned an empty vec, or if the domain logic did
        // not exist at all. Declaring it and watching it appear is what
        // proves the exclusion is a consequence of the manifest.
        let dir = workspace();
        assert!(
            !rels(dir.path(), &search_roots(dir.path()).unwrap().paths)
                .iter()
                .any(|r| r.contains("external")),
            "undeclared content must not be a search root"
        );

        fs::write(
            dir.path().join("modules.local.toml"),
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\n\
             [[repos]]\nname = \"engine\"\npath = \"repos/engine\"\n\n\
             [[repos]]\nname = \"opencode\"\npath = \"repos/external/opencode\"\n",
        )
        .unwrap();

        assert!(
            rels(dir.path(), &search_roots(dir.path()).unwrap().paths)
                .iter()
                .any(|r| r == "repos/external/opencode"),
            "a declared module must become a search root"
        );
    }

    #[test]
    fn two_declarations_of_the_same_path_collapse_to_one_root() {
        // MUTATION-CHECKED. The manifest parser rejects a *name* collision
        // within a section (C-6) but a *path* collision across sections is a
        // legal manifest — an operator entry and a repo entry can both point
        // at `repos/engine` under different names. Without `push_root`'s
        // `contains` check, that one directory would surface as two roots,
        // and every caller that walks `search_roots` (a search tool grepping
        // each root, in particular) would report every match in it twice.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("modules.toml"), "").unwrap();
        fs::write(
            dir.path().join("modules.local.toml"),
            "[[operators]]\nname = \"engine-operator\"\npath = \"repos/engine\"\n\n\
             [[repos]]\nname = \"engine\"\npath = \"repos/engine\"\n",
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("repos/engine")).unwrap();
        fs::write(dir.path().join("repos/engine/lib.rs"), "// needle\n").unwrap();

        let roots = search_roots(dir.path()).unwrap();
        let hits = roots.paths.iter().filter(|p| p.ends_with("repos/engine")).count();

        assert_eq!(
            hits, 1,
            "one path declared under two names must be one root, not two: {:?}",
            roots.paths
        );
        assert_eq!(
            roots.modules, 1,
            "a collapsed duplicate must not be counted twice in the footer either"
        );
    }

    #[test]
    fn an_empty_path_argument_means_the_default_domain_not_the_whole_disk() {
        // `list_directory` uses "" for the workspace root, so a model has
        // learned that idiom and will send it here meaning "no scope". If ""
        // resolved to the root, it would silently defeat SD-2 and search
        // every external clone — the opposite of what the model asked for.
        let dir = workspace();
        let domain = resolve_domain(dir.path(), Some("")).unwrap();

        assert!(
            matches!(domain.scope, Scope::Composed { .. }),
            "\"\" must not count as an explicit scope"
        );
        assert_eq!(
            rels(dir.path(), &domain.paths),
            vec!["CLAUDE.md", "operators/tycho", "repos/engine"]
        );
    }

    #[test]
    fn an_explicit_path_reaches_content_the_default_domain_excludes() {
        // SD-3, David's constraint: reachable when you are specifically
        // looking at it.
        let dir = workspace();
        let domain = resolve_domain(dir.path(), Some("repos/external/opencode")).unwrap();

        assert!(matches!(domain.scope, Scope::Explicit));
        assert_eq!(rels(dir.path(), &domain.paths), vec!["repos/external/opencode"]);
    }

    #[test]
    fn an_explicit_path_still_passes_the_guard() {
        let dir = workspace();
        for (path, status) in [("../escape", "outside_workspace"), ("nope", "not_found")] {
            let err = resolve_domain(dir.path(), Some(path)).unwrap_err();
            assert_eq!(err.status, status, "{path}");
        }
    }

    fn walked(dir: &Path) -> Vec<String> {
        let roots = search_roots(dir).unwrap();
        let mut seen = Vec::new();
        walk(
            dir,
            &roots.paths,
            Instant::now() + SEARCH_BUDGET,
            &mut |_abs, rel| {
                seen.push(rel.to_string());
                true
            },
        );
        seen.sort();
        seen
    }

    #[test]
    fn the_walk_visits_files_inside_declared_roots() {
        let dir = workspace();
        assert_eq!(
            walked(dir.path()),
            vec!["CLAUDE.md", "operators/tycho/self.md", "repos/engine/lib.rs"]
        );
    }

    #[test]
    fn the_walk_prunes_what_the_guard_denies() {
        // MUTATION-CHECKED. One gate, not two: search inherits every fix the
        // guard ever receives, and a credential is never opened to be matched.
        //
        // `secrets.json` is the discriminating case here, not `.env`: its
        // extension (`json`) is on `is_openable`'s allowlist, so it is the one
        // fixture in this test that is excluded *only* by `filter_entry`
        // consulting `is_hidden_from_listings` — if that prune ever breaks,
        // this is the assertion that catches it. `.env` has no allowlisted
        // extension, so `is_openable` refuses it independently of the prune;
        // it stays here as a pin on that other gate, not as prune coverage.
        let dir = workspace();
        fs::create_dir_all(dir.path().join("repos/engine/node_modules/pkg")).unwrap();
        fs::write(dir.path().join("repos/engine/node_modules/pkg/i.js"), "needle").unwrap();
        fs::create_dir_all(dir.path().join("repos/engine/.worktrees/feat/src")).unwrap();
        fs::write(dir.path().join("repos/engine/.worktrees/feat/src/lib.rs"), "needle").unwrap();
        fs::write(dir.path().join("repos/engine/.env"), "TOKEN=needle").unwrap();
        fs::write(dir.path().join("repos/engine/secrets.json"), "{\"token\":\"needle\"}").unwrap();

        let seen = walked(dir.path());

        assert!(!seen.iter().any(|p| p.contains("node_modules")), "{seen:?}");
        assert!(!seen.iter().any(|p| p.contains(".worktrees")), "{seen:?}");
        assert!(!seen.iter().any(|p| p.contains(".env")), "{seen:?}");
        assert!(!seen.iter().any(|p| p.contains("secrets.json")), "{seen:?}");
    }

    #[test]
    fn the_walk_offers_every_file_the_safety_gate_allows_including_unopenable_types() {
        // C1. The content gate used to live here, and `find_files` inherited it
        // by sharing the walker — so a `.png` sitting in plain view produced
        // "no files matching `**/*.png`" under a footer claiming thousands of
        // files were examined. `guard::is_openable`'s own doc draws this line:
        // it governs opening, and hiding what cannot be opened from a listing
        // reintroduces the projection it was changed to remove.
        let dir = workspace();
        fs::write(dir.path().join("repos/engine/icon.png"), [0x89, 0x50]).unwrap();
        fs::write(dir.path().join("repos/engine/ephe.se1"), [0x00, 0x01]).unwrap();

        let seen = walked(dir.path());

        assert!(seen.iter().any(|p| p.ends_with(".png")), "{seen:?}");
        assert!(seen.iter().any(|p| p.ends_with(".se1")), "{seen:?}");
    }

    #[test]
    fn a_type_that_is_not_served_as_text_is_listed_but_never_read() {
        // The other half of C1: moving the gate must not let `search` open a
        // `.png`. The two verbs now disagree on purpose — one lists, one reads
        // — and the disagreement is stated in the result rather than left for
        // the model to infer.
        let dir = workspace();
        fs::write(dir.path().join("repos/engine/icon.png"), b"needle").unwrap();
        fs::write(dir.path().join("repos/engine/ephe.se1"), b"needle").unwrap();

        let listed = find_files(dir.path(), "**/*.png", None).unwrap();
        assert!(
            listed.contains("repos/engine/icon.png"),
            "a listing must not hide what it cannot open: {listed}"
        );

        let searched = search(dir.path(), "needle", None, false).unwrap();
        assert!(!searched.contains("icon.png"), "search must still not read it: {searched}");
        assert!(!searched.contains("ephe.se1"), "{searched}");
        assert!(
            searched.contains("2 files not searched: the type is not served as text"),
            "the refusal must be counted, not silent: {searched}"
        );
    }

    #[test]
    fn each_footer_counts_what_its_own_verb_actually_did() {
        // The knock-on from C1, and the reason it is pinned: `stats.files` now
        // counts everything walked, so printing it beside "searched" would
        // overstate what `search` read by exactly the files it refused to open.
        // `find_files` examined all five; `search` read three and says so about
        // the other two.
        let dir = workspace();
        fs::write(dir.path().join("repos/engine/icon.png"), [0x89, 0x50]).unwrap();
        fs::write(dir.path().join("repos/engine/ephe.se1"), [0x00, 0x01]).unwrap();

        let listed = find_files(dir.path(), "**/*.nothing", None).unwrap();
        assert!(listed.contains("5 files"), "a listing examined all of them: {listed}");

        let searched = search(dir.path(), "nothing-matches-this", None, false).unwrap();
        assert!(searched.contains("3 files"), "a search read only the text ones: {searched}");
        assert!(!searched.contains("5 files"), "{searched}");
    }

    #[test]
    fn a_visitor_that_returns_false_stops_the_walk() {
        // This is how a cap ends a search: 50 matches must not cost a walk of
        // all 7,788 files.
        let dir = workspace();
        let roots = search_roots(dir.path()).unwrap();
        let mut count = 0usize;
        let stats = walk(dir.path(), &roots.paths, Instant::now() + SEARCH_BUDGET, &mut |_a, _r| {
            count += 1;
            false
        });

        assert_eq!(count, 1);
        assert_eq!(stats.files, 1);
        assert!(
            stats.stopped_short && !stats.complete(),
            "a walk cut short must say so, or its file count is read as a domain size"
        );
    }

    #[test]
    fn an_expired_budget_ends_the_walk_and_says_so() {
        let dir = workspace();
        let roots = search_roots(dir.path()).unwrap();
        let stats = walk(dir.path(), &roots.paths, Instant::now(), &mut |_a, _r| true);

        assert!(stats.timed_out, "an expired deadline must be reported, not silent");
        assert!(!stats.complete(), "and a timed-out walk is not a complete one");
    }

    #[test]
    fn a_symlink_is_not_followed_even_when_its_target_is_inside_a_declared_root() {
        // MUTATION-CHECKED (see report): this pins the behaviour the doc
        // comment on `walk` now describes honestly rather than asserting a
        // guarantee. The target here happens to sit inside the same declared
        // root, so this test cannot by itself distinguish "cycle/duplicate
        // avoidance" from "we simply never look at symlinks" — but it does
        // pin that the link itself is never visited, which is the property
        // `WalkDir::follow_links(false)` is relied on for.
        let dir = workspace();
        fs::write(dir.path().join("repos/engine/real.rs"), "needle-real").unwrap();
        std::os::unix::fs::symlink(
            dir.path().join("repos/engine/real.rs"),
            dir.path().join("repos/engine/link.rs"),
        )
        .unwrap();

        let seen = walked(dir.path());

        assert!(
            seen.iter().any(|p| p == "repos/engine/real.rs"),
            "the real file must still be walked: {seen:?}"
        );
        assert!(
            !seen.iter().any(|p| p == "repos/engine/link.rs"),
            "a symlink must not be visited, even when its target is inside a declared root: {seen:?}"
        );
    }

    #[test]
    fn find_files_matches_the_relative_path_not_just_the_file_name() {
        // Matching the name alone would make `operators/**/*.md` inexpressible,
        // which is most of what a path glob is for in this workspace.
        let dir = workspace();
        let out = find_files(dir.path(), "operators/**/*.md", None).unwrap();

        assert!(out.contains("operators/tycho/self.md"), "{out}");
        assert!(!out.contains("repos/engine/lib.rs"), "{out}");
    }

    #[test]
    fn find_files_states_its_domain_when_it_finds_nothing() {
        // MUTATION-CHECKED. The anti-collapse test: SD-2 narrows what is
        // searched, and without this sentence that narrowing is
        // indistinguishable from an empty workspace.
        let dir = workspace();
        let out = find_files(dir.path(), "**/*.ts", None).unwrap();

        assert!(!out.trim().is_empty(), "an empty text block is a 400");
        assert!(out.contains("declared roots"), "the domain must be named: {out}");
        assert!(out.contains("pass `path`"), "the recovery must be named: {out}");
    }

    #[test]
    fn find_files_reaches_undeclared_content_through_an_explicit_path() {
        let dir = workspace();
        let out = find_files(dir.path(), "**/*.ts", Some("repos/external/opencode")).unwrap();

        assert!(out.contains("x.ts"), "{out}");
    }

    #[test]
    fn find_files_caps_its_output_and_names_the_recovery() {
        let dir = workspace();
        for n in 0..MAX_PATHS + 10 {
            fs::write(dir.path().join(format!("repos/engine/note-{n:04}.md")), "x").unwrap();
        }
        let out = find_files(dir.path(), "**/*.md", None).unwrap();

        assert!(out.contains("stopped at"), "a bitten cap must announce itself: {out}");
        assert!(out.contains("path"), "the recovery must be named: {out}");

        // A truncated list must not read as a complete one: the footer's
        // file count is how far the walk got, not the size of the domain.
        assert!(
            out.contains("stopped early"),
            "a walk ended by a cap must not report its count as a domain size: {out}"
        );
        assert!(
            !out.contains("searched 3 declared roots"),
            "the verb must match how the walk ended: {out}"
        );
    }

    #[test]
    fn a_capped_listing_is_deterministic_rather_than_whatever_the_disk_returned() {
        // M1/I4. The cap keeps the first N *visited*, so an unordered walk lets
        // the filesystem choose which N survive: two identical calls could
        // return different sets, and the survivors were then sorted into
        // something that reads as a lexicographic prefix while `n-0000` was
        // missing from it. `sort_by_file_name` on the walker is what makes the
        // surviving set a fact about the workspace rather than about the disk.
        let dir = workspace();
        for n in 0..MAX_PATHS + 10 {
            fs::write(dir.path().join(format!("repos/engine/n-{n:04}.md")), "x").unwrap();
        }

        let first = find_files(dir.path(), "repos/engine/*.md", None).unwrap();
        let again = find_files(dir.path(), "repos/engine/*.md", None).unwrap();
        assert_eq!(first, again, "two identical calls must return the same result");

        assert!(first.contains("n-0000.md"), "the survivors must be a real prefix: {first}");
        assert!(first.contains("n-0099.md"), "{first}");
        assert!(!first.contains("n-0100.md"), "{first}");
    }

    #[test]
    fn an_uncompilable_glob_is_a_readable_refusal_not_zero_results() {
        // Zero results and "your pattern is broken" are the same sentence to a
        // model unless we make them different.
        let dir = workspace();
        let err = find_files(dir.path(), "[", None).unwrap_err();

        assert_eq!(err.status, "invalid_pattern");
        assert!(!err.detail.is_empty(), "the compiler's own message is the repair hint");
    }

    #[test]
    fn search_returns_the_matching_line_with_its_number_and_path() {
        let dir = workspace();
        let out = search(dir.path(), "needle", None, false).unwrap();

        assert!(out.contains("operators/tycho/self.md"), "{out}");
        assert!(out.contains("1: needle"), "{out}");
        // NOT `!out.contains("external")`: `domain_note`'s own boilerplate
        // names "repos/external/" unconditionally as an example of what SD-2
        // excludes, so that substring is present in every unscoped result
        // whether or not anything under `repos/external/` actually matched.
        // The property under test is that the undeclared file's own path
        // never appears among the hits.
        assert!(
            !out.contains("repos/external/opencode/x.ts"),
            "undeclared content must not match: {out}"
        );
    }

    #[test]
    fn search_is_case_insensitive_unless_told_otherwise() {
        let dir = workspace();
        fs::write(dir.path().join("repos/engine/case.md"), "NEEDLE\n").unwrap();

        assert!(search(dir.path(), "needle", None, false).unwrap().contains("case.md"));
        assert!(!search(dir.path(), "needle", None, true).unwrap().contains("case.md"));
    }

    #[test]
    fn a_long_line_is_windowed_around_the_match() {
        let dir = workspace();
        let line = format!("{}NEEDLE{}", "a".repeat(2000), "b".repeat(2000));
        fs::write(dir.path().join("repos/engine/wide.md"), format!("{line}\n")).unwrap();

        let out = search(dir.path(), "NEEDLE", None, true).unwrap();

        assert!(out.contains("NEEDLE"), "the match itself must survive: {out}");
        assert!(out.contains('…'), "an elided window must say so: {out}");
        assert!(out.len() < 1000, "the window did not bind: {} bytes", out.len());
    }

    #[test]
    fn a_window_never_splits_a_character() {
        // MUTATION-CHECKED. The cap is a character count and this workspace
        // writes `.爻` files at three bytes per character, so a byte-indexed
        // cut lands mid-character and renders U+FFFD. `read_file` carries the
        // same test for the same reason.
        //
        // 500 repeats, not the brief's 2000: at three bytes each, 2000
        // repeats puts NEEDLE at byte offset 6000 in the raw line, past
        // `next_line`'s own `MAX_LINE_BYTES` (4000) truncation — the match is
        // discarded before `search` ever calls `re.find`, so `window` is
        // never reached and the assertions below pass on a file with zero
        // matches (the "no matches for `NEEDLE`" message itself contains the
        // query text, satisfying `out.contains("NEEDLE")` for a reason that
        // has nothing to do with windowing). 500 repeats keeps the whole
        // line under a thousand bytes, well inside the cap, while still far
        // exceeding `MATCH_WINDOW_CHARS` (200) on both sides of the match so
        // `window`'s slicing is actually exercised.
        let dir = workspace();
        let line = format!("{}NEEDLE{}", "爻".repeat(500), "爻".repeat(500));
        fs::write(dir.path().join("repos/engine/wide.爻"), format!("{line}\n")).unwrap();

        let out = search(dir.path(), "NEEDLE", None, true).unwrap();

        // Pins that `window` actually ran, not merely that its two possible
        // failure symptoms are absent. Without this, `!out.contains('\u{FFFD}')`
        // is trivially true and `out.contains("NEEDLE")` is satisfied by the
        // echoed query inside `no matches for \`NEEDLE\`` whenever the match
        // never reaches `window` at all — exactly the fixture-vacuity found
        // above with the 2000-repeat version. `…` cannot come from a "no
        // matches" line (`domain_note` uses an em dash, not an ellipsis), so
        // its presence is real evidence the window ran.
        assert!(out.contains('…'), "the window must have run: {out}");
        assert!(!out.contains('\u{FFFD}'), "a character was split: {out}");
        assert!(out.contains("NEEDLE"), "{out}");
    }

    #[test]
    fn one_chatty_file_cannot_consume_the_whole_match_budget() {
        let dir = workspace();
        let body: String = (0..100).map(|i| format!("needle {i}\n")).collect();
        fs::write(dir.path().join("repos/engine/chatty.md"), body).unwrap();

        let out = search(dir.path(), "needle", None, false).unwrap();
        let from_chatty = out.matches("needle ").count();

        assert_eq!(
            from_chatty, MAX_MATCHES_PER_FILE,
            "the per-file cap must bind exactly, not merely bound: {out}"
        );

        // C2, and the whole point of the finding: this was the one cap that
        // set nothing. Five lines from a file of a hundred matches were
        // byte-indistinguishable from a file containing exactly five, so
        // "that is all there is" and "that is all you were shown" rendered
        // identically — the failure this branch exists to refuse.
        assert!(
            out.contains(&format!("more than {MAX_MATCHES_PER_FILE} matches")),
            "a bitten per-file cap must announce itself: {out}"
        );
        assert!(
            out.contains("1 file had"),
            "and say how many files it bit: {out}"
        );
        assert!(
            out.contains("read_file"),
            "and name the recovery: {out}"
        );
    }

    #[test]
    fn a_file_at_exactly_the_per_file_cap_does_not_claim_to_be_truncated() {
        // The other side of C2. A notice that fires whether or not anything was
        // withheld is as uninformative as no notice at all: it would make a
        // complete result read as a truncated one, which is the same collapse
        // pointing the other way.
        let dir = workspace();
        let body: String = (0..MAX_MATCHES_PER_FILE).map(|i| format!("needle {i}\n")).collect();
        fs::write(dir.path().join("repos/engine/exact.md"), body).unwrap();

        let out = search(dir.path(), "needle", None, false).unwrap();

        assert!(out.contains("exact.md"), "{out}");
        assert!(
            !out.contains("more than"),
            "nothing was withheld, so nothing may claim it was: {out}"
        );
    }

    #[test]
    fn a_bitten_total_cap_announces_itself_without_inventing_a_total() {
        // Hitting the cap STOPS the walk, so the true total is unknowable
        // without spending the walk we just saved. Saying "50 of 214" would be
        // a number nothing measured.
        let dir = workspace();
        for n in 0..40 {
            fs::write(
                dir.path().join(format!("repos/engine/f-{n:03}.md")),
                "needle\nneedle\nneedle\n",
            )
            .unwrap();
        }
        let out = search(dir.path(), "needle", None, false).unwrap();

        assert!(out.contains("stopped at"), "{out}");
        // A shaped check rather than a bare `" of "`: the domain note now
        // (correctly) contains the phrase "the size of the domain", and the
        // defect this guards is specifically an `N of M` denominator nothing
        // measured.
        assert!(
            !regex::Regex::new(r"\d+ of \d+").unwrap().is_match(&out),
            "no invented denominator: {out}"
        );

        // The cap must BIND, not merely announce itself. Without this the test
        // passes for a cap that fires at the wrong threshold, or that fires and
        // still leaks extra matches into the body — it only ever proved that a
        // string appeared somewhere in the tail. (Found by the Task 4 review,
        // which caught the same shape in `find_files`'s cap test.)
        //
        // NOT a literal `": needle"` substring check: `workspace()`'s own
        // `repos/engine/lib.rs` fixture matches with the line `// needle`, so
        // its reported line renders as `1: // needle` — a real, counted match
        // that does not contain `": needle"` because of the comment marker in
        // between. That undercounts by exactly one every time, regardless of
        // whether the cap binds correctly. A match-report line is
        // unambiguous by its own format instead: `  {lineno}: {text}`, i.e.
        // leading whitespace, then digits, then `": "` — which no path line
        // or bracketed note in the output can produce. See `count_match_lines`.
        assert_eq!(
            count_match_lines(&out), MAX_MATCHES,
            "the cap announced itself but did not bind: {out}"
        );

        // I3: the walk stopped at the cap, so its file count is how far it got
        // and not the size of the domain. The verb has to carry that, because
        // the number alone reads as a denominator.
        assert!(
            out.contains("stopped early"),
            "a count from an interrupted walk must not be reported as a search: {out}"
        );
        assert!(!out.contains("searched 3 declared roots"), "{out}");
    }

    #[test]
    fn a_credential_is_never_matched_however_well_it_matches() {
        // MUTATION-CHECKED, and `secrets.json` is the assertion that does the
        // work. `.env` and `Secrets.xcconfig` are excluded by `is_openable`
        // (neither has an allowlisted extension) whether or not the guard's
        // prune runs at all — so they pin the extension gate and discriminate
        // nothing about pruning. `secrets.json` matches `is_credential`
        // ("secret") AND is openable (`json`), so the prune is the only thing
        // standing between it and being read. Found in Task 3, where the
        // equivalent `.env` assertion stayed green under the mutation.
        let dir = workspace();
        fs::write(dir.path().join("repos/engine/.env"), "TOKEN=needle\n").unwrap();
        fs::write(dir.path().join("repos/engine/Secrets.xcconfig"), "KEY=needle\n").unwrap();
        fs::write(dir.path().join("repos/engine/secrets.json"), "{\"k\":\"needle\"}\n").unwrap();

        let out = search(dir.path(), "needle", None, false).unwrap();

        assert!(!out.contains(".env"), "{out}");
        assert!(!out.contains("Secrets"), "{out}");
        assert!(!out.contains("TOKEN"), "{out}");
        assert!(!out.contains("secrets.json"), "the prune is the only gate here: {out}");
    }

    #[test]
    fn one_unreadable_file_is_tallied_rather_than_failing_the_search() {
        let dir = workspace();
        fs::write(dir.path().join("repos/engine/bad.md"), [0xff, 0xfe, 0x00]).unwrap();

        let out = search(dir.path(), "needle", None, false).unwrap();

        assert!(out.contains("operators/tycho/self.md"), "the good hits survive: {out}");
        assert!(out.contains("skipped"), "the skip is reported, not silent: {out}");
    }

    /// Count report lines by their own format — leading whitespace, digits,
    /// then `": "` — rather than by the text of any particular query, so the
    /// count is not fooled by a match whose rendered text does not start
    /// with the query itself (`repos/engine/lib.rs`'s `// needle`, in
    /// particular).
    fn count_match_lines(out: &str) -> usize {
        out.lines()
            .filter(|l| {
                let t = l.trim_start();
                let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
                !digits.is_empty() && t[digits.len()..].starts_with(": ")
            })
            .count()
    }

    #[test]
    fn a_file_that_matches_before_going_unreadable_does_not_inflate_the_total() {
        // Established by reverting the fix and re-running: without
        // `total -= per_file.len();` before `per_file.clear()`, this test
        // fails — `total` stays at +2 for `partial.md`'s two good lines even
        // though `partial.md` is dropped from `hits` entirely (a file
        // reported as skipped renders no lines), so the header's own count
        // no longer matches what it prints, and those two withdrawn matches
        // would still have consumed budget against `MAX_MATCHES`.
        let dir = workspace();
        let mut body = b"needle\nneedle\n".to_vec();
        body.extend_from_slice(&[0xff, 0xfe, 0x00]);
        fs::write(dir.path().join("repos/engine/partial.md"), body).unwrap();

        let out = search(dir.path(), "needle", None, false).unwrap();

        assert!(
            !out.contains("partial.md"),
            "a file reported as skipped must not also appear as a hit: {out}"
        );
        assert!(out.contains("skipped"), "{out}");

        let header_total: usize = out
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);

        assert_eq!(
            header_total,
            count_match_lines(&out),
            "the header must not count matches it renders no line for: {out}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_file_that_cannot_be_opened_is_tallied_rather_than_silently_dropped() {
        // Established by reverting the fix and re-running: without folding
        // the `File::open` failure into `skipped`, this test fails — `out`
        // never says "skipped" even though `locked.md` is a real, in-domain,
        // openable-by-extension file that was never read. That is quieter
        // than the UTF-8 case, which was at least counted.
        use std::os::unix::fs::PermissionsExt;

        let dir = workspace();
        let path = dir.path().join("repos/engine/locked.md");
        fs::write(&path, "needle\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();

        let out = search(dir.path(), "needle", None, false).unwrap();

        // Restore permissions so the tempdir can clean itself up.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        assert!(
            out.contains("skipped"),
            "an unopenable file must be tallied, not silently dropped: {out}"
        );
        assert!(out.contains("operators/tycho/self.md"), "the good hits survive: {out}");
    }

    #[test]
    fn search_states_its_domain_when_it_finds_nothing() {
        let dir = workspace();
        let out = search(dir.path(), "nothing-matches-this", None, false).unwrap();

        assert!(!out.trim().is_empty(), "an empty text block is a 400");
        assert!(out.contains("declared roots"), "{out}");
        assert!(out.contains("pass `path`"), "{out}");
    }

    #[test]
    fn the_note_promises_a_recovery_only_where_one_exists() {
        // I1, and the point of running it rather than reading it: the sentence
        // this replaced said "Undeclared content — repos/external/, worktrees —
        // was not searched; pass `path` to search one directly", which promised
        // a repair the guard refuses. `.worktrees` is in `is_noise`, so
        // `guard::resolve` denies it at any depth. Two categories, and only one
        // of them has a recovery:
        //
        //   reachable if you name it — anything the manifest does not declare
        //   refused however you name it — the noise and credential set
        //
        // Worktrees were never in the first category at all: they sit *inside*
        // declared roots and are excluded by a different mechanism.
        let dir = workspace();
        fs::create_dir_all(dir.path().join("repos/engine/.worktrees/feat")).unwrap();
        fs::write(dir.path().join("repos/engine/.worktrees/feat/w.rs"), "needle\n").unwrap();
        fs::create_dir_all(dir.path().join("repos/engine/node_modules/pkg")).unwrap();
        fs::write(dir.path().join("repos/engine/node_modules/pkg/i.js"), "needle\n").unwrap();

        let note = search(dir.path(), "nothing-matches-this", None, false).unwrap();

        // Category one: named as not-searched, and the offered recovery works.
        assert!(note.contains("does not declare"), "{note}");
        assert!(note.contains("pass `path` to search it directly"), "{note}");
        let reached = search(dir.path(), "needle", Some("repos/external/opencode"), false).unwrap();
        assert!(reached.contains("x.ts"), "the promised recovery must work: {reached}");

        // Category two: named as never-searched, and every name in that
        // sentence is refused when a caller tries the recovery anyway. This is
        // the assertion that would have caught the original defect — it
        // executes the claim instead of reading it.
        assert!(note.contains("never searched at any path"), "{note}");
        for named in [".worktrees", "node_modules", "target", ".build", ".armillary"] {
            assert!(note.contains(named), "the note must name {named}: {note}");
        }
        for refused in ["repos/engine/.worktrees/feat", "repos/engine/node_modules/pkg"] {
            let err = search(dir.path(), "needle", Some(refused), false).unwrap_err();
            assert_eq!(
                err.status, "denied_noise",
                "{refused} is refused, so the note must not offer `path` for it"
            );
        }
    }

    #[test]
    fn a_scoped_result_states_its_exclusions_too() {
        // M2. Under an explicit `path` the guard still prunes credentials,
        // `node_modules`, worktrees and the rest — so a scoped note that named
        // nothing let `search("apiKey", path="repos/app")` render "not there"
        // about an answer sitting in `node_modules`. The commitment held on the
        // unscoped path only.
        let dir = workspace();
        fs::create_dir_all(dir.path().join("repos/engine/node_modules/pkg")).unwrap();
        fs::write(dir.path().join("repos/engine/node_modules/pkg/i.js"), "apiKey\n").unwrap();

        let out = search(dir.path(), "apiKey", Some("repos/engine"), false).unwrap();

        assert!(out.contains("no matches"), "{out}");
        assert!(out.contains("1 path named explicitly"), "{out}");
        assert!(
            out.contains("node_modules") && out.contains("never searched at any path"),
            "a scoped zero-match result must still name what it did not look at: {out}"
        );

        let listed = find_files(dir.path(), "**/*.js", Some("repos/engine")).unwrap();
        assert!(listed.contains("never searched at any path"), "{listed}");
    }

    #[test]
    fn an_uncompilable_regex_is_a_readable_refusal() {
        let dir = workspace();
        let err = search(dir.path(), "(unclosed", None, false).unwrap_err();

        assert_eq!(err.status, "invalid_pattern");
        assert!(!err.detail.is_empty());
    }
}

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
pub(crate) fn search_roots(root: &Path) -> Result<Vec<PathBuf>, ToolError> {
    let composition = armillary_composition::parse_workspace(root)
        .map_err(|e| ToolError::new("composition_unreadable", e.to_string()))?;

    let mut out: Vec<PathBuf> = Vec::new();

    for m in composition
        .operators
        .iter()
        .chain(&composition.commons)
        .chain(&composition.repos)
    {
        push_root(root, &m.path, &mut out);
    }
    for f in &composition.router.contains {
        push_root(root, f, &mut out);
    }

    Ok(out)
}

/// Resolve one declared path and add it if it survives the guard.
///
/// A free function rather than a closure over `out` on purpose: a closure
/// holding `&mut out` across both loops and then returning `out` is a
/// borrow-checker argument nobody needs to have.
fn push_root(root: &Path, rel: &str, out: &mut Vec<PathBuf>) {
    if let Ok(p) = crate::guard::resolve(root, rel) {
        // Canonical, so two declarations reaching the same directory through
        // different spellings collapse to one root rather than producing
        // every match twice.
        if !out.contains(&p) {
            out.push(p);
        }
    }
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
pub(crate) fn resolve_domain(
    root: &Path,
    path: Option<&str>,
) -> Result<(Vec<PathBuf>, bool), ToolError> {
    match path {
        Some(p) if !p.is_empty() => Ok((vec![crate::guard::resolve(root, p)?], true)),
        _ => Ok((search_roots(root)?, false)),
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
    /// note below — so a caller that opens files must subtract its own
    /// refusals before printing this next to the word "searched".
    pub files: usize,
    pub timed_out: bool,
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
/// reintroduces the projection it was changed to remove. `find_files` is a
/// listing; applying the content gate here made `find_files("**/*.png")` answer
/// "there are none" about files sitting in plain view, under a footer claiming
/// thousands of files had been examined.
///
/// **Entries are visited in sorted order** (`sort_by_file_name`), roots in
/// declaration order. Without it the filesystem decides, so two identical calls
/// could return different results and — once a cap bites — a different *set*.
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
    let mut stats = WalkStats { files: 0, timed_out: false };

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

/// The sentence every result ends with, naming what was and was not searched.
///
/// **The whole absence-vs-refusal lesson lives here.** SD-2 deliberately
/// narrows the domain; without this, that narrowing reads exactly like an
/// empty workspace, and the model has no way to learn otherwise. The 2026-08-01
/// sync work shipped the opposite of this — `verdict()` returned `current` for
/// twenty-four repos having contacted nothing.
fn domain_note(scoped: bool, roots: usize, files: usize) -> String {
    if scoped {
        format!("[searched 1 path explicitly, {files} files]\n")
    } else {
        format!(
            "[searched {roots} composed modules, {files} files. \
             Undeclared content — repos/external/, worktrees — was not searched; \
             pass `path` to search one directly.]\n"
        )
    }
}

/// Find files whose workspace-relative path matches a glob.
///
/// **A listing, not a read.** It opens nothing, so the content gate
/// (`is_openable`) does not apply: a `.png` or a `.se1` matching the pattern is
/// reported. The alternative — inheriting `search`'s gate because the two share
/// a walker — made this verb answer "no files matching `**/*.png`" about files
/// sitting in plain view.
pub(crate) fn find_files(
    root: &Path,
    pattern: &str,
    path: Option<&str>,
) -> Result<String, ToolError> {
    let glob = globset::Glob::new(pattern)
        .map_err(|e| ToolError::new("invalid_pattern", e.to_string()))?
        .compile_matcher();

    let (roots, scoped) = resolve_domain(root, path)?;
    let deadline = Instant::now() + SEARCH_BUDGET;

    let mut found: Vec<String> = Vec::new();
    let mut capped = false;
    let stats = walk(root, &roots, deadline, &mut |_abs, rel| {
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
        out.push_str(&format!(
            "[stopped at {MAX_PATHS} paths; narrow with `path` or a tighter pattern]\n"
        ));
    }
    if stats.timed_out {
        out.push_str("[the search budget expired; results are partial]\n");
    }
    // `stats.files` is exactly what this verb examined: every walked path was
    // tested against the glob, because nothing here refuses a file for its type.
    out.push_str(&domain_note(scoped, roots.len(), stats.files));
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

    let (roots, scoped) = resolve_domain(root, path)?;
    let deadline = Instant::now() + SEARCH_BUDGET;

    let mut hits: Vec<(String, Vec<(usize, String)>)> = Vec::new();
    let mut total = 0usize;
    let mut capped = false;
    let mut skipped = 0usize;
    let mut not_text = 0usize;

    let stats = walk(root, &roots, deadline, &mut |abs, rel| {
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
                        per_file.push((lineno, window(&line.text, m.start(), m.end())));
                        total += 1;
                        if total >= MAX_MATCHES {
                            capped = true;
                            break;
                        }
                        if per_file.len() >= MAX_MATCHES_PER_FILE {
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
    if not_text > 0 {
        out.push_str(&format!(
            "[{not_text} files not searched: the type is not served as text — \
             `find_files` lists them]\n"
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
    out.push_str(&domain_note(scoped, roots.len(), searched));
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
            rels(dir.path(), &roots),
            vec!["CLAUDE.md", "operators/tycho", "repos/engine"]
        );
    }

    #[test]
    fn a_declared_root_that_is_not_on_disk_is_skipped_not_an_error() {
        // C-4: presence-gated throughout. A manifest naming a repo this
        // machine has never cloned is the normal case, not a malformed
        // workspace.
        let dir = workspace();
        let roots = search_roots(dir.path()).expect("an absent module must not fail enumeration");

        assert!(!rels(dir.path(), &roots).iter().any(|r| r.contains("never-cloned")));
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
            !rels(dir.path(), &search_roots(dir.path()).unwrap())
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
            rels(dir.path(), &search_roots(dir.path()).unwrap())
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
        let hits = roots.iter().filter(|p| p.ends_with("repos/engine")).count();

        assert_eq!(
            hits, 1,
            "one path declared under two names must be one root, not two: {roots:?}"
        );
    }

    #[test]
    fn an_empty_path_argument_means_the_default_domain_not_the_whole_disk() {
        // `list_directory` uses "" for the workspace root, so a model has
        // learned that idiom and will send it here meaning "no scope". If ""
        // resolved to the root, it would silently defeat SD-2 and search
        // every external clone — the opposite of what the model asked for.
        let dir = workspace();
        let (roots, scoped) = resolve_domain(dir.path(), Some("")).unwrap();

        assert!(!scoped, "\"\" must not count as an explicit scope");
        assert_eq!(
            rels(dir.path(), &roots),
            vec!["CLAUDE.md", "operators/tycho", "repos/engine"]
        );
    }

    #[test]
    fn an_explicit_path_reaches_content_the_default_domain_excludes() {
        // SD-3, David's constraint: reachable when you are specifically
        // looking at it.
        let dir = workspace();
        let (roots, scoped) = resolve_domain(dir.path(), Some("repos/external/opencode")).unwrap();

        assert!(scoped);
        assert_eq!(rels(dir.path(), &roots), vec!["repos/external/opencode"]);
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
            &roots,
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
        // The content gate used to live here, and `find_files` inherited it by
        // sharing the walker — so a `.png` sitting in plain view produced "no
        // files matching `**/*.png`" under a footer claiming thousands of files
        // were examined. `guard::is_openable`'s own doc draws this line: it
        // governs opening, and hiding what cannot be opened from a listing
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
        // The other half: moving the gate must not let `search` open a `.png`.
        // The two verbs now disagree on purpose — one lists, one reads — and the
        // disagreement is stated in the result rather than left to be inferred.
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
        // The knock-on, and the reason it is pinned: `stats.files` now counts
        // everything walked, so printing it beside "searched" would overstate
        // what `search` read by exactly the files it refused to open.
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
    fn a_capped_listing_is_deterministic_rather_than_whatever_the_disk_returned() {
        // The cap keeps the first N *visited*, so an unordered walk lets the
        // filesystem choose which N survive: two identical calls could return
        // different sets, and the survivors were then sorted into something that
        // reads as a lexicographic prefix while `n-0000` was missing from it.
        // `sort_by_file_name` on the walker is what makes the surviving set a
        // fact about the workspace rather than about the disk.
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
    fn a_visitor_that_returns_false_stops_the_walk() {
        // This is how a cap ends a search: 50 matches must not cost a walk of
        // all 7,788 files.
        let dir = workspace();
        let roots = search_roots(dir.path()).unwrap();
        let mut count = 0usize;
        let stats = walk(dir.path(), &roots, Instant::now() + SEARCH_BUDGET, &mut |_a, _r| {
            count += 1;
            false
        });

        assert_eq!(count, 1);
        assert_eq!(stats.files, 1);
    }

    #[test]
    fn an_expired_budget_ends_the_walk_and_says_so() {
        let dir = workspace();
        let roots = search_roots(dir.path()).unwrap();
        let stats = walk(dir.path(), &roots, Instant::now(), &mut |_a, _r| true);

        assert!(stats.timed_out, "an expired deadline must be reported, not silent");
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
        assert!(out.contains("composed"), "the domain must be named: {out}");
        assert!(out.contains("path"), "the recovery must be named: {out}");
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

        assert!(
            from_chatty <= MAX_MATCHES_PER_FILE,
            "one file produced {from_chatty} matches: {out}"
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
        assert!(!out.contains(" of "), "no invented denominator: {out}");

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
        assert!(out.contains("composed"), "{out}");
        assert!(out.contains("path"), "{out}");
    }

    #[test]
    fn an_uncompilable_regex_is_a_readable_refusal() {
        let dir = workspace();
        let err = search(dir.path(), "(unclosed", None, false).unwrap_err();

        assert_eq!(err.status, "invalid_pattern");
        assert!(!err.detail.is_empty());
    }
}

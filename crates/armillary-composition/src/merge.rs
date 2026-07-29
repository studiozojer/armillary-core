use crate::{Composition, CompositionError, Module, Protocol, Router};
use std::collections::HashSet;

pub(crate) trait Named {
    fn name(&self) -> &str;
}

impl Named for Module {
    fn name(&self) -> &str {
        &self.name
    }
}

impl Named for Protocol {
    fn name(&self) -> &str {
        &self.name
    }
}

/// C-6 applies *within* a manifest too, not only across the pair.
///
/// The original implementation seeded `seen` from base without checking base
/// against itself, so two `[[operators]] name='tycho'` in one file merged to two
/// operators in silence while the identical mistake across two files was fatal.
///
/// The C-2 migration path generates exactly this by construction: `[[models]]`
/// and `[[operators]]` both declaring tycho concatenate unchecked, which is the
/// precise mid-rename state C-2 exists to support — and the compat symlink makes
/// both paths the same directory, so the dispatcher would boot one operator
/// twice with no diagnostic.
pub(crate) fn check_unique<T: Named>(
    section: &'static str,
    entries: &[T],
) -> Result<(), CompositionError> {
    let mut seen: HashSet<&str> = HashSet::new();
    for entry in entries {
        if !seen.insert(entry.name()) {
            return Err(CompositionError::DuplicateName {
                section,
                name: entry.name().to_string(),
            });
        }
    }
    Ok(())
}

fn union<T: Named>(
    section: &'static str,
    mut base: Vec<T>,
    overlay: Vec<T>,
) -> Result<Vec<T>, CompositionError> {
    let mut seen: HashSet<String> = base.iter().map(|e| e.name().to_string()).collect();
    for entry in overlay {
        if !seen.insert(entry.name().to_string()) {
            return Err(CompositionError::NameCollision {
                section,
                name: entry.name().to_string(),
            });
        }
        base.push(entry);
    }
    Ok(base)
}

/// C-6 for a TABLE rather than an array-of-tables: `[router]` merges
/// FIELD-WISE, overlay winning per field — not wholesale. A machine-local
/// overlay must be able to set `boot` without restating `contains`, which a
/// wholesale replace would silently erase.
///
/// No collision error here, unlike the `[[...]]` sections: those key on
/// `name` and a duplicate is genuinely ambiguous, whereas `[router]` is a
/// singleton whose whole purpose is per-machine override.
fn merge_router(base: Router, overlay: Router) -> Router {
    let mut extra = base.extra;
    extra.extend(overlay.extra);
    Router {
        contains: if overlay.contains.is_empty() { base.contains } else { overlay.contains },
        boot: overlay.boot.or(base.boot),
        extra,
    }
}

/// C-6: the overlay merges additively per section, base first then overlay,
/// preserving declaration order within each. A `name` collision within a
/// section is an error — never a silent override, never a silent duplicate.
///
/// An ambiguous composition is not a composition: a boot that quietly picks one
/// of two candidates is a boot that cannot be trusted about what it loaded.
pub fn merge(base: Composition, overlay: Composition) -> Result<Composition, CompositionError> {
    // Each side must be internally consistent before they can be combined —
    // otherwise "declared in both manifests" would be the message for a fault
    // that lives entirely in one.
    for side in [&base, &overlay] {
        check_unique("operators", &side.operators)?;
        check_unique("commons", &side.commons)?;
        check_unique("repos", &side.repos)?;
        check_unique("protocols", &side.protocols)?;
    }

    Ok(Composition {
        operators: union("operators", base.operators, overlay.operators)?,
        commons: union("commons", base.commons, overlay.commons)?,
        repos: union("repos", base.repos, overlay.repos)?,
        protocols: union("protocols", base.protocols, overlay.protocols)?,
        router: merge_router(base.router, overlay.router),
    })
}

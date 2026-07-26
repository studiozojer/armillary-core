use crate::{Composition, CompositionError, Module, Protocol};
use std::collections::HashSet;

trait Named {
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

/// C-6: the overlay merges additively per section, base first then overlay,
/// preserving declaration order within each. A `name` collision within a
/// section is an error — never a silent override, never a silent duplicate.
///
/// An ambiguous composition is not a composition: a boot that quietly picks one
/// of two candidates is a boot that cannot be trusted about what it loaded.
pub fn merge(base: Composition, overlay: Composition) -> Result<Composition, CompositionError> {
    Ok(Composition {
        operators: union("operators", base.operators, overlay.operators)?,
        commons: union("commons", base.commons, overlay.commons)?,
        repos: union("repos", base.repos, overlay.repos)?,
        protocols: union("protocols", base.protocols, overlay.protocols)?,
    })
}

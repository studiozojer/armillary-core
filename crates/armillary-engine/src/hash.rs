use sha2::{Digest, Sha256};

/// SHA-256, lowercase hex, unprefixed.
///
/// A sprint-1 design decision (D12 in the design sheet at
/// `zojercommons/projects/harness/specs/`), not a rule of the standard — the
/// constitution says nothing about hash algorithms. Cited by origin so the
/// reference resolves somewhere.
///
/// Always over raw bytes. Hashing a decoded string would make the hash describe
/// the decoder rather than the file — and the point of emitting it at all is
/// that a log can later cite exactly which bytes were in a window.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_digest() {
        // printf '# hello' | shasum -a 256
        assert_eq!(
            sha256_hex(b"# hello"),
            "ea67f39f2a707e536439ee31e49fdd586b4a8437d3408f0466112d040cd06681"
        );
    }
}

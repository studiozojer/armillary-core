//! Who is asking, resolved from the request.
//!
//! # The ordering, which is load-bearing
//!
//! Authenticate first, so an unauthenticated caller learns nothing about
//! the workspace's grants. Then the registry, then the manifest ceiling, so
//! an ungranted device does not learn the ceiling either. This is `guard.rs`'s
//! lesson in another register: judge the thing you will actually act on, and
//! leak less on the way.
//!
//! # What this does NOT gate
//!
//! Reads. `/tree`, `/file`, `/composition`, `/streams` and every `GET` stay
//! open on the overlay, which is lawful under A-5's exception as written.
//! That is a DECISION taken 2026-08-07, not an omission — an enrolled tailnet
//! device still reads the whole composed workspace with no credential. It is
//! residual R1 in the design, and the condition to revisit it is a second
//! person or a device that is not David's.

use crate::principals::{Grant, Principal, Registry};
use crate::state::SharedState;
use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use axum::http::StatusCode;

/// The authenticated principal behind one request.
#[derive(Debug, Clone)]
pub struct Caller(pub Principal);

/// A refusal, as the stable machine-readable code plus a sentence naming the
/// fix. Codes are never a `Debug` rendering — the rule `GuardError::code`
/// already states, and for the same reason: a client would start matching on
/// it and the variant names become public API.
pub fn denied(code: &'static str) -> (StatusCode, String) {
    let (status, why) = match code {
        "no_principal" => (
            StatusCode::UNAUTHORIZED,
            "this request mutates state and carried no credential. Enrol this device on the host with `armillary-engine enroll --name <name> --grants sync,push` and send its token as `Authorization: Bearer <token>`.",
        ),
        "unknown_principal" => (
            StatusCode::UNAUTHORIZED,
            "that token belongs to no principal on this host — it may have been revoked. Re-enrol with `armillary-engine enroll`.",
        ),
        "principal_not_granted" => (
            StatusCode::FORBIDDEN,
            "this device is enrolled but was not granted that authority. Re-enrol it with the grant: `armillary-engine enroll --name <name> --grants sync,push`.",
        ),
        other => (StatusCode::FORBIDDEN, other),
    };
    (status, format!("{code}: {why}"))
}

/// The `Bearer` token in an `Authorization` header value, if there is one.
///
/// The scheme is case-insensitive per RFC 6750 and clients genuinely differ.
fn bearer(header: &str) -> Option<&str> {
    let (scheme, rest) = header.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let token = rest.trim();
    (!token.is_empty()).then_some(token)
}

/// Refuse unless the caller holds this grant.
///
/// This is the REGISTRY half only. The manifest ceiling is checked separately
/// at each route — see `routes::repos`, where the reason the two are separate
/// is argued.
pub fn require(caller: &Caller, g: Grant) -> Result<(), (StatusCode, String)> {
    if caller.0.holds(g) {
        return Ok(());
    }
    Err(denied("principal_not_granted"))
}

impl FromRequestParts<SharedState> for Caller {
    type Rejection = (StatusCode, String);

    async fn from_request_parts(
        parts: &mut Parts,
        state: &SharedState,
    ) -> Result<Self, Self::Rejection> {
        let header = parts
            .headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| denied("no_principal"))?;
        let token = bearer(header).ok_or_else(|| denied("no_principal"))?;

        // The registry path comes from `AppState`, NOT from
        // `default_registry_dir()` reading `$HOME` here. Same reason
        // `models_path` is a field: "a route reading a hard-coded `$HOME`
        // path is untestable — and a test that only passes on a machine
        // which happens to lack the file is worse than no test." Resolving
        // the default at the outermost caller and passing the path inward
        // is the pattern Task 4 established when it extracted `enroll(dir,
        // …)`; reaching for the env here would force every gate test to
        // mutate process-global `HOME`, which Rust's parallel test threads
        // turn into failures that appear only under load.
        //
        // Read per request, deliberately. `revoke` takes effect on the next
        // request with nothing to restart and no cache to go stale — the
        // property the manifest gates already have and David named as
        // valuable at the grant site.
        Registry::load(&state.registry_dir)
            .authenticate(token)
            .cloned()
            .map(Caller)
            .ok_or_else(|| denied("unknown_principal"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::principals::{hash_token, mint_token, write_principal, Principal};
    use crate::state::{AppState, ModelConfig};
    use axum::http::Request;
    use std::sync::Arc;

    fn granted(grants: Vec<Grant>) -> Caller {
        Caller(Principal {
            name: "iphone".to_string(),
            token_hash: hash_token(&mint_token()),
            grants,
            minted: "2026-08-07T00:00:00Z".to_string(),
        })
    }

    /// A minimal `SharedState` whose ONLY field these tests care about is
    /// `registry_dir` — everything else is a harmless placeholder, exactly
    /// as `tests/routes.rs`'s fixtures already treat the fields they don't
    /// exercise (`models_path: "/nonexistent/models.toml"`, `boot: None`).
    fn state_with_registry(registry_dir: std::path::PathBuf) -> SharedState {
        let data_dir = tempfile::tempdir().unwrap();
        let store = crate::log::store::LogStore::open(data_dir.path()).unwrap();
        Arc::new(AppState {
            root: std::path::PathBuf::from("."),
            sessions: Arc::new(crate::sessions::Sessions::new(store)),
            model: ModelConfig { model: "claude-sonnet-5".to_string() },
            providers: crate::provider::fixed(Arc::new(crate::provider::KeylessProvider)),
            models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
            registry_dir,
            anthropic_key_present: false,
            zen_key_present: false,
            boot: None,
        })
    }

    /// A request's `Parts`, with an optional `Authorization` header — the
    /// exact seam `from_request_parts` reads. Built through a real
    /// `http::Request` rather than hand-assembled, so this exercises the
    /// SAME header lookup axum performs against a live request.
    fn parts(auth: Option<&str>) -> Parts {
        let mut builder = Request::builder().uri("/");
        if let Some(value) = auth {
            builder = builder.header(axum::http::header::AUTHORIZATION, value);
        }
        let (parts, ()) = builder.body(()).unwrap().into_parts();
        parts
    }

    #[test]
    fn a_held_grant_passes() {
        assert!(require(&granted(vec![Grant::Sync, Grant::Push]), Grant::Push).is_ok());
    }

    #[test]
    fn an_absent_grant_is_403_and_names_the_fix_for_this_caller() {
        let err = require(&granted(vec![Grant::Sync]), Grant::Push).unwrap_err();
        assert_eq!(err.0, StatusCode::FORBIDDEN);
        assert!(err.1.contains("principal_not_granted"), "{}", err.1);
        // The message must point at `enroll`, NOT at the manifest — a
        // caller told to edit `modules.local.toml` when their own grant is
        // the problem will edit the wrong file and widen the ceiling for
        // every device to fix one.
        assert!(err.1.contains("enroll"), "{}", err.1);
        assert!(!err.1.contains("modules.local.toml"), "{}", err.1);
    }

    #[test]
    fn the_two_authentication_failures_are_distinguishable() {
        // 401 for "who are you", 403 for "you may not". A caller with a
        // revoked token and a caller with a valid ungranted token are
        // different problems with different fixes.
        assert_eq!(denied("no_principal").0, StatusCode::UNAUTHORIZED);
        assert_eq!(denied("unknown_principal").0, StatusCode::UNAUTHORIZED);
        assert_eq!(denied("principal_not_granted").0, StatusCode::FORBIDDEN);
    }

    #[test]
    fn a_bearer_token_is_read_from_the_header() {
        assert_eq!(bearer("Bearer abc123"), Some("abc123"));
        // Case-insensitive scheme: RFC 6750 says the scheme is
        // case-insensitive and clients differ.
        assert_eq!(bearer("bearer abc123"), Some("abc123"));
        assert_eq!(bearer("Basic abc123"), None);
        assert_eq!(bearer("abc123"), None);
        assert_eq!(bearer("Bearer "), None);
    }

    // The four tests above cover `require`, `denied`, and `bearer` in
    // isolation — but `granted()` builds a `Caller` directly, bypassing the
    // extractor entirely. Nothing above exercises the ASSEMBLED path:
    // `Authorization` header -> `bearer` parse -> `Registry::load(&state.
    // registry_dir)` -> `Caller`. These three drive `from_request_parts`
    // itself, with no server: build `Parts`, call it, `.await` it.

    #[tokio::test]
    async fn a_valid_token_resolves_to_its_own_principal_not_merely_a_principal() {
        // Two principals, not one. With a single-principal fixture, "some
        // caller resolved" and "the RIGHT caller resolved" are the same
        // observation — a mutation that returned the wrong principal, or
        // always the first one in the directory listing, would pass a
        // one-principal test. Two tokens, each asserted against its OWN
        // name, is the only way to see that distinction (the same lesson
        // Task 2's `two_principals_each_token_resolves_to_its_own_name`
        // closed for `Registry::authenticate` directly).
        let dir = tempfile::tempdir().unwrap();
        let token_iphone = mint_token();
        let token_ipad = mint_token();
        write_principal(
            dir.path(),
            &Principal {
                name: "iphone".to_string(),
                token_hash: hash_token(&token_iphone),
                grants: vec![Grant::Sync],
                minted: "2026-08-07T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        write_principal(
            dir.path(),
            &Principal {
                name: "ipad".to_string(),
                token_hash: hash_token(&token_ipad),
                grants: vec![Grant::Push],
                minted: "2026-08-07T00:00:00Z".to_string(),
            },
        )
        .unwrap();

        let state = state_with_registry(dir.path().to_path_buf());

        let mut p1 = parts(Some(&format!("Bearer {token_iphone}")));
        let caller1 = <Caller as FromRequestParts<SharedState>>::from_request_parts(&mut p1, &state)
            .await
            .expect("iphone's token must authenticate");
        assert_eq!(caller1.0.name, "iphone");

        let mut p2 = parts(Some(&format!("Bearer {token_ipad}")));
        let caller2 = <Caller as FromRequestParts<SharedState>>::from_request_parts(&mut p2, &state)
            .await
            .expect("ipad's token must authenticate");
        assert_eq!(caller2.0.name, "ipad");
    }

    #[tokio::test]
    async fn no_authorization_header_is_401_no_principal() {
        // Never reaches the registry — the header check comes first (the
        // ordering this module's own doc argues for) — so a fixed
        // nonexistent path is honest here, matching `models_path`'s sibling
        // convention rather than paying for a tempdir this test never reads.
        let state = state_with_registry(std::path::PathBuf::from("/nonexistent/registry"));
        let mut p = parts(None);
        let err = <Caller as FromRequestParts<SharedState>>::from_request_parts(&mut p, &state)
            .await
            .unwrap_err();
        assert_eq!(err.0, StatusCode::UNAUTHORIZED);
        assert!(err.1.contains("no_principal"), "{}", err.1);
    }

    #[tokio::test]
    async fn a_token_in_no_registry_is_401_unknown_principal() {
        let dir = tempfile::tempdir().unwrap();
        // The directory exists but holds no principal matching this token —
        // distinct from a directory that does not exist at all, and the
        // case an actual revoked-token caller hits.
        let state = state_with_registry(dir.path().to_path_buf());
        let mut p = parts(Some(&format!("Bearer {}", mint_token())));
        let err = <Caller as FromRequestParts<SharedState>>::from_request_parts(&mut p, &state)
            .await
            .unwrap_err();
        assert_eq!(err.0, StatusCode::UNAUTHORIZED);
        assert!(err.1.contains("unknown_principal"), "{}", err.1);
    }
}

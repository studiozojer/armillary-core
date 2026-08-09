//! Fetch, pull and push as durable events.
//!
//! # Why a failure emits too
//!
//! If only successes were recorded, the ABSENCE of an event would mean both
//! "nothing happened" and "it failed" — the same conflation `last_fetch`
//! shipped when a failed fetch truncated `FETCH_HEAD` and made the row read
//! "fetched just now". `result` is a field of the event; it is never the
//! event's existence.
//!
//! # Why one stream and not twenty-five
//!
//! `workspace` carries host-level facts — things no instance owns. One stream
//! per repo would make "what did this device do today" the cross-stream index
//! `constitution/instances.md` § 5 explicitly defers, and would create a
//! stream per composed repo whether or not anything ever happened in it.
//!
//! # A write that fails here does not fail the verb
//!
//! The push already happened. Refusing the HTTP response because the log write
//! failed would report a failure that did not occur — the same defect class
//! this module exists to close, pointing the other way. I-5 requires the
//! failure to surface to the writer, so it is logged loudly to stderr, but the
//! verb's own result is reported honestly regardless.

use crate::auth::Caller;
use crate::git::PushReport;
use crate::log::envelope::{Actor, ActorPrincipal, Role};
use crate::repos;
use crate::sessions::{NewEvent, Sessions};
use serde_json::json;

/// The one stream for host-level facts. Served by the existing generic
/// `GET /streams/{stream}/events`, so a client tails it through A-1's
/// primitive with no new mechanism.
pub const WORKSPACE_STREAM: &str = "workspace";

/// `{role: machine, principal: <who asked>}` — the performer and the
/// requester, in the one place I-2 says "who did this" must live.
fn actor(caller: &Caller) -> Actor {
    Actor {
        role: Role::Machine,
        instance: None,
        principal: Some(ActorPrincipal { name: caller.0.name.clone() }),
    }
}

/// `"ok"`, or the typed error.
fn result(err: Option<&repos::ActionError>) -> serde_json::Value {
    match err {
        None => json!("ok"),
        Some(e) => json!({ "error": { "kind": e.kind, "message": e.message } }),
    }
}

/// Append, or say so loudly. Never propagates: see the module doc.
fn emit(sessions: &Sessions, event_type: &str, actor: Actor, data: serde_json::Value) {
    if let Err(e) = sessions.append(
        WORKSPACE_STREAM,
        NewEvent { actor, event_type: event_type.to_string(), data },
    ) {
        eprintln!("error: {event_type} happened but could not be recorded — {e:?}");
    }
}

/// Insert only when known.
///
/// An ABSENT field says "we do not know"; a `null` or a zero says "we measured
/// nothing", which is a different and false claim. Every optional field in
/// this module goes through here so no site can quietly choose otherwise.
fn insert_if_known(data: &mut serde_json::Value, key: &str, value: Option<&str>) {
    if let Some(v) = value {
        data[key] = json!(v);
    }
}

pub fn record_fetch(
    sessions: &Sessions,
    caller: &Caller,
    repo: &str,
    err: Option<&repos::ActionError>,
) {
    emit(
        sessions,
        "repo_fetched",
        actor(caller),
        json!({ "repo": repo, "result": result(err) }),
    );
}

pub fn record_pull(
    sessions: &Sessions,
    caller: &Caller,
    repo: &str,
    before: Option<&str>,
    after: Option<&str>,
    err: Option<&repos::ActionError>,
) {
    let mut data = json!({ "repo": repo, "result": result(err) });
    insert_if_known(&mut data, "before", before);
    insert_if_known(&mut data, "after", after);
    emit(sessions, "repo_pulled", actor(caller), data);
}

pub fn record_push(
    sessions: &Sessions,
    caller: &Caller,
    repo: &str,
    report: Option<&PushReport>,
    commits: Option<u32>,
    host: &str,
    err: Option<&repos::ActionError>,
) {
    let mut data = json!({
        "repo": repo,
        // The engine holds no credential: it shells out to git, which
        // authenticates through the host user's own SSH agent and keychain.
        // This field is the sentence "published under David's credential, by a
        // machine, at a device's request" made structural rather than
        // inferable.
        "executed_as": { "host": host, "credential": "host-user-git" },
        "result": result(err),
    });
    if let Some(r) = report {
        insert_if_known(&mut data, "ref", r.reference.as_deref());
        insert_if_known(&mut data, "before", r.before.as_deref());
        insert_if_known(&mut data, "after", r.after.as_deref());
    }
    if let Some(n) = commits {
        data["commits"] = json!(n);
    }
    emit(sessions, "repo_pushed", actor(caller), data);
}

/// A commit's record. No `executed_as`: nothing leaves the machine
/// (`repo_pulled`'s proportionality, not `repo_pushed`'s). `subject` is the
/// message's first line — the full message lives in git; the event indexes it.
pub fn record_commit(
    sessions: &Sessions,
    caller: &Caller,
    repo: &str,
    before: Option<&str>,
    after: Option<&str>,
    subject: Option<&str>,
    files: Option<u32>,
    err: Option<&repos::ActionError>,
) {
    let mut data = json!({ "repo": repo, "result": result(err) });
    insert_if_known(&mut data, "before", before);
    insert_if_known(&mut data, "after", after);
    insert_if_known(&mut data, "subject", subject);
    if let Some(n) = files {
        data["files"] = json!(n);
    }
    emit(sessions, "repo_committed", actor(caller), data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::store::LogStore;
    use crate::principals::{hash_token, mint_token, Grant, Principal};

    fn sessions() -> (Sessions, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        (Sessions::new(LogStore::open(dir.path()).unwrap()), dir)
    }

    fn caller() -> Caller {
        Caller(Principal {
            name: "iphone".to_string(),
            token_hash: hash_token(&mint_token()),
            grants: vec![Grant::Push],
            minted: "2026-08-07T00:00:00Z".to_string(),
        })
    }

    #[test]
    fn a_push_records_all_three_parties() {
        let (s, _d) = sessions();
        record_push(
            &s,
            &caller(),
            "kairos-engine",
            Some(&PushReport {
                reference: Some("refs/heads/main".to_string()),
                before: Some("9f2a1c4".to_string()),
                after: Some("3e7b0d8".to_string()),
            }),
            Some(3),
            "benatky",
            None,
        );

        let evs = s.store().read_from(WORKSPACE_STREAM, 0).unwrap();
        assert_eq!(evs.len(), 1);
        let ev = &evs[0];
        assert_eq!(ev.event_type, "repo_pushed");
        // performer
        assert_eq!(ev.actor.role, Role::Machine);
        // requester
        assert_eq!(ev.actor.principal.as_ref().unwrap().name, "iphone");
        // credential
        assert_eq!(ev.data["executed_as"]["host"], "benatky");
        assert_eq!(ev.data["executed_as"]["credential"], "host-user-git");
        // subject
        assert_eq!(ev.data["repo"], "kairos-engine");
        assert_eq!(ev.data["ref"], "refs/heads/main");
        assert_eq!(ev.data["before"], "9f2a1c4");
        assert_eq!(ev.data["after"], "3e7b0d8");
        assert_eq!(ev.data["commits"], 3);
        assert_eq!(ev.data["result"], "ok");
    }

    #[test]
    fn a_failed_push_records_an_event_too() {
        // The premise of § 2.1: if only successes are recorded, "no event"
        // means both "nothing happened" and "it failed" — the same conflation
        // `last_fetch` shipped. Result is a FIELD, never the event's existence.
        let (s, _d) = sessions();
        let err = repos::ActionError {
            kind: "not-fast-forwardable",
            message: "rejected".to_string(),
        };
        record_push(&s, &caller(), "kairos-engine", None, None, "benatky", Some(&err));

        let evs = s.store().read_from(WORKSPACE_STREAM, 0).unwrap();
        assert_eq!(evs.len(), 1, "a failure is recorded, not skipped");
        assert_eq!(evs[0].data["result"]["error"]["kind"], "not-fast-forwardable");
        // And the credential is still named: it was spent on the attempt.
        assert_eq!(evs[0].data["executed_as"]["host"], "benatky");
    }

    #[test]
    fn a_new_branch_push_omits_before_and_commits_rather_than_zeroing_them() {
        let (s, _d) = sessions();
        record_push(
            &s,
            &caller(),
            "zhouyi",
            Some(&PushReport {
                reference: Some("refs/heads/feature/x".to_string()),
                before: None,
                after: None,
            }),
            None,
            "benatky",
            None,
        );
        let ev = &s.store().read_from(WORKSPACE_STREAM, 0).unwrap()[0];
        assert!(ev.data.get("before").is_none(), "absent, never 000000…");
        assert!(ev.data.get("after").is_none());
        assert!(ev.data.get("commits").is_none(), "no range, no count");
        assert_eq!(ev.data["ref"], "refs/heads/feature/x");
    }

    #[test]
    fn a_fetch_carries_no_credential_because_nothing_leaves_the_machine() {
        // Proportion, from § 2.6. `executed_as` is push's alone; putting it on
        // every verb would make the record claim a credential was spent when
        // none was.
        let (s, _d) = sessions();
        record_fetch(&s, &caller(), "daoUI", None);
        let ev = &s.store().read_from(WORKSPACE_STREAM, 0).unwrap()[0];
        assert_eq!(ev.event_type, "repo_fetched");
        assert!(ev.data.get("executed_as").is_none());
        assert_eq!(ev.data["result"], "ok");
    }

    #[test]
    fn a_pull_records_the_shas_it_moved_between_and_no_credential() {
        let (s, _d) = sessions();
        record_pull(&s, &caller(), "daoUI", Some("aaaaaaa"), Some("bbbbbbb"), None);
        let ev = &s.store().read_from(WORKSPACE_STREAM, 0).unwrap()[0];
        assert_eq!(ev.event_type, "repo_pulled");
        assert_eq!(ev.data["before"], "aaaaaaa");
        assert_eq!(ev.data["after"], "bbbbbbb");
        assert!(ev.data.get("executed_as").is_none(), "nothing left the machine");
    }

    #[test]
    fn a_pull_on_an_unborn_branch_omits_the_shas_rather_than_nulling_them() {
        // `head_sha` answers `None` in a repo with no commits. An absent field
        // says "we do not know"; a null would say "we looked and there was
        // nothing", which is a different claim about a different world.
        let (s, _d) = sessions();
        record_pull(&s, &caller(), "fresh", None, None, None);
        let ev = &s.store().read_from(WORKSPACE_STREAM, 0).unwrap()[0];
        assert!(ev.data.get("before").is_none());
        assert!(ev.data.get("after").is_none());
        assert!(
            !ev.data.to_string().contains("null"),
            "no field may be serialized as null: {}",
            ev.data
        );
    }

    #[test]
    fn every_verb_lands_in_the_one_workspace_stream_in_order() {
        // One stream, not one per repo — and the seq ordering is what makes
        // "what did this device do today" a single read.
        let (s, _d) = sessions();
        record_fetch(&s, &caller(), "a", None);
        record_pull(&s, &caller(), "b", None, None, None);
        record_push(&s, &caller(), "c", None, None, "benatky", None);

        let evs = s.store().read_from(WORKSPACE_STREAM, 0).unwrap();
        let types: Vec<&str> = evs.iter().map(|e| e.event_type.as_str()).collect();
        assert_eq!(types, vec!["repo_fetched", "repo_pulled", "repo_pushed"]);
        assert!(
            evs.iter().all(|e| e.actor.principal.as_ref().unwrap().name == "iphone"),
            "every verb names its requester"
        );
    }

    #[test]
    fn record_commit_carries_principal_shas_subject_and_count() {
        let (s, _d) = sessions();
        record_commit(
            &s,
            &caller(),
            "zojercommons",
            Some("aaa"),
            Some("bbb"),
            Some("subject"),
            Some(3),
            None,
        );

        let evs = s.store().read_from(WORKSPACE_STREAM, 0).unwrap();
        let ev = &evs[0];
        assert_eq!(ev.event_type, "repo_committed");
        assert_eq!(ev.actor.principal.as_ref().unwrap().name, "iphone");
        assert_eq!(ev.data["before"], "aaa");
        assert_eq!(ev.data["after"], "bbb");
        assert_eq!(ev.data["subject"], "subject");
        assert_eq!(ev.data["files"], 3);
        assert_eq!(ev.data["result"], "ok");
    }

    #[test]
    fn record_commit_failure_emits_too_with_absent_optionals() {
        let (s, _d) = sessions();
        let err = repos::ActionError { kind: "nothing-to-commit", message: "clean".into() };
        record_commit(&s, &caller(), "r", Some("aaa"), Some("aaa"), None, Some(0), Some(&err));

        let ev = &s.store().read_from(WORKSPACE_STREAM, 0).unwrap()[0];
        assert_eq!(ev.data["result"]["error"]["kind"], "nothing-to-commit");
        assert!(
            ev.data.get("subject").is_none(),
            "an absent field says 'not known', never null"
        );
    }
}

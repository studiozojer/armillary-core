use crate::log::envelope::{Actor, EventEnvelope, Role};
use crate::loop_::title_from_events;
use crate::sessions::NewEvent;
use crate::state::SharedState;
use std::sync::Arc;
use std::time::Duration;

const TITLE_DAEMON_WINDOW: usize = 10;
const TITLE_INTERVAL: Duration = Duration::from_secs(60);
const TITLE_TIMEOUT: Duration = Duration::from_secs(15);

struct TitleGuard {
    sessions: Arc<crate::sessions::Sessions>,
    stream: String,
}

impl Drop for TitleGuard {
    fn drop(&mut self) {
        self.sessions.end_title(&self.stream);
    }
}

pub(crate) fn schedule(state: SharedState, stream: String, operator: String, model: String) {
    tokio::spawn(async move {
        let sessions = state.sessions.clone();
        let event_stream = stream.clone();
        let events =
            match tokio::task::spawn_blocking(move || sessions.store().read_from(&event_stream, 0))
                .await
            {
                Ok(Ok(events)) => events,
                _ => {
                    eprintln!("daemon_title: snapshot_failed stream={stream:?}");
                    return;
                }
            };
        let Some(cancel) = state.sessions.begin_title(
            &stream,
            title_from_events(&events).is_some(),
            TITLE_INTERVAL,
        ) else {
            return;
        };
        let _guard = TitleGuard {
            sessions: state.sessions.clone(),
            stream: stream.clone(),
        };
        let started = tokio::time::Instant::now();
        daemon_turn(&state, &stream, &operator, &model, &events, cancel).await;
        eprintln!(
            "title_timing {}",
            serde_json::json!({"stream": stream, "elapsed_ms": started.elapsed().as_millis()})
        );
    });
}

fn build_title_prompt(events: &[EventEnvelope], current_title: Option<&str>) -> String {
    let recent: Vec<&EventEnvelope> = events
        .iter()
        .rev()
        .filter(|event| {
            matches!(
                event.event_type.as_str(),
                "user_message" | "assistant_message"
            )
        })
        .take(TITLE_DAEMON_WINDOW)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    let mut conversation = String::new();
    for ev in recent {
        match ev.event_type.as_str() {
            "user_message" => {
                if let Some(text) = ev.data.get("text").and_then(|v| v.as_str()) {
                    conversation.push_str(&format!("User: {}\n", text));
                }
            }
            "assistant_message" => {
                if let Some(text) = ev.data.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        conversation.push_str(&format!("Assistant: {}\n", text));
                    }
                }
            }
            _ => {}
        }
    }

    let current = current_title.unwrap_or("(none yet)");
    format!(
        "You are a title-generating daemon for an AI work session. Your job: assess whether this session's title needs updating, and return the title that should be shown.\n\n\
         The CURRENT title is: {current}\n\n\
         Read the recent conversation below against that title:\n\
         - If the conversation continues or confirms the current topic, return the CURRENT title VERBATIM — byte-for-byte unchanged. Do not rephrase it.\n\
         - If the conversation has shifted to a new topic, return a short, specific new title (1-6 words): \"Debugging the auth flow\" not \"Working on code.\"\n\
         - Prefer keeping a good existing title over churning on wording.\n\
         - This session always has a title — if none exists yet, propose one from the conversation rather than returning the empty string.\n\n\
         Return ONLY the title as plain text, nothing else.\n\n\
         Recent conversation:\n{}",
        conversation
    )
}

/// Append the daemon's heartbeat (observability design 2026-08-19 D2): one
/// `daemon_pulse` per run, WHATEVER the run did — "checked and unchanged" is
/// a record, not a non-event. Threaded `daemon-title` like the rename, so
/// the projection's thread filter keeps it out of model context; the
/// `inspect_daemons` verb reads it back. A failed append is logged and
/// swallowed: the pulse observes the daemon, it must never fail the daemon.
///
/// `token_cost` is deliberately absent for now — `TurnOutcome` does not yet
/// surface the provider's usage frames, and inventing a number here would be
/// worse than omitting the field the design already marks optional.
fn append_pulse(
    state: &SharedState,
    stream: &str,
    operator: &str,
    disposition: &str,
    title: &str,
    previous_title: &str,
    error: Option<String>,
) {
    let mut data = serde_json::json!({
        "daemon": "title",
        "disposition": disposition,
        "title": title,
        "previous_title": previous_title,
    });
    if let Some(e) = error {
        data["error"] = serde_json::json!(e);
    }
    let pulse = NewEvent {
        actor: Actor {
            role: Role::Machine,
            instance: Some(operator.to_string()),
            principal: None,
        },
        event_type: "daemon_pulse".to_string(),
        data,
    };
    match state
        .sessions
        .append_threaded(stream, pulse, "daemon-title")
    {
        Ok(_) => {
            eprintln!("daemon_title: pulse stream={stream:?} disposition={disposition}");
        }
        Err(e) => {
            eprintln!("daemon_title: pulse_append_failed stream={stream:?} error={e:?}");
        }
    }
}

pub async fn daemon_turn(
    state: &SharedState,
    stream: &str,
    operator: &str,
    model: &str,
    events: &[EventEnvelope],
    mut cancel: tokio::sync::watch::Receiver<bool>,
) -> Option<String> {
    eprintln!("daemon_title: starting stream={stream:?}");
    let current_title = title_from_events(events);
    let prompt = build_title_prompt(events, current_title.as_deref());

    let provider = state.providers.provider_for(model);

    let turn = crate::projection::ModelTurn {
        system: None,
        messages: vec![crate::projection::ProviderMessage {
            role: crate::projection::ProviderRole::User,
            content: vec![crate::projection::ContentBlock::Text(prompt)],
        }],
    };

    let req = crate::provider::TurnRequest {
        turn,
        tools: vec![],
        tool_choice: None,
    };

    // The daemon wants the OUTCOME, not the stream — but `run_turn`'s
    // contract requires a live sink, and a provider sends every fragment into
    // it with a blocking `send().await`. A bounded channel nobody drains
    // until after `run_turn` returns is therefore a deadlock on the second
    // fragment (found via `conformance_log`'s scripted turn, which streams
    // two): the drain must run CONCURRENTLY with the turn. The task ends by
    // itself — `run_turn` takes `tx` by value, so the channel closes when
    // the provider returns.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(16);
    tokio::spawn(async move { while rx.recv().await.is_some() {} });
    if *cancel.borrow() {
        return None;
    }
    let provider_cancel = cancel.clone();
    let result = tokio::select! {
        biased;
        _ = cancel.changed() => {
            append_pulse(state, stream, operator, "cancelled", "", current_title.as_deref().unwrap_or_default(), None);
            return None;
        }
        result = tokio::time::timeout(TITLE_TIMEOUT, provider.run_turn(req, tx, provider_cancel)) => result,
    };
    let outcome = match result {
        Err(_) => {
            append_pulse(
                state,
                stream,
                operator,
                "error",
                "",
                current_title.as_deref().unwrap_or_default(),
                Some("title_timeout".to_string()),
            );
            return None;
        }
        Ok(Ok(outcome)) if !outcome.stopped => outcome,
        Ok(Ok(_)) => return None,
        Ok(Err(e)) => {
            eprintln!("daemon_title: model_call_failed stream={stream:?} error={e:?}");
            // The heartbeat fires on failure too — an errored run that left
            // no pulse would be indistinguishable from a run that never
            // happened, which is the exact gap this event exists to close.
            append_pulse(
                state,
                stream,
                operator,
                "error",
                "",
                &current_title.clone().unwrap_or_default(),
                Some(format!("{e:?}")),
            );
            return None;
        }
    };

    let title = outcome.text.trim().to_string();

    // Disposition decided BEFORE the rename guard, because the pulse records
    // all four outcomes and only one of them also renames.
    //
    // Always-strive hardening (review directive 2026-08-19): a blank model
    // response must NEVER leave a titled session untitled. When a title
    // already exists and the model returns empty, that empty is a failure,
    // not a judgment — keep the existing title (recorded "unchanged", so the
    // next good turn can still correct a genuinely shifted topic). A session
    // can only be truly untitled when there is *nothing* to title yet, which
    // is exactly the state the "always propose one" instruction is meant to
    // exit on its first substantive turn.
    if title.is_empty() {
        if let Some(existing) = current_title {
            eprintln!("daemon_title: empty_but_kept stream={stream:?} title={existing:?}");
            append_pulse(
                state,
                stream,
                operator,
                "unchanged",
                &existing,
                &existing,
                None,
            );
            return Some(existing);
        }
        eprintln!("daemon_title: still_untitled stream={stream:?}");
        append_pulse(state, stream, operator, "empty", "", "", None);
        return None;
    }
    if current_title.as_deref() == Some(&title) {
        eprintln!("daemon_title: unchanged stream={stream:?} title={title:?}");
        append_pulse(state, stream, operator, "unchanged", &title, &title, None);
        return None;
    }

    // Kept as an Option: `instance_renamed` has always serialized a missing
    // previous title as `null`, and the pulse (a new event) flattens it to ""
    // without touching the older event's wire shape.
    let previous_title = current_title;
    let ev = NewEvent {
        actor: Actor {
            role: Role::Machine,
            instance: Some(operator.to_string()),
            principal: None,
        },
        event_type: "instance_renamed".to_string(),
        data: serde_json::json!({
            "title": title,
            "previous_title": previous_title,
        }),
    };

    let expected_user = events
        .iter()
        .rev()
        .find(|event| event.event_type == "user_message");
    match state.sessions.append_title_if_current(
        stream,
        expected_user.map(|event| event.id.as_str()),
        previous_title.as_deref(),
        ev,
    ) {
        Ok(Some(_)) => {
            eprintln!("daemon_title: wrote stream={stream:?} title={title:?}");
            append_pulse(
                state,
                stream,
                operator,
                "updated",
                &title,
                previous_title.as_deref().unwrap_or_default(),
                None,
            );
            Some(title)
        }
        Ok(None) => {
            append_pulse(
                state,
                stream,
                operator,
                "stale",
                "",
                previous_title.as_deref().unwrap_or_default(),
                None,
            );
            None
        }
        Err(e) => {
            eprintln!("daemon_title: append_instance_renamed_failed stream={stream:?} error={e:?}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{self, ModelProvider, ProviderError, TurnOutcome, TurnRequest};
    use crate::sessions::{Sessions, TurnHandle};
    use crate::state::{AppState, ModelConfig};
    use tokio::sync::{mpsc, watch};

    fn event(kind: &str, data: serde_json::Value) -> NewEvent {
        NewEvent {
            actor: Actor {
                role: Role::Machine,
                instance: None,
                principal: None,
            },
            event_type: kind.to_string(),
            data,
        }
    }

    fn fixture(provider: Arc<dyn ModelProvider>) -> (tempfile::TempDir, SharedState) {
        let directory = tempfile::tempdir().unwrap();
        let state = Arc::new(AppState {
            root: directory.path().canonicalize().unwrap(),
            sessions: Arc::new(Sessions::new(
                crate::log::store::LogStore::open(directory.path()).unwrap(),
            )),
            model: ModelConfig {
                model: "scripted".to_string(),
            },
            providers: provider::fixed(provider),
            models_path: directory.path().join("models.toml"),
            hostname: "test-host".to_string(),
            registry_dir: directory.path().join("registry"),
            anthropic_key_present: false,
            zen_key_present: false,
            boot: None,
        });
        state
            .sessions
            .append(
                "session",
                event("user_message", serde_json::json!({"text": "question"})),
            )
            .unwrap();
        (directory, state)
    }

    struct NeverAnswers;

    #[async_trait::async_trait]
    impl ModelProvider for NeverAnswers {
        async fn run_turn(
            &self,
            _request: TurnRequest,
            _sink: mpsc::Sender<String>,
            _cancel: watch::Receiver<bool>,
        ) -> Result<TurnOutcome, ProviderError> {
            std::future::pending().await
        }
    }

    #[tokio::test(start_paused = true)]
    async fn title_timeout_records_failure_and_releases_slot() {
        let (_directory, state) = fixture(Arc::new(NeverAnswers));
        let events = state.sessions.store().read_from("session", 0).unwrap();
        let cancel = state
            .sessions
            .begin_title("session", false, TITLE_INTERVAL)
            .unwrap();
        let started = tokio::time::Instant::now();
        {
            let _guard = TitleGuard {
                sessions: state.sessions.clone(),
                stream: "session".to_string(),
            };
            assert_eq!(
                daemon_turn(&state, "session", "operator", "scripted", &events, cancel).await,
                None
            );
        }
        assert_eq!(started.elapsed(), TITLE_TIMEOUT);
        let events = state.sessions.store().read_from("session", 0).unwrap();
        let pulse = events.last().unwrap();
        assert_eq!(pulse.event_type, "daemon_pulse");
        assert_eq!(pulse.data["error"], "title_timeout");
        assert!(!events
            .iter()
            .any(|event| event.event_type == "instance_renamed"));
        assert!(state
            .sessions
            .begin_title("session", false, TITLE_INTERVAL)
            .is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn titles_are_single_flight_throttled_and_yield_to_foreground() {
        let (_directory, state) = fixture(Arc::new(NeverAnswers));
        let sessions = &state.sessions;
        let cancel = sessions
            .begin_title("session", false, TITLE_INTERVAL)
            .unwrap();
        assert!(sessions
            .begin_title("session", false, TITLE_INTERVAL)
            .is_none());
        let (sender, _receiver) = watch::channel(false);
        sessions
            .begin_turn(
                "session",
                TurnHandle {
                    cancel: sender,
                    generation: "next".to_string(),
                },
            )
            .unwrap();
        assert!(*cancel.borrow());
        sessions.end_title("session");
        assert!(sessions
            .begin_title("session", false, TITLE_INTERVAL)
            .is_none());
        sessions.end_turn("session");
        assert!(sessions
            .begin_title("session", true, TITLE_INTERVAL)
            .is_none());
        tokio::time::advance(TITLE_INTERVAL).await;
        assert!(sessions
            .begin_title("session", true, TITLE_INTERVAL)
            .is_some());
    }

    #[tokio::test]
    async fn stale_title_cannot_overwrite_new_user_or_manual_title() {
        let (_directory, state) =
            fixture(Arc::new(provider::ScriptedProvider::new(vec!["new title"])));
        let sessions = &state.sessions;
        let events = sessions.store().read_from("session", 0).unwrap();
        let first_user = &events[0].id;
        let second = sessions
            .append(
                "session",
                event("user_message", serde_json::json!({"text": "next"})),
            )
            .unwrap();
        let rename = || {
            event(
                "instance_renamed",
                serde_json::json!({"title": "generated"}),
            )
        };
        assert!(sessions
            .append_title_if_current("session", Some(first_user), None, rename())
            .unwrap()
            .is_none());
        sessions
            .append(
                "session",
                event("instance_renamed", serde_json::json!({"title": "manual"})),
            )
            .unwrap();
        assert!(sessions
            .append_title_if_current("session", Some(&second.id), None, rename())
            .unwrap()
            .is_none());
        assert_eq!(
            title_from_events(&sessions.store().read_from("session", 0).unwrap()).as_deref(),
            Some("manual")
        );
        assert!(sessions
            .append_title_if_current("session", Some(&second.id), Some("manual"), rename())
            .unwrap()
            .is_some());
    }

    #[test]
    fn title_prompt_keeps_recent_dialogue_despite_tool_noise() {
        let (_directory, state) = fixture(Arc::new(NeverAnswers));
        for index in 0..12 {
            state
                .sessions
                .append(
                    "session",
                    event(
                        "assistant_message",
                        serde_json::json!({"text": format!("answer-{index:02}")}),
                    ),
                )
                .unwrap();
            for _ in 0..12 {
                state
                    .sessions
                    .append(
                        "session",
                        event("tool_result", serde_json::json!({"content": "tool noise"})),
                    )
                    .unwrap();
            }
        }
        let events = state.sessions.store().read_from("session", 0).unwrap();
        let prompt = build_title_prompt(&events, None);
        assert!(!prompt.contains("answer-01"));
        assert!(prompt.contains("answer-02"));
        assert!(prompt.contains("answer-11"));
        assert!(!prompt.contains("tool noise"));
    }
}

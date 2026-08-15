use crate::log::envelope::{Actor, EventEnvelope, Role};
use crate::loop_::title_from_events;
use crate::sessions::NewEvent;
use crate::state::SharedState;

const TITLE_DAEMON_WINDOW: usize = 10;

fn build_title_prompt(events: &[EventEnvelope]) -> String {
    let recent: Vec<&EventEnvelope> = events
        .iter()
        .rev()
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

    format!(
        "You are a title-generating daemon for an AI work session. Your ONLY job: read the conversation below and return a short, descriptive title (1-6 words) that captures what this session is about.\n\n\
         Rules:\n\
         - Be specific, not generic. \"Debugging the auth flow\" not \"Working on code.\"\n\
         - If the topic shifts, update the title to reflect the new focus.\n\
         - If you can't determine a topic, return the empty string.\n\
         - Return ONLY the title as plain text, nothing else.\n\n\
         Conversation:\n{}",
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
    match state.sessions.append_threaded(stream, pulse, "daemon-title") {
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
) -> Option<String> {
    eprintln!("daemon_title: starting stream={stream:?}");
    let prompt = build_title_prompt(events);
    let current_title = title_from_events(events);

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
    let cancel = tokio::sync::watch::channel(false).1;

    let outcome = match provider.run_turn(req, tx, cancel).await {
        Ok(outcome) => outcome,
        Err(e) => {
            eprintln!(
                "daemon_title: model_call_failed stream={stream:?} error={e:?}"
            );
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
    if title.is_empty() {
        eprintln!("daemon_title: empty_title stream={stream:?}");
        append_pulse(
            state,
            stream,
            operator,
            "empty",
            "",
            &current_title.unwrap_or_default(),
            None,
        );
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

    let sessions = &state.sessions;
    match sessions.append_threaded(stream, ev, "daemon-title") {
        Ok(_) => {
            eprintln!(
                "daemon_title: wrote stream={stream:?} title={title:?}"
            );
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
        Err(e) => {
            eprintln!(
                "daemon_title: append_instance_renamed_failed stream={stream:?} error={e:?}"
            );
            None
        }
    }
}

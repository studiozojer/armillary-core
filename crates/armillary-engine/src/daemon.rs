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

    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(1);
    let cancel = tokio::sync::watch::channel(false).1;

    let outcome = match provider.run_turn(req, tx, cancel).await {
        Ok(outcome) => outcome,
        Err(e) => {
            eprintln!(
                "daemon_title: model_call_failed stream={stream:?} error={e:?}"
            );
            return None;
        }
    };

    while rx.try_recv().is_ok() {}

    let title = outcome.text.trim().to_string();
    if title.is_empty() {
        eprintln!("daemon_title: empty_title stream={stream:?}");
        return None;
    }
    if current_title.as_deref() == Some(&title) {
        eprintln!("daemon_title: unchanged stream={stream:?} title={title:?}");
        return None;
    }

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

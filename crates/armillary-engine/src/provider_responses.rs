//! OpenAI Responses streaming through OpenCode Zen for its GPT 6 models.
//!
//! The existing Zen chat-completions provider remains the transport for
//! DeepSeek and other compatible models. Responses has different input,
//! function-call, and SSE shapes, so it needs its own wire boundary.

use crate::projection::{ContentBlock, ProviderMessage, ProviderRole};
use crate::provider::{
    parse_sse_data_line, ModelProvider, ProviderError, ToolChoice, TurnOutcome, TurnRequest,
    MAX_TOKENS,
};
use futures_util::StreamExt;
use std::collections::BTreeMap;
use tokio::sync::{mpsc, watch};

pub struct ResponsesProvider {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

impl std::fmt::Debug for ResponsesProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesProvider")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

pub(crate) fn build_responses_request_body(model: &str, req: &TurnRequest) -> serde_json::Value {
    let mut input = Vec::new();
    for message in &req.turn.messages {
        push_input(&mut input, message);
    }
    let mut body = serde_json::json!({
        "model": model,
        "input": input,
        "max_output_tokens": MAX_TOKENS,
        "stream": true,
        "store": false,
    });
    if let Some(system) = &req.turn.system {
        body["instructions"] = serde_json::json!(system);
    }
    if !req.tools.is_empty() {
        body["tools"] = serde_json::Value::Array(
            req.tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.schema,
                        "strict": false,
                    })
                })
                .collect(),
        );
    }
    if matches!(req.tool_choice, Some(ToolChoice::ForceText)) {
        body["tool_choice"] = serde_json::json!("none");
    }
    body
}

fn push_input(input: &mut Vec<serde_json::Value>, message: &ProviderMessage) {
    let role = match message.role {
        ProviderRole::User => "user",
        ProviderRole::Assistant => "assistant",
    };
    for block in &message.content {
        match block {
            ContentBlock::Text(text) => {
                input.push(serde_json::json!({"role": role, "content": text}))
            }
            ContentBlock::ToolUse {
                id,
                name,
                input: args,
            } => input.push(serde_json::json!({
                "type": "function_call",
                "call_id": id,
                "name": name,
                "arguments": serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string()),
            })),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => input.push(serde_json::json!({
                "type": "function_call_output", "call_id": tool_use_id, "output": content,
            })),
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
        }
    }
}

#[derive(Default)]
struct PartialCall {
    call_id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct ResponsesAccumulator {
    text: String,
    calls: BTreeMap<u64, PartialCall>,
    terminal: Option<String>,
}

impl ResponsesAccumulator {
    fn observe(&mut self, event: &serde_json::Value) -> bool {
        let kind = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let index = event.get("output_index").and_then(|v| v.as_u64());
        match kind {
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                    self.text.push_str(delta);
                    return !delta.is_empty();
                }
            }
            "response.output_text.done" => {
                if self.text.is_empty() {
                    if let Some(text) = event.get("text").and_then(|v| v.as_str()) {
                        self.text = text.to_string();
                        return !text.is_empty();
                    }
                }
            }
            "response.output_item.added" | "response.output_item.done" => {
                if let (Some(index), Some(item)) = (index, event.get("item")) {
                    if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                        let call = self.calls.entry(index).or_default();
                        if let Some(id) = item.get("call_id").and_then(|v| v.as_str()) {
                            call.call_id = id.to_string();
                        }
                        if let Some(name) = item.get("name").and_then(|v| v.as_str()) {
                            call.name = name.to_string();
                        }
                        if kind == "response.output_item.done" {
                            if let Some(args) = item.get("arguments").and_then(|v| v.as_str()) {
                                call.arguments = args.to_string();
                            }
                        }
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                if let (Some(index), Some(delta)) =
                    (index, event.get("delta").and_then(|v| v.as_str()))
                {
                    self.calls
                        .entry(index)
                        .or_default()
                        .arguments
                        .push_str(delta);
                }
            }
            "response.function_call_arguments.done" => {
                if let Some(index) = index {
                    let call = self.calls.entry(index).or_default();
                    if let Some(args) = event.get("arguments").and_then(|v| v.as_str()) {
                        call.arguments = args.to_string();
                    }
                    if let Some(name) = event.get("name").and_then(|v| v.as_str()) {
                        call.name = name.to_string();
                    }
                }
            }
            "response.completed" => {
                self.terminal = Some(
                    if self.calls.is_empty() {
                        "end_turn"
                    } else {
                        "tool_use"
                    }
                    .to_string(),
                )
            }
            "response.incomplete" => {
                let reason = event
                    .pointer("/response/incomplete_details/reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("incomplete");
                self.terminal = Some(
                    if reason == "max_output_tokens" {
                        "max_tokens"
                    } else {
                        reason
                    }
                    .to_string(),
                );
            }
            "response.failed" => self.terminal = Some("failed".to_string()),
            _ => {}
        }
        false
    }

    fn blocks(&self) -> Vec<ContentBlock> {
        let mut out = Vec::new();
        if !self.text.is_empty() {
            out.push(ContentBlock::Text(self.text.clone()));
        }
        for call in self.calls.values() {
            if call.call_id.is_empty() || call.name.is_empty() {
                continue;
            }
            let args = if call.arguments.trim().is_empty() {
                Some(serde_json::json!({}))
            } else {
                serde_json::from_str(&call.arguments).ok()
            };
            if let Some(input) = args {
                out.push(ContentBlock::ToolUse {
                    id: call.call_id.clone(),
                    name: call.name.clone(),
                    input,
                });
            }
        }
        out
    }
}

impl ResponsesProvider {
    fn outcome(&self, acc: ResponsesAccumulator, stopped: bool) -> TurnOutcome {
        TurnOutcome {
            text: acc.text.clone(),
            blocks: acc.blocks(),
            stop_reason: acc.terminal,
            stopped,
            model: self.model.clone(),
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for ResponsesProvider {
    async fn run_turn(
        &self,
        req: TurnRequest,
        sink: mpsc::Sender<String>,
        mut cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError> {
        if *cancel.borrow() {
            return Ok(self.outcome(ResponsesAccumulator::default(), true));
        }
        let body = build_responses_request_body(&self.model, &req);
        let response = reqwest::Client::new()
            .post(format!("{}/responses", self.base_url))
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Http(e.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: response.text().await.unwrap_or_default(),
            });
        }
        let mut stream = response.bytes_stream();
        let mut lines = String::new();
        let mut acc = ResponsesAccumulator::default();
        loop {
            tokio::select! {
                biased;
                changed = cancel.changed() => {
                    if changed.is_err() || *cancel.borrow() {
                        return Ok(self.outcome(acc, true));
                    }
                }
                chunk = stream.next() => match chunk {
                    Some(Ok(bytes)) => {
                        lines.push_str(&String::from_utf8_lossy(&bytes));
                        while let Some(index) = lines.find('\n') {
                            let line = lines[..index].trim_end_matches('\r').to_string();
                            lines.drain(..=index);
                            let Some(event) = parse_sse_data_line(&line) else { continue };
                            let changed = acc.observe(&event);
                            if changed {
                                let _ = sink.send(acc.text.clone()).await;
                            }
                            if acc.terminal.as_deref() == Some("failed") {
                                return Err(ProviderError::Http("Responses stream failed".to_string()));
                            }
                            if acc.terminal.is_some() {
                                return Ok(self.outcome(acc, false));
                            }
                        }
                    }
                    Some(Err(e)) => return Err(ProviderError::Http(e.to_string())),
                    None => return Err(ProviderError::Http("Responses stream ended without a terminal event".to_string())),
                }
            }
        }
    }

    fn describe(&self) -> String {
        format!("opencode-zen:{}", self.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::ModelTurn;

    fn text(role: ProviderRole, value: &str) -> ProviderMessage {
        ProviderMessage {
            role,
            content: vec![ContentBlock::Text(value.to_string())],
        }
    }

    #[test]
    fn request_replays_text_and_tool_round_trip_in_responses_shape() {
        let turn = ModelTurn {
            system: Some("boot".to_string()),
            messages: vec![
                text(ProviderRole::User, "read it"),
                ProviderMessage {
                    role: ProviderRole::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "call_1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path":"a.md"}),
                    }],
                },
                ProviderMessage {
                    role: ProviderRole::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "call_1".to_string(),
                        content: "contents".to_string(),
                        is_error: false,
                    }],
                },
            ],
        };
        let body = build_responses_request_body(
            "gpt-6-luna",
            &TurnRequest {
                turn,
                tools: vec![crate::tools::ToolDef {
                    name: "read_file",
                    description: "Read a file".to_string(),
                    schema: serde_json::json!({"type":"object"}),
                }],
                tool_choice: Some(ToolChoice::ForceText),
            },
        );
        assert_eq!(body["instructions"], "boot");
        assert_eq!(
            body["input"],
            serde_json::json!([
                {"role":"user","content":"read it"},
                {"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"a.md\"}"},
                {"type":"function_call_output","call_id":"call_1","output":"contents"},
            ])
        );
        assert_eq!(body["tools"][0]["name"], "read_file");
        assert_eq!(body["tools"][0]["strict"], false);
        assert_eq!(body["tool_choice"], "none");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn stream_collects_text_and_call_then_normalizes_completion() {
        let mut acc = ResponsesAccumulator::default();
        for event in [
            serde_json::json!({"type":"response.output_text.delta","delta":"I will "}),
            serde_json::json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"read_file"}}),
            serde_json::json!({"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"path\":"}),
            serde_json::json!({"type":"response.function_call_arguments.done","output_index":1,"name":"read_file","arguments":"{\"path\":\"a.md\"}"}),
            serde_json::json!({"type":"response.completed"}),
        ] {
            acc.observe(&event);
        }
        assert_eq!(acc.terminal.as_deref(), Some("tool_use"));
        assert_eq!(
            acc.blocks(),
            vec![
                ContentBlock::Text("I will ".to_string()),
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path":"a.md"})
                },
            ]
        );
    }

    #[test]
    fn incomplete_and_truncated_arguments_are_not_silent_successes() {
        let mut acc = ResponsesAccumulator::default();
        acc.observe(&serde_json::json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"read_file"}}));
        acc.observe(&serde_json::json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"path\":"}));
        acc.observe(&serde_json::json!({"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}));
        assert_eq!(acc.terminal.as_deref(), Some("max_tokens"));
        assert!(acc.blocks().is_empty());
    }
}

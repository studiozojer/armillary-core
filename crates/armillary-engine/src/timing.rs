use crate::projection::{ContentBlock, ModelTurn};
use tokio::time::Instant;

pub(crate) struct TurnTiming {
    stream: String,
    generation: String,
    pub started: Instant,
    pub rounds: usize,
    pub projection_ms: u128,
    pub provider_ms: u128,
    pub tool_ms: u128,
    pub first_text_ms: Option<u128>,
}

impl TurnTiming {
    pub fn new(stream: &str, generation: &str) -> Self {
        Self {
            stream: stream.to_string(),
            generation: generation.to_string(),
            started: Instant::now(),
            rounds: 0,
            projection_ms: 0,
            provider_ms: 0,
            tool_ms: 0,
            first_text_ms: None,
        }
    }

    fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "stream": self.stream,
            "generation": self.generation,
            "elapsed_ms": self.started.elapsed().as_millis(),
            "rounds": self.rounds,
            "projection_ms": self.projection_ms,
            "provider_ms": self.provider_ms,
            "tool_ms": self.tool_ms,
            "first_text_ms": self.first_text_ms,
        })
    }
}

impl Drop for TurnTiming {
    fn drop(&mut self) {
        eprintln!("turn_timing {}", self.summary());
    }
}

pub(crate) fn context_content_bytes(turn: &ModelTurn) -> usize {
    turn.system.as_ref().map_or(0, String::len)
        + turn
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .map(|block| match block {
                ContentBlock::Text(text) => text.len(),
                ContentBlock::ToolUse { input, .. } => input.to_string().len(),
                ContentBlock::ToolResult { content, .. } => content.len(),
                ContentBlock::Thinking {
                    thinking,
                    signature,
                } => thinking.len() + signature.len(),
                ContentBlock::RedactedThinking { data } => data.len(),
            })
            .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{ProviderMessage, ProviderRole};

    #[tokio::test(start_paused = true)]
    async fn summary_preserves_missing_first_text_and_elapsed_time() {
        let timing = TurnTiming::new("session", "generation");
        tokio::time::advance(std::time::Duration::from_millis(125)).await;
        let summary = timing.summary();
        assert_eq!(summary["elapsed_ms"], 125);
        assert!(summary["first_text_ms"].is_null());
        assert_eq!(summary["rounds"], 0);
    }

    #[test]
    fn context_size_counts_utf8_and_tool_payloads_without_returning_content() {
        let turn = ModelTurn {
            system: Some("secret".to_string()),
            messages: vec![ProviderMessage {
                role: ProviderRole::User,
                content: vec![
                    ContentBlock::Text("é".to_string()),
                    ContentBlock::ToolResult {
                        tool_use_id: "id".to_string(),
                        content: "body".to_string(),
                        is_error: false,
                    },
                    ContentBlock::ToolUse {
                        id: "id".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({}),
                    },
                    ContentBlock::Thinking {
                        thinking: "abc".to_string(),
                        signature: "sig".to_string(),
                    },
                    ContentBlock::RedactedThinking {
                        data: "opaque".to_string(),
                    },
                ],
            }],
        };
        assert_eq!(context_content_bytes(&turn), 26);
        assert!(!TurnTiming::new("session", "generation")
            .summary()
            .to_string()
            .contains("secret"));
    }
}

//! Model Context Protocol (MCP) server for `armillary-engine`.
//!
//! Exposes armillary's composition, tools, prompts, and resources to Claude Code CLI
//! (or any conforming MCP client) over JSON-RPC 2.0 stdio transport.
//!
//! **The log is the truth (`constitution/instances.md` I-1..I-6)**:
//! Every tool call dispatched through MCP records durable `tool_use` and `tool_result`
//! events into an instance stream in `.armillary/`, preserving full provenance.

use crate::log::envelope::{Actor, Role};
use crate::sessions::{NewEvent, Sessions};
use crate::tools::{self, Effect, ToolCtx, TurnIdentity};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[serde(default)]
    pub jsonrpc: Option<String>,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: serde_json::Value, code: i64, message: impl Into<String>) -> Self {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

pub struct McpServer {
    pub root: PathBuf,
    pub sessions: Arc<Sessions>,
    pub stream: String,
    pub boot: Option<String>,
}

impl McpServer {
    pub fn new(root: PathBuf, sessions: Arc<Sessions>, boot: Option<String>) -> Self {
        let stream = format!("inst-mcp-{}", uuid::Uuid::new_v4());

        // Record initial instance creation event in log
        let composition_data = tools::composition_event_data(&root).unwrap_or(json!({}));
        let _ = sessions.append(
            &stream,
            NewEvent {
                actor: Actor {
                    role: Role::Machine,
                    instance: None,
                    principal: None,
                },
                event_type: "instance_created".to_string(),
                data: json!({
                    "model": "claude-code-cli",
                    "composition": composition_data,
                    "boot": boot,
                }),
            },
        );

        McpServer {
            root,
            sessions,
            stream,
            boot,
        }
    }

    pub fn handle_request(&self, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
        let id = req.id?; // Notifications have no id, return None

        let response = match req.method.as_str() {
            "initialize" => {
                let result = json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": { "listChanged": false },
                        "resources": { "subscribe": false, "listChanged": false },
                        "prompts": { "listChanged": false }
                    },
                    "serverInfo": {
                        "name": "armillary-engine",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                });
                JsonRpcResponse::success(id, result)
            }
            "ping" => JsonRpcResponse::success(id, json!({})),
            "tools/list" => {
                let all_tools = tools::registry().iter().chain(tools::repo_tools());
                let tools_list: Vec<serde_json::Value> = all_tools
                    .map(|t| {
                        json!({
                            "name": t.def.name,
                            "description": t.def.description,
                            "inputSchema": t.def.schema
                        })
                    })
                    .collect();
                JsonRpcResponse::success(id, json!({ "tools": tools_list }))
            }
            "tools/call" => {
                let params = req.params.unwrap_or(json!({}));
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or_default();
                let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

                let result = self.execute_tool(name, &arguments);
                JsonRpcResponse::success(id, result)
            }
            "resources/list" => {
                let mut resources = Vec::new();
                if let Some(boot_file) = &self.boot {
                    resources.push(json!({
                        "uri": format!("armillary://workspace/{}", boot_file),
                        "name": format!("Router Boot: {}", boot_file),
                        "description": "The workspace router boot protocol document",
                        "mimeType": "text/markdown"
                    }));
                }
                resources.push(json!({
                    "uri": "armillary://workspace/composition",
                    "name": "Workspace Composition",
                    "description": "Declared operators, commons, repositories and protocols",
                    "mimeType": "application/json"
                }));
                JsonRpcResponse::success(id, json!({ "resources": resources }))
            }
            "resources/read" => {
                let params = req.params.unwrap_or(json!({}));
                let uri = params.get("uri").and_then(|v| v.as_str()).unwrap_or_default();

                if uri == "armillary://workspace/composition" {
                    let payload = tools::composition_payload(&self.root).unwrap_or(json!({}));
                    JsonRpcResponse::success(id, json!({
                        "contents": [{
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": serde_json::to_string_pretty(&payload).unwrap_or_default()
                        }]
                    }))
                } else if let Some(path_suffix) = uri.strip_prefix("armillary://workspace/") {
                    match crate::guard::resolve(&self.root, path_suffix) {
                        Ok(p) => match std::fs::read_to_string(&p) {
                            Ok(text) => JsonRpcResponse::success(id, json!({
                                "contents": [{
                                    "uri": uri,
                                    "mimeType": "text/markdown",
                                    "text": text
                                }]
                            })),
                            Err(e) => JsonRpcResponse::error(id, -32602, format!("Resource read failed: {e}")),
                        },
                        Err(e) => JsonRpcResponse::error(id, -32602, format!("Guard error: {}", e.code())),
                    }
                } else {
                    JsonRpcResponse::error(id, -32602, format!("Unknown resource URI: {uri}"))
                }
            }
            "prompts/list" => {
                let prompts = vec![
                    json!({
                        "name": "armillary_boot",
                        "description": "Initialize session with armillary workspace boot protocol and path anchors",
                        "arguments": []
                    })
                ];
                JsonRpcResponse::success(id, json!({ "prompts": prompts }))
            }
            "prompts/get" => {
                let mut prompt_text = String::new();
                if let Some(boot_file) = &self.boot {
                    if let Ok(p) = crate::guard::resolve(&self.root, boot_file) {
                        if let Ok(content) = std::fs::read_to_string(&p) {
                            prompt_text.push_str(&content);
                            prompt_text.push_str("\n\n");
                        }
                    }
                }
                prompt_text.push_str(&format!(
                    "Path Anchor (B-4): Workspace root is {}\nAll relative tool paths resolve from this root.\n",
                    self.root.display()
                ));

                JsonRpcResponse::success(id, json!({
                    "description": "Armillary workspace boot prompt",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": prompt_text
                            }
                        }
                    ]
                }))
            }
            other => JsonRpcResponse::error(id, -32601, format!("Method not found: {other}")),
        };

        Some(response)
    }

    fn execute_tool(&self, name: &str, arguments: &serde_json::Value) -> serde_json::Value {
        let tool_call_id = uuid::Uuid::new_v4().to_string();

        // 1. Append durable tool_use event (I-6 / P-2)
        let _ = self.sessions.append(
            &self.stream,
            NewEvent {
                actor: Actor {
                    role: Role::Tool,
                    instance: Some(self.stream.clone()),
                    principal: None,
                },
                event_type: "tool_use".to_string(),
                data: json!({
                    "id": tool_call_id,
                    "name": name,
                    "input": arguments,
                }),
            },
        );

        let ctx = ToolCtx {
            root: self.root.clone(),
            may_write_composition: true,
            turn: TurnIdentity {
                device: "claude-code-cli".to_string(),
                operator: "dispatcher".to_string(),
                model: "claude-code".to_string(),
            },
            instance_events: None,
        };

        // 2. Dispatch tool execution through armillary's guarded switch
        match tools::dispatch(name, arguments, &ctx) {
            Ok(outcome) => {
                // Record durable tool_result event
                let _ = self.sessions.append(
                    &self.stream,
                    NewEvent {
                        actor: Actor {
                            role: Role::Machine,
                            instance: Some(self.stream.clone()),
                            principal: None,
                        },
                        event_type: "tool_result".to_string(),
                        data: json!({
                            "tool_use_id": tool_call_id,
                            "content": outcome.text,
                            "is_error": false,
                            "status": "ok"
                        }),
                    },
                );

                // Record any durable mutations (files or repos)
                for effect in outcome.effects {
                    match effect {
                        Effect::FileChanged { path, op, before, after } => {
                            let _ = self.sessions.append(
                                &self.stream,
                                NewEvent {
                                    actor: Actor {
                                        role: Role::Tool,
                                        instance: Some(self.stream.clone()),
                                        principal: None,
                                    },
                                    event_type: "file_changed".to_string(),
                                    data: json!({
                                        "path": path,
                                        "op": op,
                                        "before": before,
                                        "after": after,
                                    }),
                                },
                            );
                        }
                        Effect::RepoActed {
                            verb,
                            repo,
                            before,
                            after,
                            subject,
                            files,
                            reference,
                            commits,
                            error,
                        } => {
                            let mut data = json!({
                                "repo": repo,
                                "verb": format!("{verb:?}").to_lowercase(),
                                "result": match &error {
                                    None => json!("ok"),
                                    Some(e) => json!({ "error": { "kind": e.kind, "message": e.message } }),
                                }
                            });
                            if let Some(b) = before { data["before"] = json!(b); }
                            if let Some(a) = after { data["after"] = json!(a); }
                            if let Some(s) = subject { data["subject"] = json!(s); }
                            if let Some(f) = files { data["files"] = json!(f); }
                            if let Some(r) = reference { data["reference"] = json!(r); }
                            if let Some(c) = commits { data["commits"] = json!(c); }

                            let _ = self.sessions.append(
                                &self.stream,
                                NewEvent {
                                    actor: Actor {
                                        role: Role::Tool,
                                        instance: Some(self.stream.clone()),
                                        principal: None,
                                    },
                                    event_type: "repo_acted".to_string(),
                                    data,
                                },
                            );
                        }
                    }
                }

                json!({
                    "content": [{ "type": "text", "text": outcome.text }],
                    "isError": false
                })
            }
            Err(err) => {
                // S-1: Sovereign error verdict recorded in log
                let _ = self.sessions.append(
                    &self.stream,
                    NewEvent {
                        actor: Actor {
                            role: Role::Machine,
                            instance: Some(self.stream.clone()),
                            principal: None,
                        },
                        event_type: "tool_result".to_string(),
                        data: json!({
                            "tool_use_id": tool_call_id,
                            "content": err.detail,
                            "is_error": true,
                            "status": err.status
                        }),
                    },
                );

                json!({
                    "content": [{
                        "type": "text",
                        "text": format!("Error ({}): {}", err.status, err.detail)
                    }],
                    "isError": true
                })
            }
        }
    }
}

/// Run the stdio JSON-RPC loop until stdin closes.
pub fn run_stdio(root: &Path, sessions: Arc<Sessions>, boot: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let server = McpServer::new(root.to_path_buf(), sessions, boot);
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Ok(req) = serde_json::from_str::<JsonRpcRequest>(trimmed) {
            if let Some(resp) = server.handle_request(req) {
                let serialized = serde_json::to_string(&resp)?;
                writeln!(stdout, "{serialized}")?;
                stdout.flush()?;
            }
        }
    }

    Ok(())
}

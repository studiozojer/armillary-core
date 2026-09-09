use armillary_engine::mcp::{JsonRpcRequest, McpServer, MCP_PROTOCOL_VERSION};
use armillary_engine::log::store::LogStore;
use armillary_engine::sessions::Sessions;
use serde_json::json;
use std::sync::Arc;
use tempfile::tempdir;

fn setup() -> (tempfile::TempDir, McpServer) {
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join("modules.toml"),
        "[router]\ncontains = [\"CLAUDE.md\"]\nboot = \"boot.md\"\n\n[[commons]]\nname = \"alpha\"\npath = \"commons/alpha\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(dir.path().join("commons/alpha")).unwrap();
    std::fs::write(dir.path().join("boot.md"), "# Boot Protocol\nFollow the standard.").unwrap();
    std::fs::write(dir.path().join("hello.txt"), "Hello, Armillary MCP!").unwrap();

    let data_dir = dir.path().join(".armillary");
    let store = LogStore::open(&data_dir).unwrap();
    let sessions = Arc::new(Sessions::new(store));

    let server = McpServer::new(
        dir.path().to_path_buf(),
        sessions,
        Some("boot.md".to_string()),
    );
    (dir, server)
}

#[test]
fn mcp_initialize_handshake() {
    let (_dir, server) = setup();
    let req = JsonRpcRequest {
        jsonrpc: Some("2.0".to_string()),
        id: Some(json!(1)),
        method: "initialize".to_string(),
        params: None,
    };

    let resp = server.handle_request(req).unwrap();
    assert_eq!(resp.id, json!(1));
    assert!(resp.error.is_none());

    let result = resp.result.unwrap();
    assert_eq!(result["protocolVersion"], MCP_PROTOCOL_VERSION);
    assert_eq!(result["serverInfo"]["name"], "armillary-engine");
}

#[test]
fn mcp_tools_list_returns_armillary_registry() {
    let (_dir, server) = setup();
    let req = JsonRpcRequest {
        jsonrpc: Some("2.0".to_string()),
        id: Some(json!(2)),
        method: "tools/list".to_string(),
        params: None,
    };

    let resp = server.handle_request(req).unwrap();
    let result = resp.result.unwrap();
    let tools = result["tools"].as_array().unwrap();

    let tool_names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();

    assert!(tool_names.contains(&"read_file"));
    assert!(tool_names.contains(&"write_file"));
    assert!(tool_names.contains(&"edit_file"));
    assert!(tool_names.contains(&"search"));
    assert!(tool_names.contains(&"get_composition"));
    assert!(tool_names.contains(&"commit_repo"));
}

#[test]
fn mcp_tools_call_executes_and_logs() {
    let (_dir, server) = setup();
    let req = JsonRpcRequest {
        jsonrpc: Some("2.0".to_string()),
        id: Some(json!(3)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "read_file",
            "arguments": {
                "path": "hello.txt"
            }
        })),
    };

    let resp = server.handle_request(req).unwrap();
    assert!(resp.error.is_none());

    let result = resp.result.unwrap();
    assert_eq!(result["isError"], false);
    let text = result["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Hello, Armillary MCP!"));

    // Verify durable log was written
    let events = server.sessions.store().read_from(&server.stream, 1).unwrap();
    let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();

    assert!(types.contains(&"instance_created"));
    assert!(types.contains(&"tool_use"));
    assert!(types.contains(&"tool_result"));
}

#[test]
fn mcp_prompts_and_resources() {
    let (_dir, server) = setup();
    
    // Prompts
    let prompt_req = JsonRpcRequest {
        jsonrpc: Some("2.0".to_string()),
        id: Some(json!(4)),
        method: "prompts/get".to_string(),
        params: Some(json!({ "name": "armillary_boot" })),
    };
    let prompt_resp = server.handle_request(prompt_req).unwrap();
    let prompt_text = prompt_resp.result.unwrap()["messages"][0]["content"]["text"].as_str().unwrap().to_string();
    assert!(prompt_text.contains("Follow the standard"));
    assert!(prompt_text.contains("Path Anchor (B-4)"));

    // Resources
    let res_req = JsonRpcRequest {
        jsonrpc: Some("2.0".to_string()),
        id: Some(json!(5)),
        method: "resources/read".to_string(),
        params: Some(json!({ "uri": "armillary://workspace/composition" })),
    };
    let res_resp = server.handle_request(res_req).unwrap();
    let res_text = res_resp.result.unwrap()["contents"][0]["text"].as_str().unwrap().to_string();
    assert!(res_text.contains("alpha"));
}

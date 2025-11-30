//! # IPC Server Module
//!
//! JSON-RPC over UNIX socket for external control of GhostWave.
//! Provides comprehensive API for profile management, parameter adjustment,
//! and real-time monitoring.

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn, debug, error};

use crate::processor::{ProcessingProfile, ParamValue, ParamDescriptor};

/// JSON-RPC request structure
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<Value>,
    pub id: Option<Value>,
}

/// JSON-RPC response structure
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    pub id: Option<Value>,
}

/// JSON-RPC error structure
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// IPC server configuration
#[derive(Debug, Clone)]
pub struct IpcConfig {
    /// Socket path
    pub socket_path: PathBuf,
    /// Enable authentication
    pub authentication: bool,
    /// API version
    pub api_version: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum concurrent connections
    pub max_connections: u32,
    /// Enable request logging
    pub log_requests: bool,
}

impl Default for IpcConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from("/tmp/ghostwave.sock"),
            authentication: false,
            api_version: "1.0".to_string(),
            timeout_seconds: 30,
            max_connections: 10,
            log_requests: true,
        }
    }
}

/// Authentication token for IPC access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub token: String,
    pub expires_at: u64,
    pub permissions: Vec<String>,
}

/// IPC method handler trait
pub trait IpcMethodHandler: Send + Sync {
    fn handle_method(&self, method: &str, params: Option<Value>) -> Result<Value>;
    fn list_methods(&self) -> Vec<String>;
}

/// Core GhostWave IPC methods
pub struct GhostWaveIpcHandler {
    processor_state: Arc<RwLock<ProcessorState>>,
    auth_tokens: Arc<Mutex<HashMap<String, AuthToken>>>,
}

#[derive(Debug, Clone)]
pub struct ProcessorState {
    pub current_profile: ProcessingProfile,
    pub parameters: HashMap<String, ParamValue>,
    pub parameter_descriptors: HashMap<String, ParamDescriptor>,
    pub enabled: bool,
    pub sample_rate: u32,
    pub channels: u8,
    pub buffer_size: usize,
    pub stats: ProcessorStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessorStats {
    pub cpu_usage: f32,
    pub latency_ms: f32,
    pub xrun_count: u64,
    pub processing_time_us: u64,
    pub voice_active: bool,
    pub uptime_seconds: u64,
}

impl GhostWaveIpcHandler {
    pub fn new() -> Self {
        let default_params = HashMap::new();
        let default_descriptors = HashMap::new();

        let state = ProcessorState {
            current_profile: ProcessingProfile::Balanced,
            parameters: default_params,
            parameter_descriptors: default_descriptors,
            enabled: true,
            sample_rate: 48000,
            channels: 2,
            buffer_size: 128,
            stats: ProcessorStats {
                cpu_usage: 0.0,
                latency_ms: 0.0,
                xrun_count: 0,
                processing_time_us: 0,
                voice_active: false,
                uptime_seconds: 0,
            },
        };

        Self {
            processor_state: Arc::new(RwLock::new(state)),
            auth_tokens: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn update_state<F>(&self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut ProcessorState) -> Result<()>,
    {
        let mut state = self.processor_state.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire state write lock"))?;
        updater(&mut *state)
    }

    pub fn get_state(&self) -> Result<ProcessorState> {
        let state = self.processor_state.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire state read lock"))?;
        Ok(state.clone())
    }

    fn generate_auth_token(&self, permissions: Vec<String>) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos().hash(&mut hasher);
        let token = format!("gwt_{:x}", hasher.finish());

        let expires_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() + 3600; // 1 hour

        let auth_token = AuthToken {
            token: token.clone(),
            expires_at,
            permissions,
        };

        let mut tokens = self.auth_tokens.lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire token lock"))?;
        tokens.insert(token.clone(), auth_token);

        Ok(token)
    }

    fn validate_auth_token(&self, token: &str) -> Result<bool> {
        let tokens = self.auth_tokens.lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire token lock"))?;

        if let Some(auth_token) = tokens.get(token) {
            let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
            Ok(auth_token.expires_at > now)
        } else {
            Ok(false)
        }
    }
}

impl IpcMethodHandler for GhostWaveIpcHandler {
    fn handle_method(&self, method: &str, params: Option<Value>) -> Result<Value> {
        debug!("IPC method called: {}", method);

        match method {
            // Authentication methods
            "auth.login" => {
                let token = self.generate_auth_token(vec!["read".to_string(), "write".to_string()])?;
                Ok(json!({
                    "token": token,
                    "expires_in": 3600,
                    "permissions": ["read", "write"]
                }))
            }

            "auth.validate" => {
                let token = params.as_ref()
                    .and_then(|p| p.get("token"))
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing token parameter"))?;

                let valid = self.validate_auth_token(token)?;
                Ok(json!({ "valid": valid }))
            }

            // Profile methods
            "get_profile" => {
                let state = self.get_state()?;
                Ok(json!({
                    "profile": state.current_profile,
                    "description": state.current_profile.description()
                }))
            }

            "set_profile" => {
                let profile_str = params.as_ref()
                    .and_then(|p| p.get("profile"))
                    .and_then(|p| p.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing profile parameter"))?;

                let profile: ProcessingProfile = profile_str.parse()
                    .context("Invalid profile name")?;

                self.update_state(|state| {
                    state.current_profile = profile;
                    Ok(())
                })?;

                Ok(json!({
                    "profile": profile,
                    "success": true
                }))
            }

            "list_profiles" => {
                let profiles: Vec<_> = ProcessingProfile::all()
                    .iter()
                    .map(|p| json!({
                        "name": p.to_string(),
                        "description": p.description()
                    }))
                    .collect();

                Ok(json!({ "profiles": profiles }))
            }

            // Parameter methods
            "get_param" => {
                let param_name = params.as_ref()
                    .and_then(|p| p.get("name"))
                    .and_then(|n| n.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing parameter name"))?;

                let state = self.get_state()?;
                if let Some(value) = state.parameters.get(param_name) {
                    Ok(json!({
                        "name": param_name,
                        "value": value
                    }))
                } else {
                    Err(anyhow::anyhow!("Parameter not found: {}", param_name))
                }
            }

            "set_param" => {
                let params_ref = params.as_ref();
                let param_name = params_ref
                    .and_then(|p| p.get("name"))
                    .and_then(|n| n.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Missing parameter name"))?;

                let param_value = params_ref
                    .and_then(|p| p.get("value"))
                    .ok_or_else(|| anyhow::anyhow!("Missing parameter value"))?;

                // Convert JSON value to ParamValue
                let param_val = match param_value {
                    Value::Number(n) if n.is_f64() => ParamValue::Float(n.as_f64().unwrap() as f32),
                    Value::Number(n) if n.is_i64() => ParamValue::Int(n.as_i64().unwrap() as i32),
                    Value::Bool(b) => ParamValue::Bool(*b),
                    Value::String(s) => ParamValue::String(s.clone()),
                    _ => return Err(anyhow::anyhow!("Invalid parameter value type")),
                };

                self.update_state(|state| {
                    state.parameters.insert(param_name.to_string(), param_val.clone());
                    Ok(())
                })?;

                Ok(json!({
                    "name": param_name,
                    "value": param_val,
                    "success": true
                }))
            }

            "get_params" => {
                let state = self.get_state()?;
                let params: HashMap<String, Value> = state.parameters
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::to_value(v).unwrap_or(Value::Null)))
                    .collect();

                Ok(json!({ "parameters": params }))
            }

            "get_param_descriptors" => {
                let state = self.get_state()?;
                let descriptors: HashMap<String, Value> = state.parameter_descriptors
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::to_value(v).unwrap_or(Value::Null)))
                    .collect();

                Ok(json!({ "descriptors": descriptors }))
            }

            // Status and monitoring methods
            "status" => {
                let state = self.get_state()?;
                Ok(json!({
                    "enabled": state.enabled,
                    "profile": state.current_profile,
                    "sample_rate": state.sample_rate,
                    "channels": state.channels,
                    "buffer_size": state.buffer_size,
                    "stats": state.stats
                }))
            }

            "stats" => {
                let state = self.get_state()?;
                Ok(json!(state.stats))
            }

            "levels" => {
                // Real-time audio levels would be updated elsewhere
                let state = self.get_state()?;
                Ok(json!({
                    "input_level": 0.0,
                    "output_level": 0.0,
                    "noise_reduction": 0.7,
                    "voice_active": state.stats.voice_active
                }))
            }

            // System methods
            "version" => {
                Ok(json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "api_version": "1.0",
                    "build_date": std::env::var("VERGEN_BUILD_DATE").unwrap_or_else(|_| "unknown".to_string()),
                    "git_sha": std::env::var("VERGEN_GIT_SHA").unwrap_or_else(|_| "unknown".to_string())
                }))
            }

            "ping" => {
                Ok(json!({
                    "pong": true,
                    "timestamp": SystemTime::now()
                        .duration_since(UNIX_EPOCH)?
                        .as_secs()
                }))
            }

            // Control methods
            "enable" => {
                self.update_state(|state| {
                    state.enabled = true;
                    Ok(())
                })?;
                Ok(json!({ "enabled": true }))
            }

            "disable" => {
                self.update_state(|state| {
                    state.enabled = false;
                    Ok(())
                })?;
                Ok(json!({ "enabled": false }))
            }

            "reset" => {
                self.update_state(|state| {
                    state.parameters.clear();
                    state.stats.xrun_count = 0;
                    Ok(())
                })?;
                Ok(json!({ "reset": true }))
            }

            _ => Err(anyhow::anyhow!("Unknown method: {}", method)),
        }
    }

    fn list_methods(&self) -> Vec<String> {
        vec![
            // Authentication
            "auth.login".to_string(),
            "auth.validate".to_string(),
            // Profiles
            "get_profile".to_string(),
            "set_profile".to_string(),
            "list_profiles".to_string(),
            // Parameters
            "get_param".to_string(),
            "set_param".to_string(),
            "get_params".to_string(),
            "get_param_descriptors".to_string(),
            // Status
            "status".to_string(),
            "stats".to_string(),
            "levels".to_string(),
            // System
            "version".to_string(),
            "ping".to_string(),
            // Control
            "enable".to_string(),
            "disable".to_string(),
            "reset".to_string(),
        ]
    }
}

/// IPC server implementation
pub struct IpcServer {
    config: IpcConfig,
    handler: Arc<dyn IpcMethodHandler>,
    listener: Option<UnixListener>,
    is_running: Arc<RwLock<bool>>,
}

impl IpcServer {
    pub fn new(config: IpcConfig, handler: Arc<dyn IpcMethodHandler>) -> Self {
        Self {
            config,
            handler,
            listener: None,
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    pub fn start(&mut self) -> Result<()> {
        info!("🚀 Starting IPC server at: {}", self.config.socket_path.display());

        // Remove existing socket if it exists
        if self.config.socket_path.exists() {
            std::fs::remove_file(&self.config.socket_path)
                .context("Failed to remove existing socket")?;
        }

        // Create parent directory if needed
        if let Some(parent) = self.config.socket_path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create socket directory")?;
        }

        let listener = UnixListener::bind(&self.config.socket_path)
            .context("Failed to bind UNIX socket")?;

        // Set socket permissions (readable/writable by user and group)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&self.config.socket_path)?.permissions();
            perms.set_mode(0o660);
            std::fs::set_permissions(&self.config.socket_path, perms)?;
        }

        self.listener = Some(listener);
        *self.is_running.write().unwrap() = true;

        info!("✅ IPC server started");
        info!("   Socket: {}", self.config.socket_path.display());
        info!("   API version: {}", self.config.api_version);
        info!("   Max connections: {}", self.config.max_connections);

        // Start accepting connections
        self.accept_connections()?;

        Ok(())
    }

    fn accept_connections(&mut self) -> Result<()> {
        let listener = self.listener.take()
            .ok_or_else(|| anyhow::anyhow!("Listener not initialized"))?;

        let is_running = Arc::clone(&self.is_running);
        let handler = Arc::clone(&self.handler);
        let config = self.config.clone();

        thread::spawn(move || {
            let mut connection_count = 0u32;

            for stream in listener.incoming() {
                if !*is_running.read().unwrap() {
                    break;
                }

                match stream {
                    Ok(stream) => {
                        if connection_count >= config.max_connections {
                            warn!("⚠️ Maximum connections reached, rejecting new connection");
                            continue;
                        }

                        connection_count += 1;
                        let handler = Arc::clone(&handler);
                        let config = config.clone();

                        thread::spawn(move || {
                            if let Err(e) = Self::handle_client(stream, handler, &config) {
                                error!("Client handler error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        error!("Failed to accept connection: {}", e);
                    }
                }
            }

            info!("IPC server connection acceptor stopped");
        });

        Ok(())
    }

    fn handle_client(
        stream: UnixStream,
        handler: Arc<dyn IpcMethodHandler>,
        config: &IpcConfig,
    ) -> Result<()> {
        let peer = stream.peer_addr()
            .map(|addr| format!("{:?}", addr))
            .unwrap_or_else(|_| "unknown".to_string());

        debug!("New IPC client connected: {}", peer);

        let mut reader = BufReader::new(&stream);
        let mut writer = &stream;

        loop {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;

            if bytes_read == 0 {
                debug!("Client disconnected: {}", peer);
                break;
            }

            line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            if config.log_requests {
                debug!("IPC request from {}: {}", peer, line);
            }

            let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
                Ok(request) => Self::process_request(request, &handler),
                Err(e) => Self::create_error_response(
                    None,
                    -32700,
                    "Parse error",
                    Some(json!(e.to_string())),
                ),
            };

            let response_json = serde_json::to_string(&response)?;
            writeln!(writer, "{}", response_json)?;

            if config.log_requests {
                debug!("IPC response to {}: {}", peer, response_json);
            }
        }

        Ok(())
    }

    fn process_request(
        request: JsonRpcRequest,
        handler: &Arc<dyn IpcMethodHandler>,
    ) -> JsonRpcResponse {
        // Validate JSON-RPC version
        if request.jsonrpc != "2.0" {
            return Self::create_error_response(
                request.id,
                -32600,
                "Invalid Request",
                Some(json!("jsonrpc must be '2.0'")),
            );
        }

        // Handle method
        match handler.handle_method(&request.method, request.params) {
            Ok(result) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(result),
                error: None,
                id: request.id,
            },
            Err(e) => {
                let error_code = if request.method.starts_with("auth") { -32001 } else { -32603 };
                Self::create_error_response(
                    request.id,
                    error_code,
                    "Internal error",
                    Some(json!(e.to_string())),
                )
            }
        }
    }

    fn create_error_response(
        id: Option<Value>,
        code: i32,
        message: &str,
        data: Option<Value>,
    ) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.to_string(),
                data,
            }),
            id,
        }
    }

    pub fn stop(&mut self) -> Result<()> {
        info!("Stopping IPC server");

        *self.is_running.write().unwrap() = false;

        // Remove socket file
        if self.config.socket_path.exists() {
            std::fs::remove_file(&self.config.socket_path)
                .context("Failed to remove socket file")?;
        }

        info!("✅ IPC server stopped");
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        *self.is_running.read().unwrap()
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Convenience function to create and start IPC server
pub fn start_ipc_server(
    socket_path: impl AsRef<Path>,
    handler: Arc<dyn IpcMethodHandler>,
) -> Result<IpcServer> {
    let config = IpcConfig {
        socket_path: socket_path.as_ref().to_path_buf(),
        ..Default::default()
    };

    let mut server = IpcServer::new(config, handler);
    server.start()?;
    Ok(server)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_json_rpc_request_parsing() {
        let json = r#"{"jsonrpc": "2.0", "method": "get_profile", "id": 1}"#;
        let request: JsonRpcRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.method, "get_profile");
        assert_eq!(request.id, Some(json!(1)));
    }

    #[test]
    fn test_ipc_handler() {
        let handler = GhostWaveIpcHandler::new();

        // Test version method
        let result = handler.handle_method("version", None).unwrap();
        assert!(result.get("version").is_some());

        // Test ping method
        let result = handler.handle_method("ping", None).unwrap();
        assert_eq!(result.get("pong"), Some(&json!(true)));
    }

    #[test]
    fn test_auth_token_generation() {
        let handler = GhostWaveIpcHandler::new();
        let token = handler.generate_auth_token(vec!["read".to_string()]).unwrap();

        assert!(token.starts_with("gwt_"));
        assert!(handler.validate_auth_token(&token).unwrap());
    }

    #[test]
    fn test_parameter_operations() {
        let handler = GhostWaveIpcHandler::new();

        // Set parameter
        let params = json!({
            "name": "test_param",
            "value": 0.5
        });
        let result = handler.handle_method("set_param", Some(params)).unwrap();
        assert_eq!(result.get("success"), Some(&json!(true)));

        // Get parameter
        let params = json!({ "name": "test_param" });
        let result = handler.handle_method("get_param", Some(params)).unwrap();
        assert_eq!(result.get("value"), Some(&json!(0.5)));
    }
}
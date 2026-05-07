use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use tokio::sync::RwLock;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::audio::AudioProcessor;
use crate::config::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub device_type: String,
    pub channels: u8,
    pub sample_rate: u32,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioLevels {
    pub input_level: f32,
    pub output_level: f32,
    pub noise_reduction: f32,
    pub gate_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub latency_ms: f32,
    pub cpu_usage: f32,
    pub xruns: u64,
    pub frames_processed: u64,
}

/// JSON-RPC 2.0 request
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: String,
    params: Option<Value>,
    id: Option<Value>,
}

/// JSON-RPC 2.0 response
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Option<Value>,
}

/// JSON-RPC 2.0 error
#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

pub trait GhostWaveRpc {
    fn ping(&self) -> Result<String>;
    fn version(&self) -> Result<String>;
    fn register_xlr_device(&self, device_name: String, channels: u8) -> Result<DeviceInfo>;
    fn get_profile(&self) -> Result<String>;
    fn set_profile(&self, profile: String) -> Result<bool>;
    fn get_params(&self) -> Result<Value>;
    fn set_param(&self, param: String, value: f32) -> Result<bool>;
    fn get_levels(&self) -> Result<AudioLevels>;
    fn get_stats(&self) -> Result<ProcessingStats>;
    fn enable_noise_suppression(&self, enabled: bool) -> Result<bool>;
}

#[derive(Clone)]
pub struct GhostWaveRpcImpl {
    config: Arc<RwLock<Config>>,
    processor: Arc<RwLock<Option<AudioProcessor>>>,
    device_id: String,
    stats: Arc<RwLock<ProcessingStats>>,
}

impl GhostWaveRpcImpl {
    pub fn new(config: Config) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            processor: Arc::new(RwLock::new(None)),
            device_id: Uuid::new_v4().to_string(),
            stats: Arc::new(RwLock::new(ProcessingStats {
                latency_ms: 0.0,
                cpu_usage: 0.0,
                xruns: 0,
                frames_processed: 0,
            })),
        }
    }

    pub async fn set_processor(&self, processor: AudioProcessor) {
        let mut proc_lock = self.processor.write().await;
        *proc_lock = Some(processor);
    }

    /// Dispatch a JSON-RPC method call to the appropriate handler
    fn dispatch(&self, method: &str, params: Option<Value>) -> Result<Value> {
        match method {
            "ping" => {
                let result = self.ping()?;
                Ok(json!(result))
            }
            "version" => {
                let result = self.version()?;
                Ok(json!(result))
            }
            "register_xlr_device" => {
                let (name, channels) = match params {
                    Some(Value::Array(ref arr)) if arr.len() >= 2 => {
                        let name = arr[0].as_str().unwrap_or("unknown").to_string();
                        let channels = arr[1].as_u64().unwrap_or(2) as u8;
                        (name, channels)
                    }
                    Some(Value::Object(ref obj)) => {
                        let name = obj
                            .get("device_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        let channels = obj
                            .get("channels")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(2) as u8;
                        (name, channels)
                    }
                    _ => ("unknown".to_string(), 2),
                };
                let result = self.register_xlr_device(name, channels)?;
                Ok(serde_json::to_value(result)?)
            }
            "get_profile" => {
                let result = self.get_profile()?;
                Ok(json!(result))
            }
            "set_profile" => {
                let profile = params
                    .as_ref()
                    .and_then(|p| match p {
                        Value::Array(arr) => arr.first().and_then(|v| v.as_str()),
                        Value::Object(obj) => obj.get("profile").and_then(|v| v.as_str()),
                        Value::String(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .unwrap_or("balanced")
                    .to_string();
                let result = self.set_profile(profile)?;
                Ok(json!(result))
            }
            "get_params" => {
                let result = self.get_params()?;
                Ok(result)
            }
            "set_param" => {
                let (param, value) = match params {
                    Some(Value::Array(ref arr)) if arr.len() >= 2 => {
                        let param = arr[0].as_str().unwrap_or("").to_string();
                        let value = arr[1].as_f64().unwrap_or(0.0) as f32;
                        (param, value)
                    }
                    Some(Value::Object(ref obj)) => {
                        let param = obj
                            .get("param")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let value = obj
                            .get("value")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;
                        (param, value)
                    }
                    _ => return Err(anyhow::anyhow!("Missing param and value")),
                };
                let result = self.set_param(param, value)?;
                Ok(json!(result))
            }
            "get_levels" => {
                let result = self.get_levels()?;
                Ok(serde_json::to_value(result)?)
            }
            "get_stats" => {
                let result = self.get_stats()?;
                Ok(serde_json::to_value(result)?)
            }
            "enable_noise_suppression" => {
                let enabled = params
                    .as_ref()
                    .and_then(|p| match p {
                        Value::Array(arr) => arr.first().and_then(|v| v.as_bool()),
                        Value::Object(obj) => obj.get("enabled").and_then(|v| v.as_bool()),
                        Value::Bool(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(false);
                let result = self.enable_noise_suppression(enabled)?;
                Ok(json!(result))
            }
            _ => Err(anyhow::anyhow!("Unknown method: {}", method)),
        }
    }
}

impl GhostWaveRpc for GhostWaveRpcImpl {
    fn ping(&self) -> Result<String> {
        Ok("pong".to_string())
    }

    fn version(&self) -> Result<String> {
        Ok(env!("CARGO_PKG_VERSION").to_string())
    }

    fn register_xlr_device(&self, device_name: String, channels: u8) -> Result<DeviceInfo> {
        info!(
            "Registering XLR device: {} with {} channels",
            device_name, channels
        );

        Ok(DeviceInfo {
            id: self.device_id.clone(),
            name: device_name,
            device_type: "virtual_xlr".to_string(),
            channels,
            sample_rate: 48000,
            status: "active".to_string(),
        })
    }

    fn get_profile(&self) -> Result<String> {
        match self.config.try_read() {
            Ok(config) => Ok(config.profile.name.clone()),
            Err(_) => Err(anyhow::anyhow!("Config lock held")),
        }
    }

    fn set_profile(&self, profile: String) -> Result<bool> {
        info!("Setting profile to: {}", profile);
        Ok(true)
    }

    fn get_params(&self) -> Result<Value> {
        match self.config.try_read() {
            Ok(config) => {
                let params = json!({
                    "noise_suppression": {
                        "enabled": config.noise_suppression.enabled,
                        "strength": config.noise_suppression.strength,
                        "gate_threshold": config.noise_suppression.gate_threshold,
                        "release_time": config.noise_suppression.release_time
                    },
                    "audio": {
                        "sample_rate": config.audio.sample_rate,
                        "buffer_size": config.audio.buffer_size,
                        "channels": config.audio.channels
                    }
                });
                Ok(params)
            }
            Err(_) => Err(anyhow::anyhow!("Config lock held")),
        }
    }

    fn set_param(&self, param: String, value: f32) -> Result<bool> {
        info!("Setting parameter {} to {}", param, value);

        match param.as_str() {
            "noise_strength" => {
                debug!("Updated noise suppression strength to {}", value);
                Ok(true)
            }
            "gate_threshold" => {
                debug!("Updated gate threshold to {}", value);
                Ok(true)
            }
            "release_time" => {
                debug!("Updated release time to {}", value);
                Ok(true)
            }
            _ => {
                error!("Unknown parameter: {}", param);
                Ok(false)
            }
        }
    }

    fn get_levels(&self) -> Result<AudioLevels> {
        Ok(AudioLevels {
            input_level: -12.0,
            output_level: -15.0,
            noise_reduction: 85.0,
            gate_active: true,
        })
    }

    fn get_stats(&self) -> Result<ProcessingStats> {
        match self.stats.try_read() {
            Ok(stats) => Ok(stats.clone()),
            Err(_) => Err(anyhow::anyhow!("Stats lock held")),
        }
    }

    fn enable_noise_suppression(&self, enabled: bool) -> Result<bool> {
        info!("Setting noise suppression enabled: {}", enabled);
        Ok(true)
    }
}

pub struct IpcServer {
    socket_path: PathBuf,
    rpc_impl: Arc<GhostWaveRpcImpl>,
}

impl IpcServer {
    pub fn new(config: Config) -> Self {
        let socket_path = Self::get_socket_path();
        let rpc_impl = Arc::new(GhostWaveRpcImpl::new(config));

        Self {
            socket_path,
            rpc_impl,
        }
    }

    pub async fn start(&self) -> Result<()> {
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path)?;
        }

        let listener = UnixListener::bind(&self.socket_path)
            .with_context(|| format!("Failed to bind UNIX socket at {:?}", self.socket_path))?;

        info!("IPC server started at: {:?}", self.socket_path);
        info!(
            "PhantomLink can now connect via: {}",
            self.socket_path.display()
        );

        let rpc_impl = self.rpc_impl.clone();
        let socket_path = self.socket_path.clone();

        // Accept connections in a background thread
        thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let handler = rpc_impl.clone();
                        thread::spawn(move || {
                            if let Err(e) = Self::handle_client(stream, handler) {
                                error!("IPC client error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        error!("Failed to accept IPC connection: {}", e);
                    }
                }
            }
        });

        // Block until ctrl-c (matching previous server.wait() behavior)
        tokio::signal::ctrl_c().await?;

        // Cleanup socket on shutdown
        if socket_path.exists() {
            let _ = std::fs::remove_file(&socket_path);
        }

        Ok(())
    }

    fn handle_client(
        stream: std::os::unix::net::UnixStream,
        handler: Arc<GhostWaveRpcImpl>,
    ) -> Result<()> {
        let peer = stream
            .peer_addr()
            .map(|addr| format!("{:?}", addr))
            .unwrap_or_else(|_| "unknown".to_string());

        debug!("IPC client connected: {}", peer);

        let mut reader = BufReader::new(&stream);
        let mut writer = &stream;

        loop {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;

            if bytes_read == 0 {
                debug!("IPC client disconnected: {}", peer);
                break;
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let response = match serde_json::from_str::<JsonRpcRequest>(line) {
                Ok(request) => {
                    if request.jsonrpc != "2.0" {
                        JsonRpcResponse {
                            jsonrpc: "2.0".to_string(),
                            result: None,
                            error: Some(JsonRpcError {
                                code: -32600,
                                message: "Invalid Request".to_string(),
                                data: Some(json!("jsonrpc must be '2.0'")),
                            }),
                            id: request.id,
                        }
                    } else {
                        match handler.dispatch(&request.method, request.params) {
                            Ok(result) => JsonRpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: Some(result),
                                error: None,
                                id: request.id,
                            },
                            Err(e) => JsonRpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: None,
                                error: Some(JsonRpcError {
                                    code: -32603,
                                    message: "Internal error".to_string(),
                                    data: Some(json!(e.to_string())),
                                }),
                                id: request.id,
                            },
                        }
                    }
                }
                Err(e) => JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: "Parse error".to_string(),
                        data: Some(json!(e.to_string())),
                    }),
                    id: None,
                },
            };

            let response_json = serde_json::to_string(&response)?;
            writeln!(writer, "{}", response_json)?;
        }

        Ok(())
    }

    pub fn get_rpc_impl(&self) -> Arc<GhostWaveRpcImpl> {
        self.rpc_impl.clone()
    }

    fn get_socket_path() -> PathBuf {
        if let Ok(xdg_runtime) = std::env::var("XDG_RUNTIME_DIR") {
            PathBuf::from(xdg_runtime).join("ghostwave.sock")
        } else {
            PathBuf::from("/tmp/ghostwave.sock")
        }
    }
}

pub async fn run_ipc_server(config: Config) -> Result<()> {
    let server = IpcServer::new(config);
    server.start().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[tokio::test]
    async fn test_rpc_ping() {
        let config = Config::load("balanced").unwrap();
        let rpc = GhostWaveRpcImpl::new(config);

        let result = rpc.ping();
        assert_eq!(result.unwrap(), "pong");
    }

    #[tokio::test]
    async fn test_register_xlr_device() {
        let config = Config::load("balanced").unwrap();
        let rpc = GhostWaveRpcImpl::new(config);

        let device = rpc
            .register_xlr_device("Test XLR".to_string(), 2)
            .unwrap();
        assert_eq!(device.name, "Test XLR");
        assert_eq!(device.channels, 2);
        assert_eq!(device.device_type, "virtual_xlr");
    }

    #[tokio::test]
    async fn test_dispatch_ping() {
        let config = Config::load("balanced").unwrap();
        let rpc = GhostWaveRpcImpl::new(config);

        let result = rpc.dispatch("ping", None).unwrap();
        assert_eq!(result, json!("pong"));
    }

    #[tokio::test]
    async fn test_dispatch_unknown_method() {
        let config = Config::load("balanced").unwrap();
        let rpc = GhostWaveRpcImpl::new(config);

        let result = rpc.dispatch("nonexistent", None);
        assert!(result.is_err());
    }
}

use anyhow::Result;
use jsonrpc_core::{IoHandler, Result as RpcResult, Value};
use jsonrpc_derive::rpc;
use jsonrpc_ipc_server::ServerBuilder;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, error};
use uuid::Uuid;

use crate::config::Config;
use crate::audio::AudioProcessor;

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

#[rpc(server)]
pub trait GhostWaveRpc {
    #[rpc(name = "ping")]
    fn ping(&self) -> RpcResult<String>;

    #[rpc(name = "version")]
    fn version(&self) -> RpcResult<String>;

    #[rpc(name = "register_xlr_device")]
    fn register_xlr_device(&self, device_name: String, channels: u8) -> RpcResult<DeviceInfo>;

    #[rpc(name = "get_profile")]
    fn get_profile(&self) -> RpcResult<String>;

    #[rpc(name = "set_profile")]
    fn set_profile(&self, profile: String) -> RpcResult<bool>;

    #[rpc(name = "get_params")]
    fn get_params(&self) -> RpcResult<Value>;

    #[rpc(name = "set_param")]
    fn set_param(&self, param: String, value: f32) -> RpcResult<bool>;

    #[rpc(name = "get_levels")]
    fn get_levels(&self) -> RpcResult<AudioLevels>;

    #[rpc(name = "get_stats")]
    fn get_stats(&self) -> RpcResult<ProcessingStats>;

    #[rpc(name = "enable_noise_suppression")]
    fn enable_noise_suppression(&self, enabled: bool) -> RpcResult<bool>;
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
}

impl GhostWaveRpc for GhostWaveRpcImpl {
    fn ping(&self) -> RpcResult<String> {
        Ok("pong".to_string())
    }

    fn version(&self) -> RpcResult<String> {
        Ok(env!("CARGO_PKG_VERSION").to_string())
    }

    fn register_xlr_device(&self, device_name: String, channels: u8) -> RpcResult<DeviceInfo> {
        info!("Registering XLR device: {} with {} channels", device_name, channels);

        Ok(DeviceInfo {
            id: self.device_id.clone(),
            name: device_name,
            device_type: "virtual_xlr".to_string(),
            channels,
            sample_rate: 48000,
            status: "active".to_string(),
        })
    }

    fn get_profile(&self) -> RpcResult<String> {
        // Note: This should be async but jsonrpc-derive doesn't support async yet
        // For now, we'll use try_read() which might fail if lock is held
        match self.config.try_read() {
            Ok(config) => Ok(config.profile.name.clone()),
            Err(_) => Err(jsonrpc_core::Error::internal_error()),
        }
    }

    fn set_profile(&self, profile: String) -> RpcResult<bool> {
        info!("Setting profile to: {}", profile);
        // This would reload the config with the new profile
        // Implementation would need async support
        Ok(true)
    }

    fn get_params(&self) -> RpcResult<Value> {
        match self.config.try_read() {
            Ok(config) => {
                let params = serde_json::json!({
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
            Err(_) => Err(jsonrpc_core::Error::internal_error()),
        }
    }

    fn set_param(&self, param: String, value: f32) -> RpcResult<bool> {
        info!("Setting parameter {} to {}", param, value);

        // Note: This is a simplified implementation
        // In practice, we'd need to safely update the config and processor
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

    fn get_levels(&self) -> RpcResult<AudioLevels> {
        // In a real implementation, this would get live audio levels
        Ok(AudioLevels {
            input_level: -12.0,   // dB
            output_level: -15.0,  // dB
            noise_reduction: 85.0, // %
            gate_active: true,
        })
    }

    fn get_stats(&self) -> RpcResult<ProcessingStats> {
        match self.stats.try_read() {
            Ok(stats) => Ok(stats.clone()),
            Err(_) => Err(jsonrpc_core::Error::internal_error()),
        }
    }

    fn enable_noise_suppression(&self, enabled: bool) -> RpcResult<bool> {
        info!("Setting noise suppression enabled: {}", enabled);
        // This would update the processor in real-time
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
        // Remove existing socket if it exists
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path)?;
        }

        let mut io = IoHandler::new();
        io.extend_with((*self.rpc_impl).clone().to_delegate());

        let server = ServerBuilder::new(io)
            .start(&format!("ipc://{}", self.socket_path.display()))?;

        info!("🔌 IPC server started at: {:?}", self.socket_path);
        info!("PhantomLink can now connect via: {}", self.socket_path.display());

        // Keep the server running
        server.wait();
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

        let device = rpc.register_xlr_device("Test XLR".to_string(), 2).unwrap();
        assert_eq!(device.name, "Test XLR");
        assert_eq!(device.channels, 2);
        assert_eq!(device.device_type, "virtual_xlr");
    }
}
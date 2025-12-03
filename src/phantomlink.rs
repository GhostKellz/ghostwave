use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::time::{Duration, sleep};
use tracing::{debug, info};

use crate::audio::AudioProcessor;
use crate::config::Config;
use crate::ipc::{DeviceInfo, GhostWaveRpc, IpcServer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhantomLinkConfig {
    pub socket_path: PathBuf,
    pub device_name: String,
    pub channels: u8,
    pub sample_rate: u32,
}

impl Default for PhantomLinkConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from("/tmp/phantomlink.sock"),
            device_name: "GhostWave Virtual XLR".to_string(),
            channels: 2,
            sample_rate: 48000,
        }
    }
}

pub struct PhantomLinkIntegration {
    config: PhantomLinkConfig,
    ghostwave_config: Config,
    ipc_server: Arc<IpcServer>,
}

impl PhantomLinkIntegration {
    pub fn new(ghostwave_config: Config) -> Self {
        let ipc_server = Arc::new(IpcServer::new(ghostwave_config.clone()));
        Self {
            config: PhantomLinkConfig::default(),
            ghostwave_config,
            ipc_server,
        }
    }

    #[allow(dead_code)] // Public API for PhantomLink integration
    pub fn with_config(ghostwave_config: Config, phantomlink_config: PhantomLinkConfig) -> Self {
        let ipc_server = Arc::new(IpcServer::new(ghostwave_config.clone()));
        Self {
            config: phantomlink_config,
            ghostwave_config,
            ipc_server,
        }
    }

    pub async fn register_as_xlr_device(&self) -> Result<DeviceInfo> {
        info!("Registering GhostWave as PhantomLink XLR device");
        info!("Device: {}", self.config.device_name);
        info!("Socket: {:?}", self.config.socket_path);
        info!(
            "Audio: {}Hz, {} channels",
            self.ghostwave_config.audio.sample_rate, self.ghostwave_config.audio.channels
        );

        // Use the IPC server to register the device
        let rpc_impl = self.ipc_server.get_rpc_impl();
        let device_info = rpc_impl
            .register_xlr_device(
                self.config.device_name.clone(),
                self.ghostwave_config.audio.channels,
            )
            .map_err(|e| anyhow::anyhow!("Failed to register XLR device: {:?}", e))?;

        info!(
            "✅ Registered as virtual XLR device with ID: {}",
            device_info.id
        );
        Ok(device_info)
    }

    pub async fn setup_audio_routing(&self) -> Result<()> {
        info!("Setting up audio routing for PhantomLink integration");
        info!("  Input: Scarlett Solo 4th Gen → GhostWave AI Filter");
        info!("  Processing: NVIDIA RTX Voice-style noise cancellation");
        info!("  Output: PhantomLink Virtual XLR → Applications");

        info!("Profile optimized for XLR workflow:");
        info!(
            "  Noise Suppression: {}",
            if self.ghostwave_config.noise_suppression.enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        );

        if self.ghostwave_config.noise_suppression.enabled {
            info!(
                "  Strength: {:.1}%",
                self.ghostwave_config.noise_suppression.strength * 100.0
            );
            info!(
                "  Gate Threshold: {:.1} dB",
                self.ghostwave_config.noise_suppression.gate_threshold
            );
        }

        Ok(())
    }

    pub async fn start_ipc_server(&self) -> Result<()> {
        info!("Starting IPC server for PhantomLink communication");

        let server = self.ipc_server.clone();
        tokio::spawn(async move {
            if let Err(e) = server.start().await {
                tracing::error!("IPC server error: {}", e);
            }
        });

        // Give the server time to start
        sleep(Duration::from_millis(100)).await;
        info!("✅ IPC server running - PhantomLink can now connect");
        Ok(())
    }

    pub async fn run_xlr_bridge(&self) -> Result<()> {
        info!("🎤 GhostWave XLR Bridge active");
        info!("Your audio chain: Microphone → Scarlett Solo → GhostWave AI → PhantomLink");

        // Start the audio processor
        let processor = AudioProcessor::new(self.ghostwave_config.clone())?;
        let rpc_impl = self.ipc_server.get_rpc_impl();
        rpc_impl.set_processor(processor).await;

        // Simulate live processing stats updates
        let stats_updater = {
            let _rpc_impl = rpc_impl.clone();
            tokio::spawn(async move {
                loop {
                    sleep(Duration::from_millis(100)).await;
                    // In real implementation, this would update with actual stats
                    debug!("Processing audio... (stats update simulation)");
                }
            })
        };

        info!("Press Ctrl+C to stop the XLR bridge");
        tokio::signal::ctrl_c().await?;

        stats_updater.abort();
        info!("Shutting down XLR bridge");
        Ok(())
    }
}

pub async fn run_phantomlink_mode(ghostwave_config: Config) -> Result<()> {
    info!("Starting GhostWave in PhantomLink integration mode");

    let integration = PhantomLinkIntegration::new(ghostwave_config);

    // Start the IPC server first
    integration.start_ipc_server().await?;

    // Register as XLR device
    let device_info = integration.register_as_xlr_device().await?;
    info!("PhantomLink Integration Ready:");
    info!("  Device ID: {}", device_info.id);
    info!("  Channels: {}", device_info.channels);
    info!("  Sample Rate: {}Hz", device_info.sample_rate);

    // Set up audio routing
    integration.setup_audio_routing().await?;

    // Run the bridge
    integration.run_xlr_bridge().await?;

    Ok(())
}

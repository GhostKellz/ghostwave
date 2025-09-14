use anyhow::{Result, Context};
use std::fs;
use std::path::PathBuf;
use tracing::info;

pub struct PipeWireAutoload;

impl PipeWireAutoload {
    pub fn install() -> Result<()> {
        info!("Installing PipeWire autoload configuration");

        let config_dir = Self::pipewire_config_dir()?;
        fs::create_dir_all(&config_dir)
            .context("Failed to create PipeWire config directory")?;

        // Create module configuration
        let module_config = Self::generate_module_config();
        let module_path = config_dir.join("modules.conf.d").join("90-ghostwave.conf");

        if let Some(parent) = module_path.parent() {
            fs::create_dir_all(parent)
                .context("Failed to create modules.conf.d directory")?;
        }

        fs::write(&module_path, &module_config)
            .context("Failed to write module configuration")?;

        info!("Created PipeWire module config: {:?}", module_path);

        // Create virtual device configuration
        let device_config = Self::generate_device_config();
        let device_path = config_dir.join("devices.conf.d").join("90-ghostwave.conf");

        if let Some(parent) = device_path.parent() {
            fs::create_dir_all(parent)
                .context("Failed to create devices.conf.d directory")?;
        }

        fs::write(&device_path, &device_config)
            .context("Failed to write device configuration")?;

        info!("Created PipeWire device config: {:?}", device_path);
        info!("✅ PipeWire autoload installed - restart PipeWire to activate");

        Ok(())
    }

    pub fn uninstall() -> Result<()> {
        info!("Removing PipeWire autoload configuration");

        let config_dir = Self::pipewire_config_dir()?;

        let module_path = config_dir.join("modules.conf.d").join("90-ghostwave.conf");
        if module_path.exists() {
            fs::remove_file(&module_path)
                .context("Failed to remove module configuration")?;
            info!("Removed: {:?}", module_path);
        }

        let device_path = config_dir.join("devices.conf.d").join("90-ghostwave.conf");
        if device_path.exists() {
            fs::remove_file(&device_path)
                .context("Failed to remove device configuration")?;
            info!("Removed: {:?}", device_path);
        }

        info!("✅ PipeWire autoload uninstalled");
        Ok(())
    }

    fn pipewire_config_dir() -> Result<PathBuf> {
        let mut config_dir = dirs::config_dir()
            .context("Failed to get config directory")?;
        config_dir.push("pipewire");
        Ok(config_dir)
    }

    fn generate_module_config() -> String {
        r#"# GhostWave - NVIDIA RTX Voice for Linux
# Auto-loads the GhostWave noise suppression module

context.modules = [
    {
        name = libpipewire-module-filter-chain
        args = {
            node.description = "GhostWave AI Noise Suppression"
            media.name = "GhostWave Filter"
            filter.graph = {
                nodes = [
                    {
                        type = builtin
                        name = ghostwave_filter
                        label = GhostWave
                    }
                ]
            }
            capture.props = {
                node.name = "ghostwave_input"
                node.description = "GhostWave Input (Raw Microphone)"
            }
            playback.props = {
                node.name = "ghostwave_output"
                node.description = "GhostWave Output (Clean Microphone)"
                media.class = "Audio/Source"
            }
        }
    }
]
"#.to_string()
    }

    fn generate_device_config() -> String {
        r#"# GhostWave Virtual Devices
# Creates clean microphone source automatically

context.objects = [
    {
        factory = adapter
        args = {
            factory.name = support.null-audio-sink
            node.name = "ghostwave_clean_mic"
            node.description = "GhostWave Clean Microphone"
            media.class = "Audio/Source"
            audio.channels = 2
            audio.rate = 48000
            monitor = true
        }
    }
]
"#.to_string()
    }
}

pub fn setup_pipewire_autoload() -> Result<()> {
    info!("Setting up PipeWire zero-click autoload");
    PipeWireAutoload::install()
}

pub fn remove_pipewire_autoload() -> Result<()> {
    info!("Removing PipeWire autoload configuration");
    PipeWireAutoload::uninstall()
}
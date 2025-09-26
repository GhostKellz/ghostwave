//! # GhostWave Configuration System
//!
//! Comprehensive configuration management with TOML support, hot reloading,
//! and environment variable overrides.

use anyhow::{Result, Context};
use notify::{Watcher, RecommendedWatcher, RecursiveMode, Event, EventKind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::sync::mpsc;
use tracing::{info, warn, debug, error};

/// Main configuration structure for GhostWave
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GhostWaveConfig {
    /// Application metadata
    pub app: AppConfig,
    /// Audio system configuration
    pub audio: AudioConfig,
    /// DSP processing configuration
    pub processing: ProcessingConfig,
    /// Noise suppression settings
    pub noise_suppression: NoiseSuppressionConfig,
    /// IPC server configuration
    pub ipc: IpcConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Integration settings
    pub integrations: IntegrationsConfig,
    /// Performance tuning
    pub performance: PerformanceConfig,
}

/// Application-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Configuration schema version
    pub version: String,
    /// Active profile name
    pub active_profile: String,
    /// Whether to auto-save settings changes
    pub auto_save: bool,
    /// Configuration file watch for hot reloading
    pub hot_reload: bool,
}

/// Audio system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Audio sample rate in Hz
    pub sample_rate: u32,
    /// Buffer size in frames
    pub buffer_size: u32,
    /// Number of audio channels (1=mono, 2=stereo)
    pub channels: u8,
    /// Input device name (None for default)
    pub input_device: Option<String>,
    /// Output device name (None for default)
    pub output_device: Option<String>,
    /// Audio backend preference
    pub backend: AudioBackend,
    /// Enable automatic device selection
    pub auto_device_selection: bool,
    /// Hotplug detection settings
    pub hotplug: HotplugConfig,
}

/// DSP processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Enable DSP processing pipeline
    pub enabled: bool,
    /// High-pass filter frequency in Hz
    pub highpass_frequency: f32,
    /// Voice activity detection settings
    pub vad: VadConfig,
    /// Expander/gate settings
    pub gate: GateConfig,
    /// Soft limiter settings
    pub limiter: LimiterConfig,
    /// Voice enhancement settings
    pub voice_enhancement: VoiceEnhancementConfig,
}

/// Voice Activity Detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    /// Enable VAD processing
    pub enabled: bool,
    /// VAD sensitivity (0.0 = very sensitive, 1.0 = less sensitive)
    pub sensitivity: f32,
    /// Hangover time in seconds after voice stops
    pub hangover_time: f32,
    /// Energy threshold in dB
    pub energy_threshold: f32,
}

/// Gate/Expander configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateConfig {
    /// Enable gate processing
    pub enabled: bool,
    /// Gate threshold in dB
    pub threshold: f32,
    /// Expansion ratio (1.0 = no expansion, higher = more expansion)
    pub ratio: f32,
    /// Attack time in seconds
    pub attack_time: f32,
    /// Release time in seconds
    pub release_time: f32,
}

/// Soft limiter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimiterConfig {
    /// Enable limiter
    pub enabled: bool,
    /// Limiter threshold (0.0 to 1.0)
    pub threshold: f32,
    /// Knee width for soft limiting
    pub knee_width: f32,
    /// Makeup gain in dB
    pub makeup_gain: f32,
    /// Lookahead time in milliseconds
    pub lookahead_ms: f32,
}

/// Voice enhancement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceEnhancementConfig {
    /// Enable voice enhancement
    pub enabled: bool,
    /// Enhancement strength (0.0 to 1.0)
    pub strength: f32,
    /// Frequency range for enhancement
    pub frequency_range: (f32, f32),
    /// Clarity boost amount in dB
    pub clarity_boost: f32,
}

/// Noise suppression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseSuppressionConfig {
    /// Enable noise suppression
    pub enabled: bool,
    /// Noise reduction strength (0.0 to 1.0)
    pub strength: f32,
    /// Spectral subtraction over-subtraction factor
    pub over_subtraction: f32,
    /// Enable adaptive threshold
    pub adaptive_threshold: bool,
    /// Minimum noise floor in dB
    pub noise_floor_db: f32,
}

/// Audio backend options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioBackend {
    #[serde(rename = "pipewire")]
    PipeWire,
    #[serde(rename = "alsa")]
    Alsa,
    #[serde(rename = "jack")]
    Jack,
    #[serde(rename = "cpal")]
    Cpal,
    #[serde(rename = "auto")]
    Auto,
}

/// Hotplug detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotplugConfig {
    /// Enable hotplug detection
    pub enabled: bool,
    /// Debounce time in milliseconds
    pub debounce_ms: u64,
    /// Automatically switch to new devices
    pub auto_switch: bool,
    /// Priority order for device selection
    pub device_priority: Vec<String>,
}

/// IPC server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcConfig {
    /// Enable IPC server
    pub enabled: bool,
    /// Unix socket path
    pub socket_path: String,
    /// Enable authentication
    pub authentication: bool,
    /// API version
    pub api_version: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Maximum concurrent connections
    pub max_connections: u32,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    /// Enable file logging
    pub file_enabled: bool,
    /// Log file path
    pub file_path: Option<PathBuf>,
    /// Maximum log file size in MB
    pub max_file_size_mb: u64,
    /// Number of log files to rotate
    pub rotation_count: u32,
    /// Enable structured JSON logging
    pub json_format: bool,
    /// Enable performance logging
    pub performance_metrics: bool,
}

/// Log level options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    #[serde(rename = "error")]
    Error,
    #[serde(rename = "warn")]
    Warn,
    #[serde(rename = "info")]
    Info,
    #[serde(rename = "debug")]
    Debug,
    #[serde(rename = "trace")]
    Trace,
}

/// Integration configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationsConfig {
    /// PhantomLink integration settings
    pub phantomlink: PhantomLinkConfig,
    /// NVControl integration settings
    pub nvcontrol: NVControlConfig,
    /// Generic plugin configurations
    pub plugins: HashMap<String, PluginConfig>,
}

/// PhantomLink integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhantomLinkConfig {
    /// Enable PhantomLink integration
    pub enabled: bool,
    /// Integration mode
    pub mode: IntegrationMode,
    /// Control socket path
    pub control_socket: String,
    /// Shared buffer size
    pub buffer_size: usize,
}

/// NVControl integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NVControlConfig {
    /// Enable NVControl integration
    pub enabled: bool,
    /// GPU device ID
    pub gpu_device: Option<u32>,
    /// Enable RTX Voice acceleration
    pub rtx_voice: bool,
    /// RTX quality setting
    pub rtx_quality: RtxQuality,
    /// GPU memory limit in MB
    pub memory_limit_mb: u32,
    /// Power management mode
    pub power_mode: PowerMode,
}

/// Integration mode options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrationMode {
    #[serde(rename = "ipc")]
    Ipc,
    #[serde(rename = "embedded")]
    Embedded,
    #[serde(rename = "jack")]
    Jack,
}

/// RTX quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RtxQuality {
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
    #[serde(rename = "maximum")]
    Maximum,
}

/// Power management modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerMode {
    #[serde(rename = "balanced")]
    Balanced,
    #[serde(rename = "performance")]
    Performance,
    #[serde(rename = "efficiency")]
    Efficiency,
}

/// Generic plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin enabled state
    pub enabled: bool,
    /// Plugin-specific parameters
    pub parameters: HashMap<String, toml::Value>,
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable real-time priority
    pub realtime_priority: bool,
    /// CPU affinity mask (None = no affinity)
    pub cpu_affinity: Option<Vec<usize>>,
    /// Memory pool size
    pub memory_pool_size: usize,
    /// Enable NUMA optimizations
    pub numa_aware: bool,
    /// Target latency in milliseconds
    pub target_latency_ms: f32,
    /// Enable performance monitoring
    pub monitoring: bool,
}

/// Configuration manager with hot reloading support
pub struct ConfigManager {
    config: Arc<RwLock<GhostWaveConfig>>,
    config_path: PathBuf,
    watcher: Option<RecommendedWatcher>,
    reload_sender: Option<mpsc::Sender<()>>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Result<Self> {
        let config_path = Self::default_config_path()?;
        let config = if config_path.exists() {
            Self::load_from_file(&config_path)?
        } else {
            Self::create_default_config(&config_path)?
        };

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            config_path,
            watcher: None,
            reload_sender: None,
        })
    }

    /// Load configuration for a specific profile
    pub fn load_profile(profile_name: &str) -> Result<GhostWaveConfig> {
        let config_path = Self::profile_config_path(profile_name)?;

        if config_path.exists() {
            Self::load_from_file(&config_path)
        } else {
            // Create default profile configuration
            let config = Self::default_profile_config(profile_name);
            Self::save_to_file(&config, &config_path)?;
            Ok(config)
        }
    }

    /// Get current configuration (read-only)
    pub fn get_config(&self) -> Result<GhostWaveConfig> {
        let config = self.config.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire config read lock"))?;
        Ok(config.clone())
    }

    /// Update configuration
    pub fn update_config<F>(&self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut GhostWaveConfig) -> Result<()>,
    {
        let mut config = self.config.write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire config write lock"))?;

        updater(&mut *config)?;

        // Auto-save if enabled
        if config.app.auto_save {
            self.save_config()?;
        }

        Ok(())
    }

    /// Save current configuration to file
    pub fn save_config(&self) -> Result<()> {
        let config = self.config.read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire config read lock"))?;

        Self::save_to_file(&*config, &self.config_path)
    }

    /// Enable hot reloading
    pub fn enable_hot_reload(&mut self) -> Result<mpsc::Receiver<()>> {
        let (tx, rx) = mpsc::channel();

        let config_path = self.config_path.clone();
        let config_arc = Arc::clone(&self.config);

        let mut watcher = notify::recommended_watcher(move |result: Result<Event, notify::Error>| {
            match result {
                Ok(event) => {
                    if matches!(event.kind, EventKind::Modify(_)) {
                        debug!("Configuration file changed, reloading...");

                        match Self::load_from_file(&config_path) {
                            Ok(new_config) => {
                                if let Ok(mut config) = config_arc.write() {
                                    *config = new_config;
                                    info!("✅ Configuration reloaded successfully");
                                    let _ = tx.send(());
                                }
                            }
                            Err(e) => {
                                error!("❌ Failed to reload configuration: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("File watcher error: {}", e);
                }
            }
        })?;

        watcher.watch(&self.config_path, RecursiveMode::NonRecursive)?;

        self.watcher = Some(watcher);
        self.reload_sender = Some(tx);

        info!("🔄 Hot reload enabled for configuration file");
        Ok(rx)
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&self) -> Result<()> {
        self.update_config(|config| {
            // Audio configuration overrides
            if let Ok(sample_rate) = std::env::var("GHOSTWAVE_SAMPLE_RATE") {
                if let Ok(rate) = sample_rate.parse::<u32>() {
                    config.audio.sample_rate = rate;
                    debug!("Applied env override: SAMPLE_RATE={}", rate);
                }
            }

            if let Ok(buffer_size) = std::env::var("GHOSTWAVE_BUFFER_SIZE") {
                if let Ok(size) = buffer_size.parse::<u32>() {
                    config.audio.buffer_size = size;
                    debug!("Applied env override: BUFFER_SIZE={}", size);
                }
            }

            if let Ok(channels) = std::env::var("GHOSTWAVE_CHANNELS") {
                if let Ok(ch) = channels.parse::<u8>() {
                    config.audio.channels = ch;
                    debug!("Applied env override: CHANNELS={}", ch);
                }
            }

            // Processing configuration overrides
            if let Ok(noise_strength) = std::env::var("GHOSTWAVE_NOISE_STRENGTH") {
                if let Ok(strength) = noise_strength.parse::<f32>() {
                    config.noise_suppression.strength = strength;
                    debug!("Applied env override: NOISE_STRENGTH={}", strength);
                }
            }

            // Log level override
            if let Ok(log_level) = std::env::var("GHOSTWAVE_LOG_LEVEL") {
                match log_level.to_lowercase().as_str() {
                    "error" => config.logging.level = LogLevel::Error,
                    "warn" => config.logging.level = LogLevel::Warn,
                    "info" => config.logging.level = LogLevel::Info,
                    "debug" => config.logging.level = LogLevel::Debug,
                    "trace" => config.logging.level = LogLevel::Trace,
                    _ => warn!("Unknown log level: {}", log_level),
                }
            }

            Ok(())
        })
    }

    /// Get default configuration file path
    fn default_config_path() -> Result<PathBuf> {
        let mut path = dirs::config_dir()
            .context("Failed to get user config directory")?;
        path.push("ghostwave");

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&path)
            .context("Failed to create config directory")?;

        path.push("config.toml");
        Ok(path)
    }

    /// Get profile-specific configuration path
    fn profile_config_path(profile_name: &str) -> Result<PathBuf> {
        let mut path = dirs::config_dir()
            .context("Failed to get user config directory")?;
        path.push("ghostwave");
        path.push("profiles");

        std::fs::create_dir_all(&path)
            .context("Failed to create profiles directory")?;

        path.push(format!("{}.toml", profile_name));
        Ok(path)
    }

    /// Load configuration from TOML file
    fn load_from_file(path: &Path) -> Result<GhostWaveConfig> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        toml::from_str(&content)
            .with_context(|| format!("Failed to parse TOML config: {}", path.display()))
    }

    /// Save configuration to TOML file
    fn save_to_file(config: &GhostWaveConfig, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(config)
            .context("Failed to serialize config to TOML")?;

        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;

        debug!("Configuration saved to: {}", path.display());
        Ok(())
    }

    /// Create default configuration and save it
    fn create_default_config(path: &Path) -> Result<GhostWaveConfig> {
        let config = Self::default_config();
        Self::save_to_file(&config, path)?;
        info!("Created default configuration at: {}", path.display());
        Ok(config)
    }

    /// Generate default configuration
    fn default_config() -> GhostWaveConfig {
        GhostWaveConfig {
            app: AppConfig {
                version: "1.0.0".to_string(),
                active_profile: "balanced".to_string(),
                auto_save: true,
                hot_reload: true,
            },
            audio: AudioConfig {
                sample_rate: 48000,
                buffer_size: 128,
                channels: 2,
                input_device: None,
                output_device: None,
                backend: AudioBackend::Auto,
                auto_device_selection: true,
                hotplug: HotplugConfig {
                    enabled: true,
                    debounce_ms: 1000,
                    auto_switch: false,
                    device_priority: vec![],
                },
            },
            processing: ProcessingConfig {
                enabled: true,
                highpass_frequency: 80.0,
                vad: VadConfig {
                    enabled: true,
                    sensitivity: 0.5,
                    hangover_time: 0.2,
                    energy_threshold: -40.0,
                },
                gate: GateConfig {
                    enabled: true,
                    threshold: -45.0,
                    ratio: 3.0,
                    attack_time: 0.001,
                    release_time: 0.1,
                },
                limiter: LimiterConfig {
                    enabled: true,
                    threshold: 0.95,
                    knee_width: 0.1,
                    makeup_gain: 0.0,
                    lookahead_ms: 5.0,
                },
                voice_enhancement: VoiceEnhancementConfig {
                    enabled: false,
                    strength: 0.5,
                    frequency_range: (200.0, 4000.0),
                    clarity_boost: 2.0,
                },
            },
            noise_suppression: NoiseSuppressionConfig {
                enabled: true,
                strength: 0.7,
                over_subtraction: 2.0,
                adaptive_threshold: true,
                noise_floor_db: -60.0,
            },
            ipc: IpcConfig {
                enabled: false,
                socket_path: "/tmp/ghostwave.sock".to_string(),
                authentication: false,
                api_version: "1.0".to_string(),
                timeout_seconds: 30,
                max_connections: 10,
            },
            logging: LoggingConfig {
                level: LogLevel::Info,
                file_enabled: false,
                file_path: None,
                max_file_size_mb: 10,
                rotation_count: 5,
                json_format: false,
                performance_metrics: false,
            },
            integrations: IntegrationsConfig {
                phantomlink: PhantomLinkConfig {
                    enabled: false,
                    mode: IntegrationMode::Ipc,
                    control_socket: "/tmp/phantomlink.sock".to_string(),
                    buffer_size: 4096,
                },
                nvcontrol: NVControlConfig {
                    enabled: false,
                    gpu_device: None,
                    rtx_voice: false,
                    rtx_quality: RtxQuality::Medium,
                    memory_limit_mb: 1024,
                    power_mode: PowerMode::Balanced,
                },
                plugins: HashMap::new(),
            },
            performance: PerformanceConfig {
                realtime_priority: true,
                cpu_affinity: None,
                memory_pool_size: 16,
                numa_aware: false,
                target_latency_ms: 15.0,
                monitoring: true,
            },
        }
    }

    /// Generate profile-specific default configuration
    fn default_profile_config(profile_name: &str) -> GhostWaveConfig {
        let mut config = Self::default_config();
        config.app.active_profile = profile_name.to_string();

        match profile_name {
            "streaming" => {
                config.audio.buffer_size = 128;
                config.noise_suppression.strength = 0.85;
                config.processing.gate.threshold = -40.0;
                config.performance.target_latency_ms = 10.0;
            }
            "studio" => {
                config.audio.sample_rate = 96000;
                config.audio.buffer_size = 256;
                config.noise_suppression.strength = 0.3;
                config.processing.gate.threshold = -60.0;
                config.performance.target_latency_ms = 5.0;
            }
            _ => {
                // Balanced defaults are already set
            }
        }

        config
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default ConfigManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_config_serialization() {
        let config = ConfigManager::default_config();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: GhostWaveConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(config.audio.sample_rate, parsed.audio.sample_rate);
        assert_eq!(config.noise_suppression.strength, parsed.noise_suppression.strength);
    }

    #[test]
    fn test_config_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");

        let config = ConfigManager::default_config();
        ConfigManager::save_to_file(&config, &config_path).unwrap();

        let loaded_config = ConfigManager::load_from_file(&config_path).unwrap();
        assert_eq!(config.audio.sample_rate, loaded_config.audio.sample_rate);
    }

    #[test]
    fn test_profile_configs() {
        let balanced = ConfigManager::default_profile_config("balanced");
        let streaming = ConfigManager::default_profile_config("streaming");
        let studio = ConfigManager::default_profile_config("studio");

        assert_eq!(balanced.noise_suppression.strength, 0.7);
        assert_eq!(streaming.noise_suppression.strength, 0.85);
        assert_eq!(studio.noise_suppression.strength, 0.3);
    }
}
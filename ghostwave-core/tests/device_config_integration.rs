//! Integration tests for device detection and configuration management
//!
//! Tests device auto-detection, configuration persistence, and profile management.

use ghostwave_core::{
    Config, DeviceDetector, AudioDevice, AudioDeviceType,
    AudioBackend, ProcessingProfile,
};
use tempfile::TempDir;

// =============================================================================
// Configuration Tests
// =============================================================================

/// Test default configuration
#[test]
fn test_default_config() {
    let config = Config::default();

    // Verify sensible defaults
    assert!(config.audio.sample_rate >= 44100);
    assert!(config.audio.sample_rate <= 192000);
    assert!(config.audio.channels >= 1);
    assert!(config.audio.buffer_size >= 64);
    assert!(config.noise_suppression.strength >= 0.0);
    assert!(config.noise_suppression.strength <= 1.0);
}

/// Test loading built-in profiles
#[test]
fn test_load_builtin_profiles() {
    for profile in ["balanced", "streaming", "studio"] {
        let config = Config::load(profile);
        assert!(config.is_ok(), "Failed to load {} profile", profile);

        let config = config.unwrap();
        assert!(!config.profile.name.is_empty());
        assert!(config.audio.sample_rate > 0);
    }
}

/// Test configuration serialization
#[test]
fn test_config_serialization() {
    let config = Config::load("studio").unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&config).expect("Failed to serialize config");
    assert!(json.contains("96000")); // Studio uses 96kHz
    assert!(json.contains("Studio"));

    // Deserialize back
    let deserialized: Config = serde_json::from_str(&json).expect("Failed to deserialize config");
    assert_eq!(deserialized.audio.sample_rate, 96000);
}

/// Test configuration TOML serialization
#[test]
fn test_config_toml_serialization() {
    let config = Config::default();

    // Serialize to TOML
    let toml_str = toml::to_string(&config).expect("Failed to serialize to TOML");
    assert!(toml_str.contains("sample_rate"));

    // Deserialize back
    let deserialized: Config = toml::from_str(&toml_str).expect("Failed to deserialize TOML");
    assert_eq!(deserialized.audio.sample_rate, config.audio.sample_rate);
}

/// Test configuration file save and load
#[test]
fn test_config_file_persistence() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("ghostwave.toml");

    let config = Config::load("streaming").unwrap();

    // Save to file
    let toml_str = toml::to_string(&config).unwrap();
    std::fs::write(&config_path, &toml_str).expect("Failed to write config file");

    // Load from file
    let loaded_str = std::fs::read_to_string(&config_path).expect("Failed to read config file");
    let loaded: Config = toml::from_str(&loaded_str).expect("Failed to parse config file");

    assert_eq!(loaded.audio.sample_rate, config.audio.sample_rate);
    assert_eq!(loaded.profile.name, config.profile.name);
}

/// Test config with environment variable overrides
/// Note: This test manipulates global environment, so we skip it to avoid interference
/// with other tests. In real usage, run this test in isolation.
#[test]
#[ignore] // Ignored because env vars can interfere with parallel tests
fn test_config_env_overrides() {
    // Set environment variables (unsafe in Rust 2024)
    // SAFETY: This is a test that runs in isolation
    unsafe {
        std::env::set_var("GHOSTWAVE_SAMPLE_RATE", "44100");
        std::env::set_var("GHOSTWAVE_FRAMES", "512");
    }

    let config = Config::load("balanced").unwrap();

    // Environment variables should override
    assert_eq!(config.audio.sample_rate, 44100);
    assert_eq!(config.audio.buffer_size, 512);

    // Clean up
    // SAFETY: This is a test that runs in isolation
    unsafe {
        std::env::remove_var("GHOSTWAVE_SAMPLE_RATE");
        std::env::remove_var("GHOSTWAVE_FRAMES");
    }
}

/// Test config with_overrides method
#[test]
fn test_config_with_overrides() {
    let config = Config::default()
        .with_overrides(Some(96000), Some(256));

    assert_eq!(config.audio.sample_rate, 96000);
    assert_eq!(config.audio.buffer_size, 256);
}

// =============================================================================
// Profile Configuration Tests
// =============================================================================

/// Test balanced profile settings
#[test]
fn test_balanced_profile() {
    let config = Config::load("balanced").unwrap();

    assert_eq!(config.profile.name, "Balanced");
    assert_eq!(config.audio.sample_rate, 48000);
    assert_eq!(config.audio.buffer_size, 128);
    assert!(config.noise_suppression.enabled);
    assert!((config.noise_suppression.strength - 0.7).abs() < 0.01);
}

/// Test streaming profile settings
#[test]
fn test_streaming_profile() {
    let config = Config::load("streaming").unwrap();

    assert_eq!(config.profile.name, "Streaming");
    assert!(config.noise_suppression.enabled);
    // Streaming has higher noise reduction
    assert!(config.noise_suppression.strength > 0.7);
}

/// Test studio profile settings
#[test]
fn test_studio_profile() {
    let config = Config::load("studio").unwrap();

    assert_eq!(config.profile.name, "Studio");
    assert_eq!(config.audio.sample_rate, 96000);
    assert_eq!(config.audio.buffer_size, 256);
    // Studio has gentler noise reduction
    assert!(config.noise_suppression.strength < 0.7);
}

// =============================================================================
// Device Detection Tests
// =============================================================================

/// Test device detector creation
#[test]
fn test_device_detector_creation() {
    let detector = DeviceDetector::new();
    // Should create successfully
    assert!(std::mem::size_of_val(&detector) > 0);
}

/// Test device detection (async)
#[tokio::test]
async fn test_device_detection() {
    let detector = DeviceDetector::new();
    let result = detector.detect_devices().await;

    // Should complete without panic (may return empty list on CI)
    if let Ok(devices) = result {
        for device in &devices {
            // Each device should have valid properties
            assert!(!device.name.is_empty());
            println!("Detected device: {} ({:?})", device.name, device.device_type);
        }
    }
}

/// Test audio device struct
#[test]
fn test_audio_device_struct() {
    let device = AudioDevice {
        name: "Test Device".to_string(),
        id: "test-device-001".to_string(),
        device_type: AudioDeviceType::UsbAudio,
        channels: 2,
        sample_rates: vec![44100, 48000, 96000],
        supported_sample_rates: vec![44100, 48000, 96000],
        supported_buffer_sizes: vec![64, 128, 256, 512],
        vendor: "Test Vendor".to_string(),
        model: "Test Model".to_string(),
        is_xlr_interface: false,
        recommended_profile: "balanced".to_string(),
    };

    assert_eq!(device.name, "Test Device");
    assert_eq!(device.device_type, AudioDeviceType::UsbAudio);
    assert!(device.sample_rates.contains(&48000));
    assert_eq!(device.recommended_profile, "balanced");
}

/// Test audio device type enum
#[test]
fn test_audio_device_types() {
    // Test all device types
    let types = vec![
        AudioDeviceType::XlrInterface,
        AudioDeviceType::UsbMicrophone,
        AudioDeviceType::UsbAudio,
        AudioDeviceType::Microphone,
        AudioDeviceType::Headset,
        AudioDeviceType::LineIn,
        AudioDeviceType::Internal,
        AudioDeviceType::BuiltIn,
        AudioDeviceType::Virtual,
        AudioDeviceType::Unknown,
    ];

    for device_type in types {
        // Ensure Debug trait works
        let _debug = format!("{:?}", device_type);
    }
}

// =============================================================================
// Audio Backend Tests
// =============================================================================

/// Test backend enumeration
#[test]
fn test_backend_enumeration() {
    let backends = AudioBackend::available_backends();

    // Should have at least one backend compiled in
    println!("Available backends: {:?}", backends);

    // All backends should have Display impl
    for backend in &backends {
        let name = format!("{}", backend);
        assert!(!name.is_empty());
    }
}

/// Test backend availability check
#[test]
fn test_backend_availability() {
    let backends = AudioBackend::available_backends();

    for backend in backends {
        let available = backend.is_available();
        println!("Backend {} available: {}", backend, available);
        // Just verify it doesn't panic
    }
}

/// Test recommended backend
#[test]
fn test_recommended_backend() {
    let recommended = AudioBackend::recommended();

    if let Some(backend) = recommended {
        println!("Recommended backend: {}", backend);
        // Recommended backend should be available
        assert!(backend.is_available());
    } else {
        println!("No backend available");
    }
}

// =============================================================================
// Processing Profile Tests
// =============================================================================

/// Test processing profile defaults
#[test]
fn test_processing_profiles() {
    // All profiles should be valid
    let profiles = [
        ProcessingProfile::Streaming,
        ProcessingProfile::Balanced,
        ProcessingProfile::Studio,
    ];

    for profile in profiles {
        let debug = format!("{:?}", profile);
        assert!(!debug.is_empty());
    }
}

/// Test default profile
#[test]
fn test_default_profile() {
    let profile = ProcessingProfile::default();
    // Default should be Balanced or similar
    println!("Default profile: {:?}", profile);
}

/// Test profile display
#[test]
fn test_profile_display() {
    let profile = ProcessingProfile::Studio;
    let display = format!("{}", profile);
    assert!(!display.is_empty());
    assert!(display.contains("Studio") || display.contains("studio"));
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test config with profile settings
#[test]
fn test_config_profile_integration() {
    use ghostwave_core::{GhostWaveProcessor, AudioProcessor};

    // Create processor with config
    let config = Config::default();
    let mut processor = GhostWaveProcessor::new(config).unwrap();

    // Initialize
    processor.init(48000, 1, 256).unwrap();

    // Each profile should work
    for profile in [ProcessingProfile::Streaming, ProcessingProfile::Balanced, ProcessingProfile::Studio] {
        processor.set_profile(profile).unwrap();
        assert_eq!(processor.get_profile(), profile);

        // Process should work with each profile
        let mut buffer = vec![0.1f32; 256];
        assert!(processor.process_inplace(&mut buffer, 256).is_ok());
    }
}

/// Test config modification
#[test]
fn test_config_modification() {
    let mut config = Config::default();

    // Modify audio settings
    config.audio.sample_rate = 96000;
    config.audio.buffer_size = 512;
    config.audio.channels = 2;

    // Modify noise suppression settings
    config.noise_suppression.strength = 0.95;
    config.noise_suppression.gate_threshold = -35.0;

    // Verify changes
    assert_eq!(config.audio.sample_rate, 96000);
    assert_eq!(config.audio.buffer_size, 512);
    assert!((config.noise_suppression.strength - 0.95).abs() < f32::EPSILON);
}

// =============================================================================
// Stress Tests
// =============================================================================

/// Test rapid config creation
#[test]
fn test_rapid_config_creation() {
    for _ in 0..100 {
        let config = Config::default();
        assert!(config.audio.sample_rate > 0);
    }
}

/// Test loading all profiles repeatedly
#[test]
fn test_repeated_profile_loading() {
    for _ in 0..50 {
        for profile in ["balanced", "streaming", "studio"] {
            let config = Config::load(profile);
            assert!(config.is_ok(), "Failed to load {} profile", profile);
        }
    }
}

/// Test concurrent config access
#[test]
fn test_concurrent_config_access() {
    use std::sync::Arc;
    use std::thread;

    let config = Arc::new(Config::default());

    let mut handles = vec![];

    for _ in 0..10 {
        let config = Arc::clone(&config);
        handles.push(thread::spawn(move || {
            // Read config properties concurrently
            for _ in 0..100 {
                let _ = config.audio.sample_rate;
                let _ = config.noise_suppression.strength;
                let _ = config.profile.name.len();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test config JSON round-trip integrity
#[test]
fn test_config_json_roundtrip() {
    for profile in ["balanced", "streaming", "studio"] {
        let original = Config::load(profile).unwrap();

        // Serialize
        let json = serde_json::to_string(&original).unwrap();

        // Deserialize
        let restored: Config = serde_json::from_str(&json).unwrap();

        // Verify integrity
        assert_eq!(original.audio.sample_rate, restored.audio.sample_rate);
        assert_eq!(original.audio.buffer_size, restored.audio.buffer_size);
        assert_eq!(original.audio.channels, restored.audio.channels);
        assert!((original.noise_suppression.strength - restored.noise_suppression.strength).abs() < f32::EPSILON);
        assert_eq!(original.profile.name, restored.profile.name);
    }
}

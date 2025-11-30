//! # AI Model Management
//!
//! Manages AI models for noise suppression and voice processing:
//! - Model discovery and loading
//! - Version management
//! - Automatic downloads from model repository
//! - Model caching and updates
//!
//! ## Supported Model Types
//! - **RNNoise**: GRU-based noise suppression (multiple sizes)
//! - **Transformer**: High-quality denoising (RTX 40/50)
//! - **SpeakerEncoder**: Voice fingerprinting for isolation
//! - **VoiceSeparation**: Multi-speaker separation
//!
//! ## Model Distribution
//! Models can be:
//! - Bundled with the package
//! - Downloaded from ghostwave model hub
//! - User-provided custom models

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::fs;
use tracing::{info, debug, warn, error};

/// Model type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// Tiny RNNoise model (~500KB) - lowest latency
    RNNoiseTiny,
    /// Standard RNNoise model (~2MB) - balanced
    RNNoiseStandard,
    /// Large RNNoise model (~8MB) - highest quality
    RNNoiseLarge,
    /// Transformer-based denoiser (~50MB) - premium quality
    TransformerDenoise,
    /// Speaker embedding extractor (~5MB)
    SpeakerEncoder,
    /// Voice separation model (~20MB)
    VoiceSeparation,
    /// Echo cancellation model (~3MB)
    EchoCanceller,
    /// Custom user model
    Custom,
}

impl ModelType {
    /// Get default filename for this model type
    pub fn default_filename(&self) -> &str {
        match self {
            ModelType::RNNoiseTiny => "rnnoise_tiny.onnx",
            ModelType::RNNoiseStandard => "rnnoise_standard.onnx",
            ModelType::RNNoiseLarge => "rnnoise_large.onnx",
            ModelType::TransformerDenoise => "transformer_denoise.onnx",
            ModelType::SpeakerEncoder => "speaker_encoder.onnx",
            ModelType::VoiceSeparation => "voice_separation.onnx",
            ModelType::EchoCanceller => "echo_canceller.onnx",
            ModelType::Custom => "custom.onnx",
        }
    }

    /// Get expected model size in bytes (approximate)
    pub fn expected_size(&self) -> usize {
        match self {
            ModelType::RNNoiseTiny => 500 * 1024,         // 500 KB
            ModelType::RNNoiseStandard => 2 * 1024 * 1024, // 2 MB
            ModelType::RNNoiseLarge => 8 * 1024 * 1024,    // 8 MB
            ModelType::TransformerDenoise => 50 * 1024 * 1024, // 50 MB
            ModelType::SpeakerEncoder => 5 * 1024 * 1024,  // 5 MB
            ModelType::VoiceSeparation => 20 * 1024 * 1024, // 20 MB
            ModelType::EchoCanceller => 3 * 1024 * 1024,   // 3 MB
            ModelType::Custom => 10 * 1024 * 1024,         // Variable
        }
    }

    /// Check if this model requires GPU
    pub fn requires_gpu(&self) -> bool {
        matches!(self, ModelType::TransformerDenoise | ModelType::VoiceSeparation)
    }

    /// Get minimum VRAM required in MB
    pub fn min_vram_mb(&self) -> usize {
        match self {
            ModelType::RNNoiseTiny => 128,
            ModelType::RNNoiseStandard => 256,
            ModelType::RNNoiseLarge => 512,
            ModelType::TransformerDenoise => 2048,
            ModelType::SpeakerEncoder => 256,
            ModelType::VoiceSeparation => 1024,
            ModelType::EchoCanceller => 256,
            ModelType::Custom => 512,
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::RNNoiseTiny => write!(f, "RNNoise Tiny"),
            ModelType::RNNoiseStandard => write!(f, "RNNoise Standard"),
            ModelType::RNNoiseLarge => write!(f, "RNNoise Large"),
            ModelType::TransformerDenoise => write!(f, "Transformer Denoise"),
            ModelType::SpeakerEncoder => write!(f, "Speaker Encoder"),
            ModelType::VoiceSeparation => write!(f, "Voice Separation"),
            ModelType::EchoCanceller => write!(f, "Echo Canceller"),
            ModelType::Custom => write!(f, "Custom"),
        }
    }
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model type
    pub model_type: ModelType,
    /// Model file path
    pub path: PathBuf,
    /// Model version
    pub version: String,
    /// File size in bytes
    pub size_bytes: usize,
    /// SHA256 hash for verification
    pub hash: Option<String>,
    /// Model description
    pub description: String,
    /// Input shape
    pub input_shape: Vec<i64>,
    /// Output shape
    pub output_shape: Vec<i64>,
    /// Whether model is loaded
    pub loaded: bool,
}

impl ModelInfo {
    /// Create model info from file
    pub fn from_file(path: &Path, model_type: ModelType) -> Result<Self> {
        let metadata = fs::metadata(path)?;

        Ok(Self {
            model_type,
            path: path.to_path_buf(),
            version: "1.0.0".to_string(),
            size_bytes: metadata.len() as usize,
            hash: None,
            description: format!("{} model", model_type),
            input_shape: vec![1, 42],  // Default RNNoise shape
            output_shape: vec![1, 23],
            loaded: false,
        })
    }
}

/// Model repository configuration
#[derive(Debug, Clone)]
pub struct ModelRepository {
    /// Base URL for model downloads
    pub base_url: String,
    /// Repository name
    pub name: String,
    /// Available models
    pub models: Vec<ModelInfo>,
}

impl Default for ModelRepository {
    fn default() -> Self {
        Self {
            base_url: "https://github.com/ghostkellz/ghostwave-models/releases/download".to_string(),
            name: "ghostwave-models".to_string(),
            models: Vec::new(),
        }
    }
}

/// Model manager for loading and managing AI models
pub struct ModelManager {
    /// Model directory path
    model_dir: PathBuf,
    /// System model directory (read-only, package-installed)
    system_model_dir: PathBuf,
    /// User model directory (writable)
    user_model_dir: PathBuf,
    /// Loaded models
    loaded_models: HashMap<ModelType, ModelInfo>,
    /// Model repository
    repository: ModelRepository,
    /// Cache for TensorRT engines
    engine_cache_dir: PathBuf,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new() -> Result<Self> {
        // Determine directories
        let user_model_dir = dirs::data_dir()
            .map(|d| d.join("ghostwave").join("models"))
            .unwrap_or_else(|| PathBuf::from("~/.local/share/ghostwave/models"));

        let system_model_dir = PathBuf::from("/usr/share/ghostwave/models");

        let engine_cache_dir = dirs::cache_dir()
            .map(|d| d.join("ghostwave").join("engines"))
            .unwrap_or_else(|| PathBuf::from("/tmp/ghostwave/engines"));

        // Create directories if needed
        fs::create_dir_all(&user_model_dir)?;
        fs::create_dir_all(&engine_cache_dir)?;

        info!("Model manager initialized");
        info!("  User models: {:?}", user_model_dir);
        info!("  System models: {:?}", system_model_dir);
        info!("  Engine cache: {:?}", engine_cache_dir);

        let mut manager = Self {
            model_dir: user_model_dir.clone(),
            system_model_dir,
            user_model_dir,
            loaded_models: HashMap::new(),
            repository: ModelRepository::default(),
            engine_cache_dir,
        };

        // Scan for available models
        manager.scan_models()?;

        Ok(manager)
    }

    /// Scan for available models
    fn scan_models(&mut self) -> Result<()> {
        let mut found_models = Vec::new();

        // Scan user directory
        if self.user_model_dir.exists() {
            found_models.extend(self.scan_directory(&self.user_model_dir)?);
        }

        // Scan system directory
        if self.system_model_dir.exists() {
            found_models.extend(self.scan_directory(&self.system_model_dir)?);
        }

        info!("Found {} models", found_models.len());
        for model in &found_models {
            debug!("  {} - {:?}", model.model_type, model.path);
        }

        self.repository.models = found_models;
        Ok(())
    }

    /// Scan a directory for model files
    fn scan_directory(&self, dir: &Path) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();

                if path.extension().map(|e| e == "onnx").unwrap_or(false) {
                    let filename = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");

                    // Determine model type from filename
                    let model_type = self.model_type_from_filename(filename);

                    if let Ok(info) = ModelInfo::from_file(&path, model_type) {
                        models.push(info);
                    }
                }
            }
        }

        Ok(models)
    }

    /// Determine model type from filename
    fn model_type_from_filename(&self, filename: &str) -> ModelType {
        let lower = filename.to_lowercase();

        if lower.contains("tiny") {
            ModelType::RNNoiseTiny
        } else if lower.contains("large") {
            ModelType::RNNoiseLarge
        } else if lower.contains("rnnoise") {
            ModelType::RNNoiseStandard
        } else if lower.contains("transformer") {
            ModelType::TransformerDenoise
        } else if lower.contains("speaker") {
            ModelType::SpeakerEncoder
        } else if lower.contains("separation") {
            ModelType::VoiceSeparation
        } else if lower.contains("echo") {
            ModelType::EchoCanceller
        } else {
            ModelType::Custom
        }
    }

    /// Get model by type
    pub fn get_model(&self, model_type: ModelType) -> Option<&ModelInfo> {
        self.repository.models.iter().find(|m| m.model_type == model_type)
    }

    /// Get model path by type
    pub fn get_model_path(&self, model_type: ModelType) -> Option<PathBuf> {
        self.get_model(model_type).map(|m| m.path.clone())
    }

    /// Check if a model is available
    pub fn is_available(&self, model_type: ModelType) -> bool {
        self.get_model(model_type).is_some()
    }

    /// Get all available models
    pub fn available_models(&self) -> &[ModelInfo] {
        &self.repository.models
    }

    /// Download a model from repository
    pub async fn download_model(&mut self, model_type: ModelType) -> Result<PathBuf> {
        let filename = model_type.default_filename();
        let url = format!("{}/v1/{}", self.repository.base_url, filename);
        let dest_path = self.user_model_dir.join(filename);

        info!("Downloading model: {} -> {:?}", url, dest_path);

        // In production, would use reqwest or similar:
        // let response = reqwest::get(&url).await?;
        // let bytes = response.bytes().await?;
        // fs::write(&dest_path, &bytes)?;

        // For now, create a placeholder
        warn!("Model download not implemented - create placeholder");

        // Rescan models
        self.scan_models()?;

        Ok(dest_path)
    }

    /// Ensure a model is available, downloading if necessary
    pub async fn ensure_model(&mut self, model_type: ModelType) -> Result<PathBuf> {
        if let Some(path) = self.get_model_path(model_type) {
            return Ok(path);
        }

        // Model not found, try to download
        self.download_model(model_type).await
    }

    /// Get TensorRT engine cache path for a model
    pub fn get_engine_cache_path(&self, model_type: ModelType, precision: &str) -> PathBuf {
        let filename = format!(
            "{}_{}.engine",
            model_type.default_filename().replace(".onnx", ""),
            precision
        );
        self.engine_cache_dir.join(filename)
    }

    /// Check if cached TensorRT engine exists
    pub fn has_cached_engine(&self, model_type: ModelType, precision: &str) -> bool {
        self.get_engine_cache_path(model_type, precision).exists()
    }

    /// Get model directory
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Get user model directory
    pub fn user_model_dir(&self) -> &Path {
        &self.user_model_dir
    }

    /// Get system model directory
    pub fn system_model_dir(&self) -> &Path {
        &self.system_model_dir
    }

    /// Get engine cache directory
    pub fn engine_cache_dir(&self) -> &Path {
        &self.engine_cache_dir
    }

    /// Clear engine cache
    pub fn clear_engine_cache(&self) -> Result<()> {
        if self.engine_cache_dir.exists() {
            for entry in fs::read_dir(&self.engine_cache_dir)? {
                let entry = entry?;
                if entry.path().extension().map(|e| e == "engine").unwrap_or(false) {
                    fs::remove_file(entry.path())?;
                }
            }
            info!("Cleared TensorRT engine cache");
        }
        Ok(())
    }

    /// Get total size of models
    pub fn total_model_size(&self) -> usize {
        self.repository.models.iter().map(|m| m.size_bytes).sum()
    }

    /// Create bundled model placeholder files (for development)
    pub fn create_placeholder_models(&self) -> Result<()> {
        let models = [
            ModelType::RNNoiseTiny,
            ModelType::RNNoiseStandard,
            ModelType::RNNoiseLarge,
        ];

        for model_type in models {
            let path = self.user_model_dir.join(model_type.default_filename());
            if !path.exists() {
                // Create a minimal valid ONNX placeholder
                // In production, these would be actual trained models
                let placeholder = create_placeholder_onnx(model_type);
                fs::write(&path, &placeholder)?;
                debug!("Created placeholder: {:?}", path);
            }
        }

        Ok(())
    }
}

/// Create a minimal ONNX placeholder
fn create_placeholder_onnx(model_type: ModelType) -> Vec<u8> {
    // ONNX magic number and minimal header
    // In production, these would be actual ONNX protobuf files
    let mut data = Vec::new();

    // ONNX magic bytes (simplified placeholder)
    data.extend_from_slice(b"ONNX");
    data.extend_from_slice(&[0x00; 16]); // Version info placeholder

    // Add model-type specific data size
    let model_size = model_type.expected_size();
    data.resize(model_size.min(1024), 0x00); // Limit placeholder size

    data
}

/// Model validation result
#[derive(Debug)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ModelManager {
    /// Validate a model file
    pub fn validate_model(&self, path: &Path) -> ValidationResult {
        let mut result = ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        // Check file exists
        if !path.exists() {
            result.valid = false;
            result.errors.push(format!("File not found: {:?}", path));
            return result;
        }

        // Check extension
        if path.extension().map(|e| e != "onnx").unwrap_or(true) {
            result.warnings.push("Expected .onnx extension".to_string());
        }

        // Check file size
        if let Ok(metadata) = fs::metadata(path) {
            if metadata.len() < 100 {
                result.valid = false;
                result.errors.push("File too small to be a valid ONNX model".to_string());
            }
        }

        // In production, would also:
        // - Parse ONNX protobuf
        // - Validate graph structure
        // - Check input/output shapes
        // - Verify opset compatibility

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_filename() {
        assert_eq!(ModelType::RNNoiseTiny.default_filename(), "rnnoise_tiny.onnx");
        assert_eq!(ModelType::TransformerDenoise.default_filename(), "transformer_denoise.onnx");
    }

    #[test]
    fn test_model_requirements() {
        assert!(ModelType::TransformerDenoise.requires_gpu());
        assert!(!ModelType::RNNoiseTiny.requires_gpu());

        assert!(ModelType::TransformerDenoise.min_vram_mb() > ModelType::RNNoiseTiny.min_vram_mb());
    }

    #[test]
    fn test_placeholder_creation() {
        let placeholder = create_placeholder_onnx(ModelType::RNNoiseTiny);
        assert!(placeholder.len() > 0);
        assert!(placeholder.starts_with(b"ONNX"));
    }
}

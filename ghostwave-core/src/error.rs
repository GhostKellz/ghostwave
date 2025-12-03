//! # GhostWave Error Types
//!
//! Custom error types for the ghostwave-core library.
//! Provides structured errors for library consumers with C FFI compatibility.

use std::fmt;
use std::error::Error;

/// GhostWave library error type
///
/// Provides detailed, actionable error information for library consumers.
/// Each variant includes context to help diagnose and recover from errors.
#[derive(Debug, Clone)]
pub enum GhostWaveError {
    // === Initialization Errors (1xx) ===
    /// GPU is not available or unsupported
    GpuNotAvailable {
        reason: String,
        fallback_available: bool,
    },

    /// Configuration is invalid
    InvalidConfiguration {
        field: String,
        value: String,
        reason: String,
    },

    /// Required model file not found
    ModelNotFound {
        model_type: String,
        search_paths: Vec<String>,
    },

    /// Failed to load AI model
    ModelLoadFailed {
        path: String,
        reason: String,
    },

    /// Unsupported sample rate
    UnsupportedSampleRate {
        rate: u32,
        supported: Vec<u32>,
    },

    /// Unsupported buffer size
    UnsupportedBufferSize {
        size: usize,
        min: usize,
        max: usize,
    },

    // === Runtime Processing Errors (2xx) ===
    /// Audio processing failed
    ProcessingError {
        stage: String,
        reason: String,
    },

    /// Input/output buffer size mismatch
    BufferMismatch {
        expected: usize,
        got: usize,
    },

    /// Buffer too small for operation
    BufferTooSmall {
        required: usize,
        provided: usize,
    },

    /// Internal lock acquisition failed
    LockError {
        resource: String,
    },

    /// Processor not initialized
    NotInitialized,

    // === Device Errors (3xx) ===
    /// Audio device error
    AudioDeviceError {
        device: String,
        operation: String,
        reason: String,
    },

    /// Device not found
    DeviceNotFound {
        device_name: String,
    },

    /// Device busy or unavailable
    DeviceBusy {
        device: String,
    },

    // === Resource Errors (4xx) ===
    /// Insufficient system memory
    InsufficientMemory {
        required_mb: usize,
        available_mb: usize,
    },

    /// Insufficient GPU VRAM
    InsufficientVram {
        required_mb: u64,
        available_mb: u64,
    },

    /// Resource allocation failed
    AllocationFailed {
        resource: String,
        size: usize,
    },

    // === IO Errors (5xx) ===
    /// File read/write error
    IoError {
        path: String,
        operation: String,
        reason: String,
    },

    /// Network error (model download)
    NetworkError {
        url: String,
        reason: String,
    },

    // === FFI Errors (6xx) ===
    /// FFI null pointer
    FfiNullPointer {
        parameter: String,
    },

    /// FFI invalid handle
    FfiInvalidHandle,

    /// FFI string conversion error
    FfiStringError {
        reason: String,
    },

    /// FFI panic caught
    FfiPanic {
        message: String,
    },

    // === Generic Errors (9xx) ===
    /// Unknown/uncategorized error
    Unknown {
        message: String,
    },
}

impl GhostWaveError {
    /// Get error code for C FFI
    pub fn code(&self) -> i32 {
        match self {
            // Initialization (1xx)
            Self::GpuNotAvailable { .. } => 101,
            Self::InvalidConfiguration { .. } => 102,
            Self::ModelNotFound { .. } => 103,
            Self::ModelLoadFailed { .. } => 104,
            Self::UnsupportedSampleRate { .. } => 105,
            Self::UnsupportedBufferSize { .. } => 106,

            // Runtime (2xx)
            Self::ProcessingError { .. } => 201,
            Self::BufferMismatch { .. } => 202,
            Self::BufferTooSmall { .. } => 203,
            Self::LockError { .. } => 204,
            Self::NotInitialized => 205,

            // Device (3xx)
            Self::AudioDeviceError { .. } => 301,
            Self::DeviceNotFound { .. } => 302,
            Self::DeviceBusy { .. } => 303,

            // Resource (4xx)
            Self::InsufficientMemory { .. } => 401,
            Self::InsufficientVram { .. } => 402,
            Self::AllocationFailed { .. } => 403,

            // IO (5xx)
            Self::IoError { .. } => 501,
            Self::NetworkError { .. } => 502,

            // FFI (6xx)
            Self::FfiNullPointer { .. } => 601,
            Self::FfiInvalidHandle => 602,
            Self::FfiStringError { .. } => 603,
            Self::FfiPanic { .. } => 604,

            // Unknown (9xx)
            Self::Unknown { .. } => 999,
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::GpuNotAvailable { fallback_available: true, .. }
                | Self::BufferMismatch { .. }
                | Self::BufferTooSmall { .. }
                | Self::LockError { .. }
                | Self::DeviceBusy { .. }
        )
    }

    /// Get recovery suggestion
    pub fn recovery_suggestion(&self) -> Option<&'static str> {
        match self {
            Self::GpuNotAvailable { fallback_available: true, .. } => {
                Some("CPU fallback available - processing will continue with higher latency")
            }
            Self::BufferMismatch { .. } | Self::BufferTooSmall { .. } => {
                Some("Resize buffer to match expected size")
            }
            Self::LockError { .. } => Some("Retry operation - temporary contention"),
            Self::DeviceBusy { .. } => Some("Wait and retry - device may become available"),
            Self::NotInitialized => Some("Call init() before processing"),
            Self::ModelNotFound { .. } => {
                Some("Download models or set GHOSTWAVE_MODEL_PATH environment variable")
            }
            _ => None,
        }
    }
}

impl fmt::Display for GhostWaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuNotAvailable { reason, fallback_available } => {
                write!(f, "GPU not available: {} (fallback: {})", reason, fallback_available)
            }
            Self::InvalidConfiguration { field, value, reason } => {
                write!(f, "Invalid configuration: {} = '{}' - {}", field, value, reason)
            }
            Self::ModelNotFound { model_type, search_paths } => {
                write!(f, "Model '{}' not found in: {:?}", model_type, search_paths)
            }
            Self::ModelLoadFailed { path, reason } => {
                write!(f, "Failed to load model '{}': {}", path, reason)
            }
            Self::UnsupportedSampleRate { rate, supported } => {
                write!(f, "Unsupported sample rate {}Hz (supported: {:?})", rate, supported)
            }
            Self::UnsupportedBufferSize { size, min, max } => {
                write!(f, "Buffer size {} out of range [{}, {}]", size, min, max)
            }
            Self::ProcessingError { stage, reason } => {
                write!(f, "Processing error in {}: {}", stage, reason)
            }
            Self::BufferMismatch { expected, got } => {
                write!(f, "Buffer size mismatch: expected {}, got {}", expected, got)
            }
            Self::BufferTooSmall { required, provided } => {
                write!(f, "Buffer too small: need {}, have {}", required, provided)
            }
            Self::LockError { resource } => {
                write!(f, "Failed to acquire lock on '{}'", resource)
            }
            Self::NotInitialized => {
                write!(f, "Processor not initialized - call init() first")
            }
            Self::AudioDeviceError { device, operation, reason } => {
                write!(f, "Audio device '{}' error during {}: {}", device, operation, reason)
            }
            Self::DeviceNotFound { device_name } => {
                write!(f, "Audio device not found: {}", device_name)
            }
            Self::DeviceBusy { device } => {
                write!(f, "Audio device '{}' is busy", device)
            }
            Self::InsufficientMemory { required_mb, available_mb } => {
                write!(f, "Insufficient memory: need {}MB, have {}MB", required_mb, available_mb)
            }
            Self::InsufficientVram { required_mb, available_mb } => {
                write!(f, "Insufficient VRAM: need {}MB, have {}MB", required_mb, available_mb)
            }
            Self::AllocationFailed { resource, size } => {
                write!(f, "Failed to allocate {} bytes for '{}'", size, resource)
            }
            Self::IoError { path, operation, reason } => {
                write!(f, "IO error {} '{}': {}", operation, path, reason)
            }
            Self::NetworkError { url, reason } => {
                write!(f, "Network error fetching '{}': {}", url, reason)
            }
            Self::FfiNullPointer { parameter } => {
                write!(f, "FFI null pointer: {}", parameter)
            }
            Self::FfiInvalidHandle => {
                write!(f, "FFI invalid handle")
            }
            Self::FfiStringError { reason } => {
                write!(f, "FFI string error: {}", reason)
            }
            Self::FfiPanic { message } => {
                write!(f, "FFI panic caught: {}", message)
            }
            Self::Unknown { message } => {
                write!(f, "Unknown error: {}", message)
            }
        }
    }
}

impl Error for GhostWaveError {}

// Conversion from anyhow for internal use
impl From<anyhow::Error> for GhostWaveError {
    fn from(err: anyhow::Error) -> Self {
        Self::Unknown {
            message: err.to_string(),
        }
    }
}

// Conversion from std::io::Error
impl From<std::io::Error> for GhostWaveError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            path: String::new(),
            operation: "unknown".to_string(),
            reason: err.to_string(),
        }
    }
}

/// Result type alias using GhostWaveError
pub type Result<T> = std::result::Result<T, GhostWaveError>;

/// C-compatible error structure for FFI
#[repr(C)]
pub struct CError {
    /// Error code (0 = success)
    pub code: i32,
    /// Error message (null-terminated, max 255 chars)
    pub message: [u8; 256],
}

impl CError {
    /// Create success result
    pub fn success() -> Self {
        Self {
            code: 0,
            message: [0; 256],
        }
    }

    /// Create error from GhostWaveError
    pub fn from_error(err: &GhostWaveError) -> Self {
        let mut message = [0u8; 256];
        let msg = err.to_string();
        let bytes = msg.as_bytes();
        let len = bytes.len().min(255);
        message[..len].copy_from_slice(&bytes[..len]);

        Self {
            code: err.code(),
            message,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let err = GhostWaveError::GpuNotAvailable {
            reason: "No CUDA".to_string(),
            fallback_available: true,
        };
        assert_eq!(err.code(), 101);
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_error_display() {
        let err = GhostWaveError::BufferMismatch {
            expected: 256,
            got: 512,
        };
        assert!(err.to_string().contains("256"));
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn test_c_error() {
        let err = GhostWaveError::NotInitialized;
        let c_err = CError::from_error(&err);
        assert_eq!(c_err.code, 205);
    }
}

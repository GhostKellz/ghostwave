//! # C FFI Bindings for GhostWave
//!
//! Safe C API for non-Rust consumers (PhantomLink, etc.)
//!
//! ## Usage from C
//!
//! ```c
//! #include "ghostwave.h"
//!
//! GhostWaveHandle handle;
//! GhostWaveError err = ghostwave_create(&handle, 48000, 1, 256);
//! if (err.code != 0) {
//!     printf("Error: %s\n", err.message);
//!     return 1;
//! }
//!
//! float input[256], output[256];
//! err = ghostwave_process(handle, input, output, 256);
//!
//! ghostwave_destroy(handle);
//! ```

use std::ffi::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use crate::error::{CError, GhostWaveError};
use crate::{Config, GhostWaveProcessor};

#[cfg(test)]
use std::ffi::CStr;

/// Opaque handle to GhostWave processor
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GhostWaveHandle {
    ptr: *mut c_void,
}

impl GhostWaveHandle {
    fn new(processor: Box<GhostWaveProcessor>) -> Self {
        Self {
            ptr: Box::into_raw(processor) as *mut c_void,
        }
    }

    /// Create a null handle (for use before ghostwave_create)
    pub fn null() -> Self {
        Self { ptr: ptr::null_mut() }
    }

    fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    unsafe fn as_processor(&self) -> Option<&GhostWaveProcessor> {
        if self.is_null() {
            None
        } else {
            // SAFETY: Caller ensures ptr is valid and points to GhostWaveProcessor
            Some(unsafe { &*(self.ptr as *const GhostWaveProcessor) })
        }
    }

    #[allow(dead_code)] // Reserved for future mutable operations API
    unsafe fn as_processor_mut(&self) -> Option<&mut GhostWaveProcessor> {
        if self.is_null() {
            None
        } else {
            // SAFETY: Caller ensures ptr is valid and points to GhostWaveProcessor
            Some(unsafe { &mut *(self.ptr as *mut GhostWaveProcessor) })
        }
    }

    unsafe fn take(self) -> Option<Box<GhostWaveProcessor>> {
        if self.is_null() {
            None
        } else {
            // SAFETY: Caller ensures ptr was created by Box::into_raw
            Some(unsafe { Box::from_raw(self.ptr as *mut GhostWaveProcessor) })
        }
    }
}

/// Processing profile enum for C
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GhostWaveProfile {
    Balanced = 0,
    Streaming = 1,
    Studio = 2,
}

/// GPU status info
#[repr(C)]
pub struct GhostWaveGpuInfo {
    pub available: bool,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub memory_gb: f32,
    pub name: [u8; 64],
}

/// Audio statistics
#[repr(C)]
pub struct GhostWaveStats {
    pub frames_processed: u64,
    pub xrun_count: u64,
    pub latency_us: u64,
    pub cpu_usage_pct: f32,
}

// ============================================================================
// Core API Functions
// ============================================================================

/// Create a new GhostWave processor
///
/// # Safety
/// - `handle_out` must be a valid pointer
/// - Caller must call `ghostwave_destroy` when done
///
/// # Parameters
/// - `handle_out`: Output pointer for the created handle
/// - `sample_rate`: Audio sample rate (44100, 48000, 96000, 192000)
/// - `channels`: Number of audio channels (1 or 2)
/// - `buffer_size`: Processing buffer size in frames (32-4096, power of 2)
///
/// # Returns
/// Error code (0 = success)
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_create(
    handle_out: *mut GhostWaveHandle,
    sample_rate: u32,
    channels: u32,
    buffer_size: u32,
) -> CError {
    if handle_out.is_null() {
        return CError::from_error(&GhostWaveError::FfiNullPointer {
            parameter: "handle_out".to_string(),
        });
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        // Validate parameters
        if ![44100, 48000, 96000, 192000].contains(&sample_rate) {
            return Err(GhostWaveError::UnsupportedSampleRate {
                rate: sample_rate,
                supported: vec![44100, 48000, 96000, 192000],
            });
        }

        if channels < 1 || channels > 2 {
            return Err(GhostWaveError::InvalidConfiguration {
                field: "channels".to_string(),
                value: channels.to_string(),
                reason: "Must be 1 or 2".to_string(),
            });
        }

        if buffer_size < 32 || buffer_size > 4096 || !buffer_size.is_power_of_two() {
            return Err(GhostWaveError::UnsupportedBufferSize {
                size: buffer_size as usize,
                min: 32,
                max: 4096,
            });
        }

        // Create config
        let mut config = Config::default();
        config.audio.sample_rate = sample_rate;
        config.audio.channels = channels as u8;
        config.audio.buffer_size = buffer_size;

        // Create processor
        let processor = GhostWaveProcessor::new(config)
            .map_err(|e| GhostWaveError::Unknown { message: e.to_string() })?;

        Ok(Box::new(processor))
    }));

    match result {
        Ok(Ok(processor)) => {
            unsafe {
                *handle_out = GhostWaveHandle::new(processor);
            }
            CError::success()
        }
        Ok(Err(e)) => {
            unsafe {
                *handle_out = GhostWaveHandle::null();
            }
            CError::from_error(&e)
        }
        Err(_) => {
            unsafe {
                *handle_out = GhostWaveHandle::null();
            }
            CError::from_error(&GhostWaveError::FfiPanic {
                message: "Panic during processor creation".to_string(),
            })
        }
    }
}

/// Create processor with a specific profile
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_create_with_profile(
    handle_out: *mut GhostWaveHandle,
    sample_rate: u32,
    channels: u32,
    buffer_size: u32,
    profile: GhostWaveProfile,
) -> CError {
    if handle_out.is_null() {
        return CError::from_error(&GhostWaveError::FfiNullPointer {
            parameter: "handle_out".to_string(),
        });
    }

    let profile_name = match profile {
        GhostWaveProfile::Balanced => "balanced",
        GhostWaveProfile::Streaming => "streaming",
        GhostWaveProfile::Studio => "studio",
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        let config = Config::load(profile_name)
            .map_err(|e| GhostWaveError::Unknown { message: e.to_string() })?;

        let mut config = config;
        config.audio.sample_rate = sample_rate;
        config.audio.channels = channels as u8;
        config.audio.buffer_size = buffer_size;

        let processor = GhostWaveProcessor::new(config)
            .map_err(|e| GhostWaveError::Unknown { message: e.to_string() })?;

        Ok(Box::new(processor))
    }));

    match result {
        Ok(Ok(processor)) => {
            unsafe {
                *handle_out = GhostWaveHandle::new(processor);
            }
            CError::success()
        }
        Ok(Err(e)) => {
            unsafe {
                *handle_out = GhostWaveHandle::null();
            }
            CError::from_error(&e)
        }
        Err(_) => {
            unsafe {
                *handle_out = GhostWaveHandle::null();
            }
            CError::from_error(&GhostWaveError::FfiPanic {
                message: "Panic during processor creation".to_string(),
            })
        }
    }
}

/// Destroy a GhostWave processor and free resources
///
/// # Safety
/// - `handle` must be a valid handle from `ghostwave_create`
/// - Handle must not be used after this call
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_destroy(handle: GhostWaveHandle) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            if let Some(processor) = handle.take() {
                drop(processor);
            }
        }
    }));
}

/// Process audio through noise suppression
///
/// # Safety
/// - `handle` must be valid
/// - `input` must point to `frames` float samples
/// - `output` must point to `frames` float samples (writable)
///
/// # Parameters
/// - `handle`: Processor handle
/// - `input`: Input audio samples (f32, -1.0 to 1.0)
/// - `output`: Output buffer for processed audio
/// - `frames`: Number of samples to process
///
/// # Returns
/// Error code (0 = success)
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_process(
    handle: GhostWaveHandle,
    input: *const f32,
    output: *mut f32,
    frames: usize,
) -> CError {
    if handle.is_null() {
        return CError::from_error(&GhostWaveError::FfiInvalidHandle);
    }
    if input.is_null() {
        return CError::from_error(&GhostWaveError::FfiNullPointer {
            parameter: "input".to_string(),
        });
    }
    if output.is_null() {
        return CError::from_error(&GhostWaveError::FfiNullPointer {
            parameter: "output".to_string(),
        });
    }
    if frames == 0 {
        return CError::success();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            let processor = handle.as_processor().ok_or(GhostWaveError::FfiInvalidHandle)?;

            // Create slices from raw pointers
            let input_slice = std::slice::from_raw_parts(input, frames);
            let output_slice = std::slice::from_raw_parts_mut(output, frames);

            processor
                .process(input_slice, output_slice)
                .map_err(|e| GhostWaveError::ProcessingError {
                    stage: "noise_suppression".to_string(),
                    reason: e.to_string(),
                })
        }
    }));

    match result {
        Ok(Ok(())) => CError::success(),
        Ok(Err(e)) => CError::from_error(&e),
        Err(_) => CError::from_error(&GhostWaveError::FfiPanic {
            message: "Panic during audio processing".to_string(),
        }),
    }
}

/// Process audio in-place (input buffer is modified)
///
/// # Safety
/// Same as `ghostwave_process`, but `buffer` is both input and output
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_process_inplace(
    handle: GhostWaveHandle,
    buffer: *mut f32,
    frames: usize,
) -> CError {
    if handle.is_null() {
        return CError::from_error(&GhostWaveError::FfiInvalidHandle);
    }
    if buffer.is_null() {
        return CError::from_error(&GhostWaveError::FfiNullPointer {
            parameter: "buffer".to_string(),
        });
    }
    if frames == 0 {
        return CError::success();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            let processor = handle.as_processor().ok_or(GhostWaveError::FfiInvalidHandle)?;

            let buffer_slice = std::slice::from_raw_parts_mut(buffer, frames);

            // Create temp input copy for processing
            let input_copy: Vec<f32> = buffer_slice.to_vec();

            processor
                .process(&input_copy, buffer_slice)
                .map_err(|e| GhostWaveError::ProcessingError {
                    stage: "noise_suppression".to_string(),
                    reason: e.to_string(),
                })
        }
    }));

    match result {
        Ok(Ok(())) => CError::success(),
        Ok(Err(e)) => CError::from_error(&e),
        Err(_) => CError::from_error(&GhostWaveError::FfiPanic {
            message: "Panic during audio processing".to_string(),
        }),
    }
}

// ============================================================================
// Configuration Functions
// ============================================================================

/// Set noise suppression strength
///
/// # Parameters
/// - `strength`: 0.0 (disabled) to 1.0 (maximum)
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_set_noise_strength(handle: GhostWaveHandle, strength: f32) -> CError {
    if handle.is_null() {
        return CError::from_error(&GhostWaveError::FfiInvalidHandle);
    }
    if strength < 0.0 || strength > 1.0 {
        return CError::from_error(&GhostWaveError::InvalidConfiguration {
            field: "strength".to_string(),
            value: strength.to_string(),
            reason: "Must be between 0.0 and 1.0".to_string(),
        });
    }

    // Note: This would need mutable access to update the config
    // For now, return success (config update would be implemented)
    CError::success()
}

/// Enable or disable noise suppression
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_set_enabled(handle: GhostWaveHandle, _enabled: bool) -> CError {
    if handle.is_null() {
        return CError::from_error(&GhostWaveError::FfiInvalidHandle);
    }

    // TODO: Implement enable/disable via processor state
    CError::success()
}

// ============================================================================
// Query Functions
// ============================================================================

/// Get GPU acceleration status
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_get_gpu_info(handle: GhostWaveHandle, info_out: *mut GhostWaveGpuInfo) -> CError {
    if handle.is_null() {
        return CError::from_error(&GhostWaveError::FfiInvalidHandle);
    }
    if info_out.is_null() {
        return CError::from_error(&GhostWaveError::FfiNullPointer {
            parameter: "info_out".to_string(),
        });
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            let processor = handle.as_processor().ok_or(GhostWaveError::FfiInvalidHandle)?;

            #[allow(unused_mut)]
            let mut info = GhostWaveGpuInfo {
                available: processor.has_rtx_acceleration(),
                compute_major: 0,
                compute_minor: 0,
                memory_gb: 0.0,
                name: [0; 64],
            };

            #[cfg(feature = "nvidia-rtx")]
            if let Some(caps) = processor.get_rtx_capabilities() {
                info.compute_major = caps.compute_capability.0 as i32;
                info.compute_minor = caps.compute_capability.1 as i32;
                info.memory_gb = caps.memory_gb;

                // Derive name from generation
                let name = format!("RTX {:?}", caps.gpu_generation);
                let name_bytes = name.as_bytes();
                let len = name_bytes.len().min(63);
                info.name[..len].copy_from_slice(&name_bytes[..len]);
            }

            *info_out = info;
            Ok(())
        }
    }));

    match result {
        Ok(Ok(())) => CError::success(),
        Ok(Err(e)) => CError::from_error(&e),
        Err(_) => CError::from_error(&GhostWaveError::FfiPanic {
            message: "Panic getting GPU info".to_string(),
        }),
    }
}

/// Get current processing mode string
///
/// # Safety
/// - `mode_out` must point to at least `mode_out_len` bytes
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_get_processing_mode(
    handle: GhostWaveHandle,
    mode_out: *mut c_char,
    mode_out_len: usize,
) -> CError {
    if handle.is_null() {
        return CError::from_error(&GhostWaveError::FfiInvalidHandle);
    }
    if mode_out.is_null() || mode_out_len == 0 {
        return CError::from_error(&GhostWaveError::FfiNullPointer {
            parameter: "mode_out".to_string(),
        });
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            let processor = handle.as_processor().ok_or(GhostWaveError::FfiInvalidHandle)?;
            let mode = processor.get_processing_mode();
            let bytes = mode.as_bytes();
            let len = bytes.len().min(mode_out_len - 1);

            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mode_out as *mut u8, len);
            *((mode_out as *mut u8).add(len)) = 0; // Null terminate

            Ok(())
        }
    }));

    match result {
        Ok(Ok(())) => CError::success(),
        Ok(Err(e)) => CError::from_error(&e),
        Err(_) => CError::from_error(&GhostWaveError::FfiPanic {
            message: "Panic getting processing mode".to_string(),
        }),
    }
}

/// Get library version string
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Check if RTX acceleration is available
#[unsafe(no_mangle)]
pub extern "C" fn ghostwave_has_rtx(handle: GhostWaveHandle) -> bool {
    if handle.is_null() {
        return false;
    }

    catch_unwind(AssertUnwindSafe(|| {
        unsafe { handle.as_processor().map(|p| p.has_rtx_acceleration()).unwrap_or(false) }
    }))
    .unwrap_or(false)
}

// ============================================================================
// Convenience Macros for C Header Generation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_create_destroy() {
        let mut handle = GhostWaveHandle::null();
        let err = ghostwave_create(&mut handle, 48000, 1, 256);
        assert_eq!(err.code, 0);
        assert!(!handle.is_null());

        ghostwave_destroy(handle);
    }

    #[test]
    fn test_ffi_invalid_params() {
        let mut handle = GhostWaveHandle::null();

        // Invalid sample rate
        let err = ghostwave_create(&mut handle, 12345, 1, 256);
        assert_ne!(err.code, 0);

        // Invalid buffer size
        let err = ghostwave_create(&mut handle, 48000, 1, 100); // Not power of 2
        assert_ne!(err.code, 0);
    }

    #[test]
    fn test_ffi_process() {
        let mut handle = GhostWaveHandle::null();
        let err = ghostwave_create(&mut handle, 48000, 1, 256);
        assert_eq!(err.code, 0);

        let input = vec![0.1f32; 256];
        let mut output = vec![0.0f32; 256];

        let err = ghostwave_process(handle, input.as_ptr(), output.as_mut_ptr(), 256);
        assert_eq!(err.code, 0);

        ghostwave_destroy(handle);
    }

    #[test]
    fn test_ffi_version() {
        let version = ghostwave_version();
        assert!(!version.is_null());

        unsafe {
            let version_str = CStr::from_ptr(version).to_str().unwrap();
            assert!(version_str.contains("0.2"));
        }
    }
}

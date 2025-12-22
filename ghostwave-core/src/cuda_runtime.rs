//! # Runtime CUDA Detection and Loading
//!
//! This module provides runtime CUDA detection and loading without requiring
//! compile-time feature flags. It uses `libloading` to dynamically load
//! CUDA libraries and detect GPU capabilities at runtime.
//!
//! ## Features
//! - Automatic CUDA library detection
//! - GPU enumeration and capability detection
//! - Runtime feature detection (FP4, FP16, Tensor Cores)
//! - Graceful fallback when CUDA is unavailable
//!
//! ## Usage
//! ```rust,no_run
//! use ghostwave_core::cuda_runtime::CudaRuntime;
//!
//! // Try to initialize CUDA at runtime
//! match CudaRuntime::new() {
//!     Ok(runtime) => {
//!         println!("CUDA available: {} GPUs", runtime.device_count());
//!         if let Some(info) = runtime.get_device_info(0) {
//!             println!("GPU: {}", info.name);
//!         }
//!     }
//!     Err(e) => {
//!         println!("CUDA not available: {}", e);
//!         // Continue with CPU fallback
//!     }
//! }
//! ```

use anyhow::{Result, anyhow};
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::{Arc, OnceLock};
use tracing::{info, debug, warn};

/// CUDA error type (matches CUresult)
type CuResult = c_int;

/// CUDA device type
type CuDevice = c_int;

/// CUDA context type
type CuContext = *mut c_void;

/// Global CUDA runtime singleton
static CUDA_RUNTIME: OnceLock<Option<Arc<CudaRuntime>>> = OnceLock::new();

/// GPU architecture generation detected at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeGpuArch {
    Turing,      // SM 7.5 - RTX 20 series
    Ampere,      // SM 8.0/8.6 - RTX 30 series
    AdaLovelace, // SM 8.9 - RTX 40 series
    Blackwell,   // SM 10.0/12.0 - RTX 50 series
    PreTuring,   // Older than RTX (Maxwell, Pascal, Volta)
    Unknown,
}

impl RuntimeGpuArch {
    pub fn from_compute_capability(major: i32, minor: i32) -> Self {
        match (major, minor) {
            (7, 5) => Self::Turing,
            (8, 0) | (8, 6) | (8, 7) => Self::Ampere,
            (8, 9) => Self::AdaLovelace,
            (10, _) | (12, _) => Self::Blackwell,
            _ if major >= 10 => Self::Blackwell,
            _ if major < 7 || (major == 7 && minor < 5) => Self::PreTuring,
            _ => Self::Unknown,
        }
    }

    pub fn supports_rtx(&self) -> bool {
        !matches!(self, Self::PreTuring | Self::Unknown)
    }

    pub fn supports_fp4(&self) -> bool {
        matches!(self, Self::Blackwell)
    }

    pub fn supports_fp8(&self) -> bool {
        matches!(self, Self::Blackwell | Self::AdaLovelace)
    }

    pub fn tensor_core_generation(&self) -> u8 {
        match self {
            Self::Turing => 2,
            Self::Ampere => 3,
            Self::AdaLovelace => 4,
            Self::Blackwell => 5,
            _ => 0,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Turing => "Turing (RTX 20)",
            Self::Ampere => "Ampere (RTX 30)",
            Self::AdaLovelace => "Ada Lovelace (RTX 40)",
            Self::Blackwell => "Blackwell (RTX 50)",
            Self::PreTuring => "Pre-RTX (Maxwell/Pascal/Volta)",
            Self::Unknown => "Unknown",
        }
    }
}

/// GPU device information detected at runtime
#[derive(Debug, Clone)]
pub struct RuntimeGpuInfo {
    pub index: usize,
    pub name: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub architecture: RuntimeGpuArch,
    pub memory_bytes: usize,
    pub memory_gb: f32,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub warp_size: i32,
    pub supports_rtx: bool,
    pub supports_tensor_cores: bool,
    pub supports_fp4: bool,
    pub tensor_core_gen: u8,
}

/// CUDA driver capabilities detected at runtime
#[derive(Debug, Clone)]
pub struct CudaCapabilities {
    pub driver_version: i32,
    pub runtime_version: i32,
    pub driver_version_string: String,
    pub supports_fp4: bool,
    pub supports_async_copy: bool,
    pub supports_graphs: bool,
}

/// Runtime CUDA library wrapper
#[allow(dead_code)] // Fields kept for library lifetime and future API extensions
pub struct CudaRuntime {
    /// Loaded libcuda.so - must be kept alive while function pointers are in use
    library: Library,

    /// Function pointers (loaded lazily)
    fn_init: Symbol<'static, unsafe extern "C" fn() -> CuResult>,
    fn_device_get_count: Symbol<'static, unsafe extern "C" fn(*mut c_int) -> CuResult>,
    fn_device_get: Symbol<'static, unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult>,
    fn_device_get_name: Symbol<'static, unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult>,
    fn_device_get_attribute: Symbol<'static, unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult>,
    fn_device_total_mem: Symbol<'static, unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult>,
    fn_ctx_create: Symbol<'static, unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult>,
    fn_ctx_destroy: Symbol<'static, unsafe extern "C" fn(CuContext) -> CuResult>,
    fn_driver_get_version: Symbol<'static, unsafe extern "C" fn(*mut c_int) -> CuResult>,

    /// Cached device information
    devices: Vec<RuntimeGpuInfo>,

    /// Driver capabilities
    capabilities: CudaCapabilities,
}

// CUDA device attributes we query
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;

// CUDA success return code
const CUDA_SUCCESS: CuResult = 0;

impl CudaRuntime {
    /// Attempt to load CUDA runtime dynamically
    pub fn new() -> Result<Self> {
        info!("Attempting runtime CUDA detection...");

        // Try to load libcuda.so from common locations
        let library = Self::load_cuda_library()?;

        // Load function pointers
        // Safety: These are the standard CUDA driver API functions
        let fn_init: Symbol<'static, unsafe extern "C" fn() -> CuResult> = unsafe {
            std::mem::transmute(library.get::<unsafe extern "C" fn() -> CuResult>(b"cuInit")?)
        };

        let fn_device_get_count: Symbol<'static, unsafe extern "C" fn(*mut c_int) -> CuResult> = unsafe {
            std::mem::transmute(library.get::<unsafe extern "C" fn(*mut c_int) -> CuResult>(b"cuDeviceGetCount")?)
        };

        let fn_device_get: Symbol<'static, unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult> = unsafe {
            std::mem::transmute(library.get::<unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult>(b"cuDeviceGet")?)
        };

        let fn_device_get_name: Symbol<'static, unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult> = unsafe {
            std::mem::transmute(library.get::<unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult>(b"cuDeviceGetName")?)
        };

        let fn_device_get_attribute: Symbol<'static, unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult> = unsafe {
            std::mem::transmute(library.get::<unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult>(b"cuDeviceGetAttribute")?)
        };

        let fn_device_total_mem: Symbol<'static, unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult> = unsafe {
            std::mem::transmute(library.get::<unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult>(b"cuDeviceTotalMem_v2")?)
        };

        let fn_ctx_create: Symbol<'static, unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult> = unsafe {
            std::mem::transmute(library.get::<unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult>(b"cuCtxCreate_v2")?)
        };

        let fn_ctx_destroy: Symbol<'static, unsafe extern "C" fn(CuContext) -> CuResult> = unsafe {
            std::mem::transmute(library.get::<unsafe extern "C" fn(CuContext) -> CuResult>(b"cuCtxDestroy_v2")?)
        };

        let fn_driver_get_version: Symbol<'static, unsafe extern "C" fn(*mut c_int) -> CuResult> = unsafe {
            std::mem::transmute(library.get::<unsafe extern "C" fn(*mut c_int) -> CuResult>(b"cuDriverGetVersion")?)
        };

        // Initialize CUDA
        let result = unsafe { fn_init() };
        if result != CUDA_SUCCESS {
            return Err(anyhow!("cuInit failed with error code {}", result));
        }

        // Get driver version
        let mut driver_version: c_int = 0;
        let result = unsafe { fn_driver_get_version(&mut driver_version) };
        if result != CUDA_SUCCESS {
            return Err(anyhow!("cuDriverGetVersion failed with error code {}", result));
        }

        let driver_major = driver_version / 1000;
        let driver_minor = (driver_version % 1000) / 10;
        let driver_version_string = format!("{}.{}", driver_major, driver_minor);

        info!("CUDA driver version: {}", driver_version_string);

        // Create runtime instance
        let mut runtime = Self {
            library,
            fn_init,
            fn_device_get_count,
            fn_device_get,
            fn_device_get_name,
            fn_device_get_attribute,
            fn_device_total_mem,
            fn_ctx_create,
            fn_ctx_destroy,
            fn_driver_get_version,
            devices: Vec::new(),
            capabilities: CudaCapabilities {
                driver_version,
                runtime_version: driver_version,
                driver_version_string: driver_version_string.clone(),
                supports_fp4: driver_major >= 12, // CUDA 12+ for FP4
                supports_async_copy: driver_major >= 11,
                supports_graphs: driver_major >= 10,
            },
        };

        // Enumerate devices
        runtime.enumerate_devices()?;

        if runtime.devices.is_empty() {
            return Err(anyhow!("No CUDA-capable GPUs found"));
        }

        for device in &runtime.devices {
            info!("Found GPU {}: {} ({}, {:.1} GB)",
                device.index,
                device.name,
                device.architecture.name(),
                device.memory_gb
            );

            if device.supports_fp4 {
                info!("  FP4 Tensor Core support available (5th-gen)");
            } else if device.supports_tensor_cores {
                info!("  Tensor Core generation: {}", device.tensor_core_gen);
            }
        }

        Ok(runtime)
    }

    /// Load CUDA library from common locations
    fn load_cuda_library() -> Result<Library> {
        let search_paths = [
            "libcuda.so.1",
            "libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib64/libcuda.so.1",
            "/usr/lib/libcuda.so.1",
            "/opt/cuda/lib64/libcuda.so.1",
            "/usr/local/cuda/lib64/libcuda.so.1",
        ];

        for path in &search_paths {
            debug!("Trying to load CUDA from: {}", path);
            match unsafe { Library::new(path) } {
                Ok(lib) => {
                    info!("Loaded CUDA library from: {}", path);
                    return Ok(lib);
                }
                Err(e) => {
                    debug!("Failed to load from {}: {}", path, e);
                }
            }
        }

        Err(anyhow!("Could not find libcuda.so - is the NVIDIA driver installed?"))
    }

    /// Enumerate all CUDA devices
    fn enumerate_devices(&mut self) -> Result<()> {
        let mut device_count: c_int = 0;
        let result = unsafe { (self.fn_device_get_count)(&mut device_count) };
        if result != CUDA_SUCCESS {
            return Err(anyhow!("cuDeviceGetCount failed with error code {}", result));
        }

        info!("Found {} CUDA device(s)", device_count);

        for i in 0..device_count {
            if let Ok(info) = self.get_device_info_internal(i as usize) {
                self.devices.push(info);
            }
        }

        Ok(())
    }

    /// Get device information for a specific device index
    fn get_device_info_internal(&self, index: usize) -> Result<RuntimeGpuInfo> {
        let mut device: CuDevice = 0;
        let result = unsafe { (self.fn_device_get)(&mut device, index as c_int) };
        if result != CUDA_SUCCESS {
            return Err(anyhow!("cuDeviceGet failed for device {}", index));
        }

        // Get device name
        let mut name_buf = [0i8; 256];
        let result = unsafe { (self.fn_device_get_name)(name_buf.as_mut_ptr(), 256, device) };
        if result != CUDA_SUCCESS {
            return Err(anyhow!("cuDeviceGetName failed"));
        }
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .to_string();

        // Get compute capability
        let mut compute_major: c_int = 0;
        let mut compute_minor: c_int = 0;
        unsafe {
            (self.fn_device_get_attribute)(&mut compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
            (self.fn_device_get_attribute)(&mut compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        }

        // Get other attributes
        let mut mp_count: c_int = 0;
        let mut max_threads: c_int = 0;
        let mut warp_size: c_int = 0;
        unsafe {
            (self.fn_device_get_attribute)(&mut mp_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            (self.fn_device_get_attribute)(&mut max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
            (self.fn_device_get_attribute)(&mut warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
        }

        // Get memory
        let mut memory_bytes: usize = 0;
        unsafe { (self.fn_device_total_mem)(&mut memory_bytes, device) };

        let architecture = RuntimeGpuArch::from_compute_capability(compute_major, compute_minor);
        let supports_tensor_cores = architecture.tensor_core_generation() > 0;
        let supports_fp4 = architecture.supports_fp4() && self.capabilities.supports_fp4;

        Ok(RuntimeGpuInfo {
            index,
            name,
            compute_major,
            compute_minor,
            architecture,
            memory_bytes,
            memory_gb: memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0),
            multiprocessor_count: mp_count,
            max_threads_per_block: max_threads,
            warp_size,
            supports_rtx: architecture.supports_rtx(),
            supports_tensor_cores,
            supports_fp4,
            tensor_core_gen: architecture.tensor_core_generation(),
        })
    }

    /// Get the global CUDA runtime instance (singleton)
    pub fn global() -> Option<Arc<CudaRuntime>> {
        CUDA_RUNTIME.get_or_init(|| {
            match CudaRuntime::new() {
                Ok(runtime) => Some(Arc::new(runtime)),
                Err(e) => {
                    warn!("CUDA runtime initialization failed: {}", e);
                    None
                }
            }
        }).clone()
    }

    /// Check if CUDA is available at runtime
    pub fn is_available() -> bool {
        Self::global().is_some()
    }

    /// Get number of available devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get device information by index
    pub fn get_device_info(&self, index: usize) -> Option<&RuntimeGpuInfo> {
        self.devices.get(index)
    }

    /// Get the best device (highest compute capability)
    pub fn best_device(&self) -> Option<&RuntimeGpuInfo> {
        self.devices.iter().max_by(|a, b| {
            let cap_a = (a.compute_major, a.compute_minor, a.memory_bytes);
            let cap_b = (b.compute_major, b.compute_minor, b.memory_bytes);
            cap_a.cmp(&cap_b)
        })
    }

    /// Get all RTX-capable devices
    pub fn rtx_devices(&self) -> Vec<&RuntimeGpuInfo> {
        self.devices.iter().filter(|d| d.supports_rtx).collect()
    }

    /// Check if any device supports FP4
    pub fn has_fp4_support(&self) -> bool {
        self.devices.iter().any(|d| d.supports_fp4)
    }

    /// Check if any device supports RTX features
    pub fn has_rtx_support(&self) -> bool {
        self.devices.iter().any(|d| d.supports_rtx)
    }

    /// Get CUDA capabilities
    pub fn capabilities(&self) -> &CudaCapabilities {
        &self.capabilities
    }

    /// Create a CUDA context on a device (for advanced use)
    pub fn create_context(&self, device_index: usize) -> Result<CudaContext> {
        if device_index >= self.devices.len() {
            return Err(anyhow!("Invalid device index"));
        }

        let mut device: CuDevice = 0;
        let result = unsafe { (self.fn_device_get)(&mut device, device_index as c_int) };
        if result != CUDA_SUCCESS {
            return Err(anyhow!("Failed to get device"));
        }

        let mut ctx: CuContext = std::ptr::null_mut();
        let result = unsafe { (self.fn_ctx_create)(&mut ctx, 0, device) };
        if result != CUDA_SUCCESS {
            return Err(anyhow!("Failed to create CUDA context"));
        }

        Ok(CudaContext {
            ctx,
            fn_destroy: self.fn_ctx_destroy.clone(),
        })
    }
}

/// CUDA context wrapper with RAII cleanup
pub struct CudaContext {
    ctx: CuContext,
    fn_destroy: Symbol<'static, unsafe extern "C" fn(CuContext) -> CuResult>,
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { (self.fn_destroy)(self.ctx) };
        }
    }
}

/// Runtime GPU information for use without compile-time CUDA dependency
pub fn detect_gpus() -> Vec<RuntimeGpuInfo> {
    match CudaRuntime::global() {
        Some(runtime) => runtime.devices.clone(),
        None => Vec::new(),
    }
}

/// Check if RTX acceleration is available at runtime
pub fn is_rtx_available_runtime() -> bool {
    match CudaRuntime::global() {
        Some(runtime) => runtime.has_rtx_support(),
        None => false,
    }
}

/// Check if FP4 tensor cores are available at runtime
pub fn is_fp4_available_runtime() -> bool {
    match CudaRuntime::global() {
        Some(runtime) => runtime.has_fp4_support(),
        None => false,
    }
}

/// Get the best RTX GPU name
pub fn best_gpu_name() -> Option<String> {
    CudaRuntime::global()
        .and_then(|rt| rt.best_device().map(|d| d.name.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_arch_detection() {
        assert_eq!(RuntimeGpuArch::from_compute_capability(7, 5), RuntimeGpuArch::Turing);
        assert_eq!(RuntimeGpuArch::from_compute_capability(8, 6), RuntimeGpuArch::Ampere);
        assert_eq!(RuntimeGpuArch::from_compute_capability(8, 9), RuntimeGpuArch::AdaLovelace);
        assert_eq!(RuntimeGpuArch::from_compute_capability(12, 0), RuntimeGpuArch::Blackwell);
        assert_eq!(RuntimeGpuArch::from_compute_capability(10, 0), RuntimeGpuArch::Blackwell);

        assert!(RuntimeGpuArch::Blackwell.supports_fp4());
        assert!(!RuntimeGpuArch::AdaLovelace.supports_fp4());
        assert!(RuntimeGpuArch::AdaLovelace.supports_fp8());
    }

    #[test]
    fn test_tensor_core_gen() {
        assert_eq!(RuntimeGpuArch::Turing.tensor_core_generation(), 2);
        assert_eq!(RuntimeGpuArch::Ampere.tensor_core_generation(), 3);
        assert_eq!(RuntimeGpuArch::AdaLovelace.tensor_core_generation(), 4);
        assert_eq!(RuntimeGpuArch::Blackwell.tensor_core_generation(), 5);
    }

    #[test]
    fn test_runtime_detection() {
        // This test will pass even without CUDA installed
        let available = CudaRuntime::is_available();
        println!("CUDA available at runtime: {}", available);

        if available {
            let gpus = detect_gpus();
            println!("Found {} GPU(s)", gpus.len());
            for gpu in &gpus {
                println!("  {}: {} (SM {}.{})", gpu.index, gpu.name, gpu.compute_major, gpu.compute_minor);
            }
        }
    }
}

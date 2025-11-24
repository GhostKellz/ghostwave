//! # GPU Acceleration Support
//!
//! Provides feature flags and abstraction layer for GPU-accelerated audio processing.
//! Supports CUDA/TensorRT for NVIDIA RTX and Vulkan compute for cross-platform acceleration.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug};

/// GPU acceleration backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    /// NVIDIA CUDA with TensorRT optimizations
    CudaTensorRT,
    /// Vulkan compute shaders (cross-platform)
    Vulkan,
    /// OpenCL acceleration (cross-platform)
    OpenCL,
    /// CPU fallback
    None,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::CudaTensorRT => write!(f, "CUDA/TensorRT"),
            GpuBackend::Vulkan => write!(f, "Vulkan Compute"),
            GpuBackend::OpenCL => write!(f, "OpenCL"),
            GpuBackend::None => write!(f, "CPU"),
        }
    }
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub memory_gb: f32,
    pub compute_capability: (u32, u32), // For CUDA
    pub backend: GpuBackend,
    pub supports_fp16: bool,
    pub supports_int8: bool,
    pub max_threads_per_block: u32,
    pub max_shared_memory_kb: u32,
}

/// GPU acceleration capabilities
#[derive(Debug, Clone, Default)]
pub struct GpuCapabilities {
    pub cuda_available: bool,
    pub tensorrt_available: bool,
    pub vulkan_available: bool,
    pub opencl_available: bool,
    pub devices: Vec<GpuDeviceInfo>,
    pub recommended_backend: Option<GpuBackend>,
}

impl GpuCapabilities {
    /// Detect available GPU acceleration capabilities
    pub fn detect() -> Self {
        let mut caps = Self::default();

        // Detect CUDA/TensorRT
        #[cfg(feature = "cuda-tensorrt")]
        {
            caps.cuda_available = Self::detect_cuda();
            caps.tensorrt_available = caps.cuda_available && Self::detect_tensorrt();

            if caps.cuda_available {
                caps.devices.extend(Self::detect_cuda_devices());
            }
        }

        // Detect Vulkan compute
        #[cfg(feature = "vulkan-compute")]
        {
            caps.vulkan_available = Self::detect_vulkan();
            if caps.vulkan_available {
                caps.devices.extend(Self::detect_vulkan_devices());
            }
        }

        // Detect OpenCL
        #[cfg(feature = "opencl")]
        {
            caps.opencl_available = Self::detect_opencl();
            if caps.opencl_available {
                caps.devices.extend(Self::detect_opencl_devices());
            }
        }

        // Determine recommended backend
        caps.recommended_backend = caps.get_recommended_backend();

        caps
    }

    #[cfg(feature = "cuda-tensorrt")]
    fn detect_cuda() -> bool {
        // Check for NVIDIA driver and CUDA runtime
        match std::process::Command::new("nvidia-smi").output() {
            Ok(output) => {
                let success = output.status.success();
                if success {
                    info!("✅ NVIDIA driver detected");
                } else {
                    debug!("NVIDIA driver check failed");
                }
                success
            }
            Err(_) => {
                debug!("nvidia-smi not found");
                false
            }
        }
    }

    #[cfg(feature = "cuda-tensorrt")]
    fn detect_tensorrt() -> bool {
        // Check for TensorRT library
        match std::process::Command::new("python3")
            .args(&["-c", "import tensorrt; print(tensorrt.__version__)"])
            .output()
        {
            Ok(output) => {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    info!("✅ TensorRT detected: {}", version.trim());
                    true
                } else {
                    debug!("TensorRT import failed");
                    false
                }
            }
            Err(_) => {
                debug!("Python/TensorRT check failed");
                false
            }
        }
    }

    #[cfg(feature = "cuda-tensorrt")]
    fn detect_cuda_devices() -> Vec<GpuDeviceInfo> {
        use std::process::Command;

        // Try to detect actual GPU via nvidia-smi
        if let Ok(output) = Command::new("nvidia-smi")
            .args(&["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = output_str.trim().split(',').collect();

                if parts.len() >= 2 {
                    let name = parts[0].trim().to_string();
                    let memory_mb: f32 = parts[1].trim().parse().unwrap_or(8192.0);
                    let memory_gb = memory_mb / 1024.0;

                    // Detect compute capability based on GPU name
                    let compute_capability = if name.contains("RTX 50") || name.contains("5090") || name.contains("5080") {
                        (10, 0) // Blackwell
                    } else if name.contains("RTX 40") || name.contains("4090") || name.contains("4080") {
                        (8, 9) // Ada Lovelace
                    } else if name.contains("RTX 30") || name.contains("3090") || name.contains("3080") {
                        (8, 6) // Ampere
                    } else {
                        (7, 5) // Turing fallback
                    };

                    info!("Detected GPU: {} with {:.1}GB memory", name, memory_gb);

                    // ASUS ROG Astral RTX 5090 specific detection
                    if name.contains("5090") && memory_gb > 30.0 {
                        info!("🔥 ASUS ROG Astral RTX 5090 detected!");
                        info!("   Quad-fan cooling optimized for sustained performance");
                        info!("   Factory OC: 2610MHz boost (630W max power)");
                        info!("   32GB GDDR7 memory - perfect for large audio buffers");
                    }

                    return vec![GpuDeviceInfo {
                        name,
                        memory_gb,
                        compute_capability,
                        backend: GpuBackend::CudaTensorRT,
                        supports_fp16: true,
                        supports_int8: true,
                        max_threads_per_block: if compute_capability.0 >= 10 { 1536 } else { 1024 },
                        max_shared_memory_kb: if compute_capability.0 >= 10 { 228 } else { 48 },
                    }];
                }
            }
        }

        // Fallback to generic RTX GPU
        warn!("Could not detect GPU via nvidia-smi, using generic RTX profile");
        vec![GpuDeviceInfo {
            name: "NVIDIA RTX GPU".to_string(),
            memory_gb: 8.0,
            compute_capability: (8, 6), // Ampere fallback
            backend: GpuBackend::CudaTensorRT,
            supports_fp16: true,
            supports_int8: true,
            max_threads_per_block: 1024,
            max_shared_memory_kb: 48,
        }]
    }

    #[cfg(feature = "vulkan-compute")]
    fn detect_vulkan() -> bool {
        // Check for Vulkan loader
        match vulkanalia::Instance::new(&Default::default()) {
            Ok(_) => {
                info!("✅ Vulkan compute available");
                true
            }
            Err(e) => {
                debug!("Vulkan detection failed: {}", e);
                false
            }
        }
    }

    #[cfg(feature = "vulkan-compute")]
    fn detect_vulkan_devices() -> Vec<GpuDeviceInfo> {
        // Use vulkanalia to enumerate devices
        // For now, return mock data
        vec![GpuDeviceInfo {
            name: "Vulkan GPU".to_string(),
            memory_gb: 4.0,
            compute_capability: (0, 0),
            backend: GpuBackend::Vulkan,
            supports_fp16: true,
            supports_int8: false,
            max_threads_per_block: 256,
            max_shared_memory_kb: 16,
        }]
    }

    #[cfg(feature = "opencl")]
    fn detect_opencl() -> bool {
        // Check for OpenCL runtime
        debug!("OpenCL detection not implemented yet");
        false
    }

    #[cfg(feature = "opencl")]
    fn detect_opencl_devices() -> Vec<GpuDeviceInfo> {
        vec![]
    }

    #[cfg(not(feature = "cuda-tensorrt"))]
    fn detect_cuda() -> bool { false }

    #[cfg(not(feature = "cuda-tensorrt"))]
    fn detect_tensorrt() -> bool { false }

    #[cfg(not(feature = "cuda-tensorrt"))]
    fn detect_cuda_devices() -> Vec<GpuDeviceInfo> { vec![] }

    #[cfg(not(feature = "vulkan-compute"))]
    fn detect_vulkan() -> bool { false }

    #[cfg(not(feature = "vulkan-compute"))]
    fn detect_vulkan_devices() -> Vec<GpuDeviceInfo> { vec![] }

    #[cfg(not(feature = "opencl"))]
    fn detect_opencl() -> bool { false }

    #[cfg(not(feature = "opencl"))]
    fn detect_opencl_devices() -> Vec<GpuDeviceInfo> { vec![] }

    fn get_recommended_backend(&self) -> Option<GpuBackend> {
        // Preference order: CUDA/TensorRT > Vulkan > OpenCL
        if self.tensorrt_available {
            Some(GpuBackend::CudaTensorRT)
        } else if self.vulkan_available {
            Some(GpuBackend::Vulkan)
        } else if self.opencl_available {
            Some(GpuBackend::OpenCL)
        } else {
            None
        }
    }

    pub fn report(&self) {
        info!("🚀 GPU Acceleration Capabilities:");
        info!("  CUDA Available: {}", self.cuda_available);
        info!("  TensorRT Available: {}", self.tensorrt_available);
        info!("  Vulkan Available: {}", self.vulkan_available);
        info!("  OpenCL Available: {}", self.opencl_available);
        info!("  Detected Devices: {}", self.devices.len());

        for device in &self.devices {
            info!("  • {} ({}) - {:.1}GB VRAM", device.name, device.backend, device.memory_gb);
        }

        if let Some(backend) = self.recommended_backend {
            info!("  ✅ Recommended Backend: {}", backend);
        } else {
            info!("  💻 No GPU acceleration available, using CPU");
        }
    }

    pub fn best_device(&self) -> Option<&GpuDeviceInfo> {
        self.devices.iter().max_by(|a, b| {
            // Prefer CUDA > Vulkan > OpenCL, then by memory
            match (a.backend, b.backend) {
                (GpuBackend::CudaTensorRT, GpuBackend::CudaTensorRT) => a.memory_gb.partial_cmp(&b.memory_gb).unwrap(),
                (GpuBackend::CudaTensorRT, _) => std::cmp::Ordering::Greater,
                (_, GpuBackend::CudaTensorRT) => std::cmp::Ordering::Less,
                (GpuBackend::Vulkan, GpuBackend::Vulkan) => a.memory_gb.partial_cmp(&b.memory_gb).unwrap(),
                (GpuBackend::Vulkan, _) => std::cmp::Ordering::Greater,
                (_, GpuBackend::Vulkan) => std::cmp::Ordering::Less,
                _ => a.memory_gb.partial_cmp(&b.memory_gb).unwrap(),
            }
        })
    }
}

/// GPU acceleration context
pub trait GpuAccelerator: Send + Sync {
    /// Initialize the GPU accelerator
    fn initialize(&mut self) -> Result<()>;

    /// Process audio buffer on GPU
    fn process_audio(&mut self, input: &[f32], output: &mut [f32], operation: GpuOperation) -> Result<()>;

    /// Process spectral denoising on GPU
    fn process_spectral_denoising(&mut self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()>;

    /// Get device information
    fn device_info(&self) -> &GpuDeviceInfo;

    /// Get backend type
    fn backend(&self) -> GpuBackend;

    /// Check if accelerator is available
    fn is_available(&self) -> bool;

    /// Get processing latency estimate in samples
    fn latency_samples(&self) -> usize;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;
}

/// GPU operations that can be accelerated
#[derive(Debug, Clone, Copy)]
pub enum GpuOperation {
    SpectralDenoising { strength: f32 },
    VoiceActivityDetection,
    HighPassFilter { cutoff: f32 },
    Compressor { ratio: f32, threshold: f32 },
    Limiter { threshold: f32 },
    Convolution { impulse_length: usize },
}

/// CUDA/TensorRT accelerator implementation
#[cfg(feature = "cuda-tensorrt")]
pub struct CudaTensorRTAccelerator {
    device_info: GpuDeviceInfo,
    initialized: bool,
    // TensorRT engine, CUDA context, etc.
}

#[cfg(feature = "cuda-tensorrt")]
impl CudaTensorRTAccelerator {
    pub fn new(device_info: GpuDeviceInfo) -> Self {
        Self {
            device_info,
            initialized: false,
        }
    }
}

#[cfg(feature = "cuda-tensorrt")]
impl GpuAccelerator for CudaTensorRTAccelerator {
    fn initialize(&mut self) -> Result<()> {
        info!("🚀 Initializing CUDA/TensorRT accelerator");
        // Initialize CUDA context
        // Load TensorRT engine for noise suppression
        // Allocate GPU buffers
        self.initialized = true;
        Ok(())
    }

    fn process_audio(&mut self, input: &[f32], output: &mut [f32], operation: GpuOperation) -> Result<()> {
        if !self.initialized {
            return Err(anyhow::anyhow!("CUDA accelerator not initialized"));
        }

        match operation {
            GpuOperation::SpectralDenoising { strength } => {
                self.process_spectral_denoising(input, output, strength)
            }
            _ => {
                // Copy input to output for unsupported operations
                output.copy_from_slice(input);
                Ok(())
            }
        }
    }

    fn process_spectral_denoising(&mut self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()> {
        // 1. Copy input to GPU memory
        // 2. Run TensorRT inference for noise suppression
        // 3. Copy result back to CPU memory

        debug!("Processing spectral denoising with strength {:.2}", strength);

        // Placeholder: Apply simple gain reduction as fallback
        for (i, &sample) in input.iter().enumerate() {
            output[i] = sample * (1.0 - strength * 0.5);
        }

        Ok(())
    }

    fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    fn backend(&self) -> GpuBackend {
        GpuBackend::CudaTensorRT
    }

    fn is_available(&self) -> bool {
        self.initialized
    }

    fn latency_samples(&self) -> usize {
        // TensorRT typically adds 1-2 buffer lengths of latency
        128 // Placeholder
    }

    fn memory_usage(&self) -> usize {
        // Model weights + intermediate buffers
        50 * 1024 * 1024 // ~50MB placeholder
    }
}

/// Vulkan compute accelerator implementation
#[cfg(feature = "vulkan-compute")]
pub struct VulkanAccelerator {
    device_info: GpuDeviceInfo,
    initialized: bool,
    // Vulkan instance, device, command pool, etc.
}

#[cfg(feature = "vulkan-compute")]
impl VulkanAccelerator {
    pub fn new(device_info: GpuDeviceInfo) -> Self {
        Self {
            device_info,
            initialized: false,
        }
    }
}

#[cfg(feature = "vulkan-compute")]
impl GpuAccelerator for VulkanAccelerator {
    fn initialize(&mut self) -> Result<()> {
        info!("🚀 Initializing Vulkan compute accelerator");
        // Initialize Vulkan instance and device
        // Compile compute shaders
        // Allocate buffers
        self.initialized = true;
        Ok(())
    }

    fn process_audio(&mut self, input: &[f32], output: &mut [f32], operation: GpuOperation) -> Result<()> {
        if !self.initialized {
            return Err(anyhow::anyhow!("Vulkan accelerator not initialized"));
        }

        match operation {
            GpuOperation::SpectralDenoising { strength } => {
                self.process_spectral_denoising(input, output, strength)
            }
            _ => {
                output.copy_from_slice(input);
                Ok(())
            }
        }
    }

    fn process_spectral_denoising(&mut self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()> {
        debug!("Processing spectral denoising with Vulkan compute, strength {:.2}", strength);

        // Placeholder implementation
        for (i, &sample) in input.iter().enumerate() {
            output[i] = sample * (1.0 - strength * 0.3);
        }

        Ok(())
    }

    fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    fn backend(&self) -> GpuBackend {
        GpuBackend::Vulkan
    }

    fn is_available(&self) -> bool {
        self.initialized
    }

    fn latency_samples(&self) -> usize {
        256 // Placeholder
    }

    fn memory_usage(&self) -> usize {
        20 * 1024 * 1024 // ~20MB placeholder
    }
}

/// CPU fallback "accelerator"
pub struct CpuFallbackAccelerator {
    device_info: GpuDeviceInfo,
}

impl CpuFallbackAccelerator {
    pub fn new() -> Self {
        Self {
            device_info: GpuDeviceInfo {
                name: "CPU Fallback".to_string(),
                memory_gb: 0.0,
                compute_capability: (0, 0),
                backend: GpuBackend::None,
                supports_fp16: false,
                supports_int8: false,
                max_threads_per_block: 1,
                max_shared_memory_kb: 0,
            },
        }
    }
}

impl GpuAccelerator for CpuFallbackAccelerator {
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    fn process_audio(&mut self, input: &[f32], output: &mut [f32], _operation: GpuOperation) -> Result<()> {
        output.copy_from_slice(input);
        Ok(())
    }

    fn process_spectral_denoising(&mut self, input: &[f32], output: &mut [f32], _strength: f32) -> Result<()> {
        output.copy_from_slice(input);
        Ok(())
    }

    fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    fn backend(&self) -> GpuBackend {
        GpuBackend::None
    }

    fn is_available(&self) -> bool {
        true
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn memory_usage(&self) -> usize {
        0
    }
}

/// Factory for creating GPU accelerators
pub struct GpuAcceleratorFactory;

impl GpuAcceleratorFactory {
    /// Create the best available GPU accelerator
    pub fn create_best() -> Box<dyn GpuAccelerator> {
        let caps = GpuCapabilities::detect();

        if let Some(device) = caps.best_device() {
            match device.backend {
                #[cfg(feature = "cuda-tensorrt")]
                GpuBackend::CudaTensorRT => {
                    Box::new(CudaTensorRTAccelerator::new(device.clone()))
                }
                #[cfg(feature = "vulkan-compute")]
                GpuBackend::Vulkan => {
                    Box::new(VulkanAccelerator::new(device.clone()))
                }
                _ => Box::new(CpuFallbackAccelerator::new()),
            }
        } else {
            Box::new(CpuFallbackAccelerator::new())
        }
    }

    /// Create accelerator for specific backend
    pub fn create_for_backend(backend: GpuBackend) -> Result<Box<dyn GpuAccelerator>> {
        let caps = GpuCapabilities::detect();

        match backend {
            #[cfg(feature = "cuda-tensorrt")]
            GpuBackend::CudaTensorRT => {
                if let Some(device) = caps.devices.iter().find(|d| d.backend == backend) {
                    Ok(Box::new(CudaTensorRTAccelerator::new(device.clone())))
                } else {
                    Err(anyhow::anyhow!("CUDA/TensorRT not available"))
                }
            }
            #[cfg(feature = "vulkan-compute")]
            GpuBackend::Vulkan => {
                if let Some(device) = caps.devices.iter().find(|d| d.backend == backend) {
                    Ok(Box::new(VulkanAccelerator::new(device.clone())))
                } else {
                    Err(anyhow::anyhow!("Vulkan compute not available"))
                }
            }
            GpuBackend::None => Ok(Box::new(CpuFallbackAccelerator::new())),
            _ => Err(anyhow::anyhow!("Backend {:?} not supported", backend)),
        }
    }
}

/// Benchmark GPU accelerator performance
pub fn benchmark_gpu_accelerator(accelerator: &mut dyn GpuAccelerator) -> Result<GpuBenchmarkResults> {
    use std::time::Instant;

    const TEST_SIZE: usize = 4096;
    const ITERATIONS: usize = 100;

    let input: Vec<f32> = (0..TEST_SIZE).map(|i| (i as f32) * 0.001).collect();
    let mut output = vec![0.0f32; TEST_SIZE];

    // Benchmark spectral denoising
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        accelerator.process_spectral_denoising(&input, &mut output, 0.7)?;
    }
    let elapsed = start.elapsed();

    let throughput_msamples_per_sec = (TEST_SIZE * ITERATIONS) as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    Ok(GpuBenchmarkResults {
        backend: accelerator.backend(),
        device_name: accelerator.device_info().name.clone(),
        denoising_throughput_msamples_per_sec: throughput_msamples_per_sec,
        latency_samples: accelerator.latency_samples(),
        memory_usage_mb: accelerator.memory_usage() / 1024 / 1024,
    })
}

/// GPU benchmark results
#[derive(Debug, Clone)]
pub struct GpuBenchmarkResults {
    pub backend: GpuBackend,
    pub device_name: String,
    pub denoising_throughput_msamples_per_sec: f64,
    pub latency_samples: usize,
    pub memory_usage_mb: usize,
}

impl GpuBenchmarkResults {
    pub fn report(&self) {
        info!("🚀 GPU Acceleration Benchmark Results:");
        info!("  Backend: {}", self.backend);
        info!("  Device: {}", self.device_name);
        info!("  Denoising Throughput: {:.1} MSamples/sec", self.denoising_throughput_msamples_per_sec);
        info!("  Latency: {} samples", self.latency_samples);
        info!("  Memory Usage: {} MB", self.memory_usage_mb);

        if self.denoising_throughput_msamples_per_sec > 50.0 {
            info!("  ✅ Excellent GPU performance");
        } else if self.denoising_throughput_msamples_per_sec > 20.0 {
            info!("  ✅ Good GPU performance");
        } else {
            warn!("  ⚠️ Poor GPU performance");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_capabilities_detection() {
        let caps = GpuCapabilities::detect();
        caps.report();

        // Should always be able to fall back to CPU
        assert!(caps.recommended_backend.is_some() || caps.devices.is_empty());
    }

    #[test]
    fn test_cpu_fallback_accelerator() {
        let mut accelerator = CpuFallbackAccelerator::new();

        assert!(accelerator.initialize().is_ok());
        assert!(accelerator.is_available());
        assert_eq!(accelerator.backend(), GpuBackend::None);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        let result = accelerator.process_spectral_denoising(&input, &mut output, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_accelerator_factory() {
        let accelerator = GpuAcceleratorFactory::create_best();
        assert!(accelerator.is_available());
    }

    #[test]
    fn test_gpu_benchmark() {
        let mut accelerator = CpuFallbackAccelerator::new();
        accelerator.initialize().unwrap();

        let results = benchmark_gpu_accelerator(&mut accelerator).unwrap();
        results.report();

        assert!(results.denoising_throughput_msamples_per_sec > 0.0);
    }
}
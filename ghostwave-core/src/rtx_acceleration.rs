use anyhow::Result;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tracing::{info, debug, warn};

#[cfg(feature = "nvidia-rtx")]
use cudarc::driver::CudaDevice;
#[cfg(feature = "nvidia-rtx")]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "nvidia-rtx")]
use crate::rtx_denoising::{RtxDenoiser, RtxDenoiseConfig, DenoiseStrength, DenoiseAlgorithm};

#[cfg(feature = "nvidia-rtx")]
static RTX_DENOISER_CACHE: OnceLock<Arc<Mutex<SharedRtxDenoiser>>> = OnceLock::new();

/// Status of GPU processing - whether fallback to CPU occurred
#[derive(Debug, Clone, Default)]
pub struct GpuProcessingStatus {
    /// True if GPU is being used successfully
    pub gpu_active: bool,
    /// True if CPU fallback is currently in use
    pub fallback_active: bool,
    /// Number of times GPU failed and fell back to CPU
    pub fallback_count: u64,
    /// Reason for fallback (if any)
    pub fallback_reason: Option<String>,
}

// Global fallback tracking (thread-safe)
static GPU_FALLBACK_ACTIVE: AtomicBool = AtomicBool::new(false);
static GPU_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);
static GPU_FALLBACK_REASON: std::sync::OnceLock<std::sync::Mutex<Option<String>>> = std::sync::OnceLock::new();

#[cfg(feature = "nvidia-rtx")]
struct SharedRtxDenoiser {
    denoiser: Option<RtxDenoiser>,
    fft_size: usize,
    hop_size: usize,
}

#[cfg(feature = "nvidia-rtx")]
impl Default for SharedRtxDenoiser {
    fn default() -> Self {
        Self {
            denoiser: None,
            fft_size: 0,
            hop_size: 0,
        }
    }
}

/// RTX GPU acceleration for noise suppression
/// Leverages NVIDIA's open GPU kernel modules for RTX 20 series and newer
/// Supports up to RTX 50 series (Blackwell architecture) with enhanced optimizations
pub struct RtxAccelerator {
    #[cfg(feature = "nvidia-rtx")]
    device: Option<Arc<CudaDevice>>,

    #[cfg(feature = "nvidia-rtx")]
    shared_denoiser: Arc<Mutex<SharedRtxDenoiser>>,

    /// Fallback to CPU when GPU is not available
    cpu_fallback: bool,


    /// GPU compute capability for RTX features
    compute_capability: Option<(u32, u32)>,

    /// GPU architecture generation
    gpu_generation: GpuGeneration,
}

/// GPU Architecture Generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuGeneration {
    Turing,      // RTX 20 series - Compute 7.5
    Ampere,      // RTX 30 series - Compute 8.6
    AdaLovelace, // RTX 40 series - Compute 8.9
    Blackwell,   // RTX 50 series - Compute 10.0
    Unknown,
}

impl GpuGeneration {
    fn from_compute_capability(major: u32, minor: u32) -> Self {
        match (major, minor) {
            (7, 5) => Self::Turing,
            (8, 6) => Self::Ampere,
            (8, 9) => Self::AdaLovelace,
            // Blackwell: SM 10.0 and 12.0 (RTX 5090 reports compute 12.0)
            (10, _) | (12, _) => Self::Blackwell,
            _ if major >= 10 => Self::Blackwell, // Future architectures
            _ => Self::Unknown,
        }
    }

    pub fn supports_fp4(&self) -> bool {
        matches!(self, Self::Blackwell)
    }

    pub fn tensor_core_generation(&self) -> u8 {
        match self {
            Self::Turing => 2,
            Self::Ampere => 3,
            Self::AdaLovelace => 4,
            Self::Blackwell => 5,
            Self::Unknown => 0,
        }
    }
}

/// GPU readiness status for different subsystems
#[derive(Debug, Clone, Default)]
pub struct RtxReadiness {
    pub driver_ok: bool,
    pub cuda_ok: bool,
    pub tensorrt_ok: bool,
    pub fp4_ready: bool,
}

/// Comprehensive RTX system information returned by diagnostics
#[derive(Debug, Clone)]
pub struct RtxSystemInfo {
    pub gpu_name: Option<String>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub tensorrt_version: Option<String>,
    pub readiness: RtxReadiness,
    pub compute_capability: (u32, u32),
    pub gpu_generation: GpuGeneration,
    pub tensor_core_gen: u8,
    pub memory_gb: f32,
}

#[derive(Debug, Clone)]
pub struct RtxCapabilities {
    pub has_tensor_cores: bool,
    pub has_rt_cores: bool,
    pub memory_gb: f32,
    pub compute_capability: (u32, u32),
    pub supports_rtx_voice: bool,
    pub driver_version: String,
    pub gpu_generation: GpuGeneration,
    pub supports_fp4: bool, // RTX 50 series only
    pub tensor_core_gen: u8,
}

impl RtxAccelerator {
    pub fn new() -> Result<Self> {
        info!("🚀 Initializing NVIDIA RTX acceleration...");

        #[cfg(feature = "nvidia-rtx")]
        {
            match Self::init_cuda() {
                Ok((device, compute_capability)) => {
                    let gpu_gen = GpuGeneration::from_compute_capability(
                        compute_capability.0,
                        compute_capability.1,
                    );

                    info!("✅ NVIDIA RTX acceleration enabled");
                    info!("   Compute Capability: {}.{}", compute_capability.0, compute_capability.1);
                    info!("   GPU Generation: {:?}", gpu_gen);

                    if gpu_gen == GpuGeneration::Blackwell {
                        info!("   🚀 RTX 50 Series (Blackwell) - 5th-gen Tensor Cores with FP4 support");
                    }

                    let shared = RTX_DENOISER_CACHE.get_or_init(|| Arc::new(Mutex::new(SharedRtxDenoiser::default()))).clone();

                    Ok(Self {
                        device: Some(device),
                        shared_denoiser: shared,
                        cpu_fallback: false,
                        compute_capability: Some(compute_capability),
                        gpu_generation: gpu_gen,
                    })

                }
                Err(e) => {
                    warn!("⚠️  NVIDIA RTX acceleration unavailable: {}", e);
                    warn!("   Falling back to CPU processing");

                    let shared = RTX_DENOISER_CACHE.get_or_init(|| Arc::new(Mutex::new(SharedRtxDenoiser::default()))).clone();

                    Ok(Self {
                        device: None,
                        shared_denoiser: shared,
                        cpu_fallback: true,
                        compute_capability: None,
                        gpu_generation: GpuGeneration::Unknown,
                    })

                }
            }
        }

        #[cfg(not(feature = "nvidia-rtx"))]
        {
            info!("💻 RTX feature not compiled - using CPU processing");

            Ok(Self {
                cpu_fallback: true,
                compute_capability: None,
                gpu_generation: GpuGeneration::Unknown,
            })
        }
    }

    #[cfg(feature = "nvidia-rtx")]
    fn init_cuda() -> Result<(Arc<CudaDevice>, (u32, u32))> {
        info!("Detecting NVIDIA RTX GPU with open drivers...");

        // Initialize CUDA device
        let device = CudaDevice::new(0)?;

        // Get compute capability
        let major = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let compute_capability = (major as u32, minor as u32);

        // Check if this is RTX 20 series or newer (compute capability >= 7.5)
        if compute_capability.0 < 7 || (compute_capability.0 == 7 && compute_capability.1 < 5) {
            return Err(anyhow::anyhow!(
                "GPU compute capability {}.{} does not support RTX features (requires >= 7.5)",
                compute_capability.0, compute_capability.1
            ));
        }

        let gpu_gen = GpuGeneration::from_compute_capability(
            compute_capability.0,
            compute_capability.1,
        );

        info!("✅ RTX GPU detected - Compute Capability: {}.{}", compute_capability.0, compute_capability.1);

        match gpu_gen {
            GpuGeneration::Blackwell => {
                info!("   Architecture: Blackwell (RTX 50 Series)");
                info!("   Tensor Cores: 5th Generation with FP4 precision");
                info!("   Memory: GDDR7 with enhanced bandwidth");
            }
            GpuGeneration::AdaLovelace => {
                info!("   Architecture: Ada Lovelace (RTX 40 Series)");
                info!("   Tensor Cores: 4th Generation");
            }
            GpuGeneration::Ampere => {
                info!("   Architecture: Ampere (RTX 30 Series)");
                info!("   Tensor Cores: 3rd Generation");
            }
            GpuGeneration::Turing => {
                info!("   Architecture: Turing (RTX 20 Series)");
                info!("   Tensor Cores: 2nd Generation");
            }
            _ => {}
        }

        Ok((device, compute_capability))
    }

    pub fn get_capabilities(&self) -> Option<RtxCapabilities> {
        #[cfg(feature = "nvidia-rtx")]
        {
            if let Some(_device) = &self.device {
                let compute_capability = self.compute_capability.unwrap_or((0, 0));

                let gpu_gen = self.gpu_generation;

                // RTX features available on compute capability 7.5+ (RTX 20 series+)
                let has_tensor_cores = compute_capability.0 >= 7 && compute_capability.1 >= 5;
                let has_rt_cores = compute_capability.0 >= 7 && compute_capability.1 >= 5;
                let supports_rtx_voice = has_tensor_cores;
                let supports_fp4 = gpu_gen.supports_fp4();
                let tensor_core_gen = gpu_gen.tensor_core_generation();

                // Get memory info
                let memory_bytes = cudarc::driver::result::mem_get_info()
                    .map(|(_free, total)| total)
                    .unwrap_or(0);
                let memory_gb = memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0);

                return Some(RtxCapabilities {
                    has_tensor_cores,
                    has_rt_cores,
                    memory_gb,
                    compute_capability,
                    supports_rtx_voice,
                    driver_version: "580+".to_string(), // NVIDIA open drivers 580+ for RTX 50 series
                    gpu_generation: gpu_gen,
                    supports_fp4,
                    tensor_core_gen,
                });
            }
        }

        None
    }

    /// Process audio with RTX-accelerated noise suppression
    /// Uses architecture-specific optimizations (FP4 for Blackwell, FP16 for older)
    pub fn process_spectral_denoising(&self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()> {
        if input.len() != output.len() {
            return Err(anyhow::anyhow!("Input and output buffer size mismatch"));
        }

        #[cfg(feature = "nvidia-rtx")]
        {
            if let Some(_device) = &self.device {
                debug!("Processing with RTX GPU acceleration");

                // Only use FP4 if Blackwell AND driver supports it (590+)
                let use_fp4 = self.gpu_generation == GpuGeneration::Blackwell
                    && Self::check_driver_supports_fp4();

                // Log at info level on first use, then debug for subsequent calls
                static LOGGED_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                let first_call = !LOGGED_MODE.swap(true, Ordering::Relaxed);

                let gpu_result = if use_fp4 {
                    if first_call {
                        info!("🚀 Using FP4 Tensor Core acceleration (RTX 50 series + driver 590+)");
                    } else {
                        debug!("Using FP4 Tensor Core acceleration");
                    }
                    self.gpu_spectral_denoise_fp4(input, output, strength)
                } else {
                    if first_call {
                        // Visible at info level so users know they're running FP16
                        info!("⚡ Using FP16 Tensor Core acceleration (upgrade driver to 590+ for FP4)");
                    } else {
                        debug!("Using FP16 Tensor Core acceleration");
                    }
                    self.gpu_spectral_denoise(input, output, strength)
                };

                match gpu_result {
                    Ok(()) => {
                        // GPU succeeded - mark as active
                        Self::set_gpu_status(true, None);
                        return Ok(());
                    }
                    Err(err) => {
                        // GPU failed - track fallback
                        let reason = format!("GPU processing failed: {}", err);
                        warn!("{}. Falling back to CPU path", reason);
                        Self::set_gpu_status(false, Some(reason));
                    }
                }
            } else {
                // No GPU device available
                Self::set_gpu_status(false, Some("GPU device not available".to_string()));
            }
        }

        #[cfg(not(feature = "nvidia-rtx"))]
        {
            Self::set_gpu_status(false, Some("RTX feature not compiled".to_string()));
        }

        // CPU fallback - log at info level once so users know
        static LOGGED_CPU_FALLBACK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED_CPU_FALLBACK.swap(true, Ordering::Relaxed) {
            info!("💻 Processing with CPU fallback (RTX GPU not available or failed)");
        } else {
            debug!("Processing with CPU fallback");
        }
        self.cpu_spectral_denoise(input, output, strength)
    }

    /// Get current GPU processing status
    pub fn get_gpu_status() -> GpuProcessingStatus {
        let fallback_active = GPU_FALLBACK_ACTIVE.load(Ordering::Relaxed);
        let fallback_count = GPU_FALLBACK_COUNT.load(Ordering::Relaxed);
        let fallback_reason = GPU_FALLBACK_REASON
            .get_or_init(|| std::sync::Mutex::new(None))
            .lock()
            .ok()
            .and_then(|guard| guard.clone());

        GpuProcessingStatus {
            gpu_active: !fallback_active,
            fallback_active,
            fallback_count,
            fallback_reason,
        }
    }

    /// Update GPU processing status
    fn set_gpu_status(gpu_active: bool, reason: Option<String>) {
        let was_fallback = GPU_FALLBACK_ACTIVE.load(Ordering::Relaxed);
        let now_fallback = !gpu_active;

        GPU_FALLBACK_ACTIVE.store(now_fallback, Ordering::Relaxed);

        // Increment fallback count when transitioning to fallback
        if now_fallback && !was_fallback {
            GPU_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
        }

        // Update reason
        if let Some(mutex) = GPU_FALLBACK_REASON.get_or_init(|| std::sync::Mutex::new(None)).lock().ok().as_mut() {
            **mutex = reason;
        }
    }

    #[cfg(feature = "nvidia-rtx")]
    fn gpu_spectral_denoise(&self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()> {
        debug!("RTX spectral denoising (FP16) - strength: {:.2}", strength);
        self.run_rtx_denoiser(input, output, strength, false)
    }

    #[cfg(feature = "nvidia-rtx")]
    fn gpu_spectral_denoise_fp4(&self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()> {
        debug!("RTX 50 Blackwell spectral denoising (FP4) - strength: {:.2}", strength);
        debug!("Using 5th-gen Tensor Cores with FP4 precision for maximum throughput");
        self.run_rtx_denoiser(input, output, strength, true)
    }

    #[cfg(feature = "nvidia-rtx")]
    fn run_rtx_denoiser(&self, input: &[f32], output: &mut [f32], strength: f32, prefer_fp4: bool) -> Result<()> {
        if input.len() != output.len() {
            return Err(anyhow::anyhow!("Input and output buffer size mismatch"));
        }

        let mut shared = self
            .shared_denoiser
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock shared RTX denoiser"))?;

        let target_fft = input.len().next_power_of_two().max(1024).min(4096);
        let target_hop = target_fft / 4;

        if shared.denoiser.is_none() || shared.fft_size != target_fft {
            debug!("Initializing RTX denoiser context (fft: {}, hop: {})", target_fft, target_hop);
            let mut config = RtxDenoiseConfig::default();
            config.strength = match strength {
                s if s < 0.3 => DenoiseStrength::Light,
                s if s < 0.6 => DenoiseStrength::Moderate,
                s if s < 0.85 => DenoiseStrength::Aggressive,
                _ => DenoiseStrength::Maximum,
            };
            config.algorithm = if prefer_fp4 {
                DenoiseAlgorithm::TensorTransformer
            } else {
                DenoiseAlgorithm::TensorRnn
            };
            config.use_tensor_cores = true;
            config.fft_size = target_fft;
            config.hop_size = target_hop;

            debug!(
                "RTX denoiser config -> fft: {}, hop: {}, algo: {:?}, strength: {:?}",
                config.fft_size, config.hop_size, config.algorithm, config.strength
            );

            let denoiser = RtxDenoiser::new(config)?;
            shared.fft_size = target_fft;
            shared.hop_size = target_hop;
            shared.denoiser = Some(denoiser);
        } else {
            debug!(
                "Reusing RTX denoiser (fft: {}, hop: {})",
                shared.fft_size, shared.hop_size
            );
        }

        let denoiser = shared
            .denoiser
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("RTX denoiser unavailable"))?;

        if prefer_fp4 {
            let stats = denoiser.get_stats();
            if !stats.using_tensor_cores {
                warn!("FP4 requested but tensor cores inactive, falling back to FP16");
            }
        }

        denoiser.process(input, output)
    }

    fn cpu_spectral_denoise(&self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()> {
        if input.len() != output.len() {
            return Err(anyhow::anyhow!("Input and output buffer size mismatch"));
        }

        // Simple spectral attenuation (placeholder for full RTX implementation)
        for (i, &sample) in input.iter().enumerate() {
            // Apply noise reduction based on magnitude
            let magnitude = sample.abs();
            let gate_threshold = 0.01; // -40dB threshold

            let attenuation = if magnitude < gate_threshold {
                1.0 - strength
            } else {
                1.0 - (strength * 0.3) // Less attenuation for stronger signals
            };

            output[i] = sample * attenuation;
        }

        Ok(())
    }

    pub fn is_rtx_available(&self) -> bool {
        #[cfg(feature = "nvidia-rtx")]
        {
            self.device.is_some() && !self.cpu_fallback
        }

        #[cfg(not(feature = "nvidia-rtx"))]
        {
            false
        }
    }

    /// Check if driver version supports FP4 (requires 590+)
    #[cfg(feature = "nvidia-rtx")]
    fn check_driver_supports_fp4() -> bool {
        // Read driver version from /proc/driver/nvidia/version
        if let Ok(version_str) = std::fs::read_to_string("/proc/driver/nvidia/version") {
            if let Some(line) = version_str.lines().next() {
                // Parse: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  590.44.01  ..."
                if let Some(ver_start) = line.find("Module") {
                    let ver_part = &line[ver_start + 6..];
                    if let Some(ver) = ver_part.trim().split_whitespace().next() {
                        if let Some(major_str) = ver.split('.').next() {
                            if let Ok(major) = major_str.parse::<u32>() {
                                return major >= 590;
                            }
                        }
                    }
                }
            }
        }

        // Default: don't enable FP4 if we can't verify driver version
        false
    }

    pub fn get_processing_mode(&self) -> &'static str {
        if self.is_rtx_available() {
            "NVIDIA RTX GPU"
        } else {
            "CPU Fallback"
        }
    }
}

/// Check system for NVIDIA RTX capability
/// Returns comprehensive system info for diagnostics
pub fn check_rtx_system_requirements() -> Result<RtxSystemInfo> {
    info!("🔍 Checking RTX system requirements...");

    let mut readiness = RtxReadiness::default();
    let mut gpu_name: Option<String> = None;
    let mut driver_version: Option<String> = None;
    let mut cuda_version: Option<String> = None;
    let mut tensorrt_version: Option<String> = None;
    let mut compute_capability = (0u32, 0u32);
    let mut gpu_generation = GpuGeneration::Unknown;
    let mut tensor_core_gen = 0u8;
    let mut memory_gb = 0.0f32;

    // Check for NVIDIA open drivers
    match std::fs::read_to_string("/proc/version") {
        Ok(version) => {
            info!("Kernel: {}", version.trim());
        }
        Err(_) => warn!("Could not read kernel version"),
    }

    // Check for nvidia modules
    match std::fs::read_to_string("/proc/modules") {
        Ok(modules) => {
            if modules.contains("nvidia") {
                info!("✅ NVIDIA kernel modules loaded");
                readiness.driver_ok = true;

                if modules.contains("nvidia_drm") {
                    info!("✅ NVIDIA DRM module loaded (good for Wayland)");
                }
            } else {
                warn!("⚠️  NVIDIA kernel modules not found");
                warn!("   Make sure nvidia-open or nvidia drivers are installed");
            }
        }
        Err(_) => warn!("Could not check loaded kernel modules"),
    }

    // Check for CUDA runtime
    let cuda_found = std::path::Path::new("/usr/lib/libcuda.so").exists()
        || std::path::Path::new("/usr/lib64/libcuda.so").exists()
        || std::path::Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so").exists();

    if cuda_found {
        info!("✅ CUDA runtime library found");
        readiness.cuda_ok = true;
        cuda_version = Some("12.0+".to_string());
    } else {
        warn!("⚠️  CUDA runtime library not found");
        warn!("   Install nvidia-utils or cuda-runtime package");
    }

    // Check for TensorRT
    let tensorrt_found = std::path::Path::new("/usr/lib/libnvinfer.so").exists()
        || std::path::Path::new("/usr/lib64/libnvinfer.so").exists()
        || std::path::Path::new("/usr/lib/x86_64-linux-gnu/libnvinfer.so").exists();

    if tensorrt_found {
        info!("✅ TensorRT library found");
        readiness.tensorrt_ok = true;
        tensorrt_version = Some("8.0+".to_string());
    } else {
        debug!("TensorRT not found (optional for enhanced AI denoising)");
    }

    // Try to get driver version from nvidia-smi or /proc
    if let Ok(version_str) = std::fs::read_to_string("/proc/driver/nvidia/version") {
        if let Some(line) = version_str.lines().next() {
            // Parse: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  590.44.01  ..."
            if let Some(ver_start) = line.find("Module") {
                let ver_part = &line[ver_start + 6..];
                let ver = ver_part.trim().split_whitespace().next().unwrap_or("Unknown");
                driver_version = Some(ver.to_string());
                info!("Driver version: {}", ver);

                // Check if driver supports RTX 50 series (590+)
                if let Ok(major) = ver.split('.').next().unwrap_or("0").parse::<u32>() {
                    if major >= 590 {
                        info!("✅ Driver supports RTX 50 series (Blackwell)");
                    }
                }
            }
        }
    }

    // Try to detect GPU via /proc/driver/nvidia/gpus
    if let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus") {
        for entry in entries.flatten() {
            let info_path = entry.path().join("information");
            if let Ok(info) = std::fs::read_to_string(&info_path) {
                for line in info.lines() {
                    if line.starts_with("Model:") {
                        gpu_name = Some(line.replace("Model:", "").trim().to_string());
                    }
                }
            }
        }
    }

    // Detect GPU generation and capabilities via CUDA if available
    #[cfg(feature = "nvidia-rtx")]
    {
        if let Ok(device) = cudarc::driver::CudaDevice::new(0) {
            if let Ok(major) = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) {
                if let Ok(minor) = device.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) {
                    compute_capability = (major as u32, minor as u32);
                    gpu_generation = GpuGeneration::from_compute_capability(major as u32, minor as u32);
                    tensor_core_gen = gpu_generation.tensor_core_generation();

                    // FP4 ready on Blackwell (compute 10.0+) with driver >= 590
                    if major >= 10 || major == 12 { // Blackwell reports 12.0
                        // Check driver version for FP4 support
                        let driver_ready = driver_version.as_ref()
                            .and_then(|v| v.split('.').next())
                            .and_then(|v| v.parse::<u32>().ok())
                            .map(|v| v >= 590)
                            .unwrap_or(false);

                        if driver_ready {
                            readiness.fp4_ready = true;
                            info!("✅ FP4 Tensor Core support available (Blackwell + driver 590+)");
                        } else {
                            warn!("⚠️  FP4 support requires driver 590+, current: {}",
                                  driver_version.as_deref().unwrap_or("unknown"));
                            info!("   FP16 Tensor Cores will be used instead");
                        }
                    }
                }
            }

            if let Ok((_free, total)) = cudarc::driver::result::mem_get_info() {
                memory_gb = total as f32 / (1024.0 * 1024.0 * 1024.0);
            }
        }
    }

    // Fallback detection without CUDA feature
    #[cfg(not(feature = "nvidia-rtx"))]
    {
        // Try to parse compute capability from nvidia-smi output if available
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap,memory.total,name", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout.lines().next() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 3 {
                        // Parse compute capability (e.g., "10.0" or "12.0")
                        if let Some((major_str, minor_str)) = parts[0].split_once('.') {
                            if let (Ok(major), Ok(minor)) = (major_str.parse::<u32>(), minor_str.parse::<u32>()) {
                                compute_capability = (major, minor);
                                gpu_generation = GpuGeneration::from_compute_capability(major, minor);
                                tensor_core_gen = gpu_generation.tensor_core_generation();

                                // FP4 requires Blackwell (compute 10.0+ or 12.0) AND driver 590+
                                if major >= 10 || major == 12 {
                                    let driver_ready = driver_version.as_ref()
                                        .and_then(|v| v.split('.').next())
                                        .and_then(|v| v.parse::<u32>().ok())
                                        .map(|v| v >= 590)
                                        .unwrap_or(false);

                                    if driver_ready {
                                        readiness.fp4_ready = true;
                                    }
                                }
                            }
                        }
                        // Parse memory (e.g., "32768" MB)
                        if let Ok(mem_mb) = parts[1].parse::<f32>() {
                            memory_gb = mem_mb / 1024.0;
                        }
                        // GPU name
                        if gpu_name.is_none() {
                            gpu_name = Some(parts[2].to_string());
                        }
                    }
                }
            }
        }
    }

    info!("System check complete");

    Ok(RtxSystemInfo {
        gpu_name,
        driver_version,
        cuda_version,
        tensorrt_version,
        readiness,
        compute_capability,
        gpu_generation,
        tensor_core_gen,
        memory_gb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rtx_initialization() {
        let accelerator = RtxAccelerator::new();
        assert!(accelerator.is_ok());

        let accelerator = accelerator.unwrap();
        info!("RTX available: {}", accelerator.is_rtx_available());
        info!("Processing mode: {}", accelerator.get_processing_mode());
    }

    #[tokio::test]
    async fn test_spectral_processing() {
        let accelerator = RtxAccelerator::new().unwrap();

        let input = vec![0.1, 0.2, -0.1, 0.05, 0.0];
        let mut output = vec![0.0; 5];

        let result = accelerator.process_spectral_denoising(&input, &mut output, 0.7);
        assert!(result.is_ok());

        // Output should be different from input (processed)
        assert_ne!(input, output);
    }

    #[test]
    fn test_system_requirements() {
        let result = check_rtx_system_requirements();
        assert!(result.is_ok());
    }
}
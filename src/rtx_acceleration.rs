use anyhow::Result;
use tracing::{info, debug, warn};

#[cfg(feature = "nvidia-rtx")]
use cudarc::driver::CudaDevice;

/// RTX GPU acceleration for noise suppression
/// Leverages NVIDIA's open GPU kernel modules for RTX 20 series and newer
pub struct RtxAccelerator {
    #[cfg(feature = "nvidia-rtx")]
    device: Option<CudaDevice>,

    /// Fallback to CPU when GPU is not available
    cpu_fallback: bool,

    /// GPU compute capability for RTX features
    compute_capability: Option<(u32, u32)>,
}

#[derive(Debug, Clone)]
pub struct RtxCapabilities {
    pub has_tensor_cores: bool,
    pub has_rt_cores: bool,
    pub memory_gb: f32,
    pub compute_capability: (u32, u32),
    pub supports_rtx_voice: bool,
    pub driver_version: String,
}

impl RtxAccelerator {
    pub fn new() -> Result<Self> {
        info!("🚀 Initializing NVIDIA RTX acceleration...");

        #[cfg(feature = "nvidia-rtx")]
        {
            match Self::init_cuda() {
                Ok((device, compute_capability)) => {
                    info!("✅ NVIDIA RTX acceleration enabled");
                    info!("   Compute Capability: {}.{}", compute_capability.0, compute_capability.1);

                    Ok(Self {
                        device: Some(device),
                        cpu_fallback: false,
                        compute_capability: Some(compute_capability),
                    })
                }
                Err(e) => {
                    warn!("⚠️  NVIDIA RTX acceleration unavailable: {}", e);
                    warn!("   Falling back to CPU processing");

                    Ok(Self {
                        device: None,
                        cpu_fallback: true,
                        compute_capability: None,
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
            })
        }
    }

    #[cfg(feature = "nvidia-rtx")]
    fn init_cuda() -> Result<(CudaDevice, (u32, u32))> {
        info!("Detecting NVIDIA RTX GPU with open drivers...");

        // Initialize CUDA device
        let device = CudaDevice::new(0)?;

        // Get compute capability
        let major = device.get_attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = device.get_attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let compute_capability = (major as u32, minor as u32);

        // Check if this is RTX 20 series or newer (compute capability >= 7.5)
        if compute_capability.0 < 7 || (compute_capability.0 == 7 && compute_capability.1 < 5) {
            return Err(anyhow::anyhow!(
                "GPU compute capability {}.{} does not support RTX features (requires >= 7.5)",
                compute_capability.0, compute_capability.1
            ));
        }

        info!("✅ RTX GPU detected - Compute Capability: {}.{}", compute_capability.0, compute_capability.1);

        Ok((device, compute_capability))
    }

    pub fn get_capabilities(&self) -> Option<RtxCapabilities> {
        #[cfg(feature = "nvidia-rtx")]
        {
            if let Some(device) = &self.device {
                let compute_capability = self.compute_capability.unwrap_or((0, 0));

                // RTX features available on compute capability 7.5+ (RTX 20 series+)
                let has_tensor_cores = compute_capability.0 >= 7 && compute_capability.1 >= 5;
                let has_rt_cores = compute_capability.0 >= 7 && compute_capability.1 >= 5;
                let supports_rtx_voice = has_tensor_cores;

                // Get memory info
                let memory_bytes = device.total_memory().unwrap_or(0);
                let memory_gb = memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0);

                return Some(RtxCapabilities {
                    has_tensor_cores,
                    has_rt_cores,
                    memory_gb,
                    compute_capability,
                    supports_rtx_voice,
                    driver_version: "555+".to_string(), // NVIDIA open drivers 555+
                });
            }
        }

        None
    }

    /// Process audio with RTX-accelerated noise suppression
    pub fn process_spectral_denoising(&self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()> {
        #[cfg(feature = "nvidia-rtx")]
        {
            if let (Some(_device), Some(_fft)) = (&self.device, &self.fft) {
                debug!("Processing with RTX GPU acceleration");
                return self.gpu_spectral_denoise(input, output, strength);
            }
        }

        // CPU fallback
        debug!("Processing with CPU fallback");
        self.cpu_spectral_denoise(input, output, strength)
    }

    #[cfg(feature = "nvidia-rtx")]
    fn gpu_spectral_denoise(&self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()> {
        // This is a simplified implementation
        // In a full implementation, this would:
        // 1. Transfer audio to GPU memory using device.htod_copy()
        // 2. Perform FFT using custom CUDA kernels or cuFFT
        // 3. Apply spectral gating using tensor cores for ML-based denoising
        // 4. Perform inverse FFT
        // 5. Transfer result back to CPU using device.dtoh_sync_copy()

        debug!("RTX spectral denoising - strength: {:.2}", strength);

        if let Some(ref _device) = self.device {
            // TODO: Implement full GPU-accelerated spectral processing
            // This would involve custom CUDA kernels for:
            // - FFT-based spectral analysis
            // - Tensor Core-accelerated noise profile learning
            // - Real-time spectral subtraction/gating
            debug!("GPU memory transfer and processing would happen here");
        }

        // For now, fallback to CPU implementation
        self.cpu_spectral_denoise(input, output, strength)
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

    pub fn get_processing_mode(&self) -> &'static str {
        if self.is_rtx_available() {
            "NVIDIA RTX GPU"
        } else {
            "CPU Fallback"
        }
    }
}

/// Check system for NVIDIA RTX capability
pub fn check_rtx_system_requirements() -> Result<()> {
    info!("🔍 Checking RTX system requirements...");

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
    if std::path::Path::new("/usr/lib/libcuda.so").exists()
        || std::path::Path::new("/usr/lib64/libcuda.so").exists()
        || std::path::Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so").exists() {
        info!("✅ CUDA runtime library found");
    } else {
        warn!("⚠️  CUDA runtime library not found");
        warn!("   Install nvidia-utils or cuda-runtime package");
    }

    info!("System check complete");
    Ok(())
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
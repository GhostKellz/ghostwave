#[cfg(feature = "nvidia-rtx")]
pub use ghostwave_core::rtx_acceleration::{
    RtxAccelerator, check_rtx_system_requirements,
};

// Stub types when nvidia-rtx feature is not enabled
#[cfg(not(feature = "nvidia-rtx"))]
pub use self::stubs::*;

#[cfg(not(feature = "nvidia-rtx"))]
#[allow(dead_code)] // Public API stubs for when nvidia-rtx feature is disabled
mod stubs {
    use anyhow::Result;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum GpuGeneration {
        Turing,
        Ampere,
        AdaLovelace,
        Blackwell,
        Unknown,
    }

    #[derive(Debug, Clone, Default)]
    pub struct RtxReadiness {
        pub driver_ok: bool,
        pub cuda_ok: bool,
        pub tensorrt_ok: bool,
        pub fp4_ready: bool,
    }

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
        pub supports_fp4: bool,
        pub tensor_core_gen: u8,
    }

    pub struct RtxAccelerator {
        cpu_fallback: bool,
    }

    impl RtxAccelerator {
        pub fn new() -> Result<Self> {
            Ok(Self { cpu_fallback: true })
        }

        pub fn is_rtx_available(&self) -> bool {
            false
        }

        pub fn get_capabilities(&self) -> Option<RtxCapabilities> {
            None
        }

        pub fn get_processing_mode(&self) -> &'static str {
            "CPU Fallback (nvidia-rtx feature not compiled)"
        }
    }

    /// Stub check_rtx_system_requirements for non-RTX builds
    pub fn check_rtx_system_requirements() -> Result<RtxSystemInfo> {
        Ok(RtxSystemInfo {
            gpu_name: None,
            driver_version: None,
            cuda_version: None,
            tensorrt_version: None,
            readiness: RtxReadiness::default(),
            compute_capability: (0, 0),
            gpu_generation: GpuGeneration::Unknown,
            tensor_core_gen: 0,
            memory_gb: 0.0,
        })
    }
}

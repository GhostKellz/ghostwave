//! Build script for GhostWave
//!
//! This script auto-detects CUDA availability at build time and sets
//! appropriate configuration flags. If CUDA is found, RTX acceleration
//! features are automatically enabled.

use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    // Register custom cfg flags so rustc doesn't warn about them
    println!("cargo::rustc-check-cfg=cfg(has_cuda)");
    println!("cargo::rustc-check-cfg=cfg(has_nvidia_gpu)");

    // Check for CUDA availability
    let cuda_available = detect_cuda();

    if cuda_available {
        println!("cargo:warning=CUDA detected - RTX acceleration will be enabled");
        println!("cargo:rustc-cfg=has_cuda");
    } else {
        println!("cargo:warning=CUDA not detected - building with CPU-only support");
    }

    // Check for nvidia-smi (indicates NVIDIA GPU present)
    if detect_nvidia_gpu() {
        println!("cargo:rustc-cfg=has_nvidia_gpu");
    }

    // Rerun if CUDA installation changes
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rerun-if-changed={}", cuda_path);
    }
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}

/// Detect if CUDA toolkit is installed
fn detect_cuda() -> bool {
    // Method 1: Check for nvcc compiler
    if Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        println!("cargo:warning=Found CUDA via nvcc");
        return true;
    }

    // Method 2: Check CUDA_PATH environment variable
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc_path = Path::new(&cuda_path).join("bin/nvcc");
        if nvcc_path.exists() {
            println!("cargo:warning=Found CUDA via CUDA_PATH: {}", cuda_path);
            return true;
        }
    }

    // Method 3: Check CUDA_HOME environment variable
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let nvcc_path = Path::new(&cuda_home).join("bin/nvcc");
        if nvcc_path.exists() {
            println!("cargo:warning=Found CUDA via CUDA_HOME: {}", cuda_home);
            return true;
        }
    }

    // Method 4: Check common installation paths
    let common_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-11",
        "/opt/cuda",
    ];

    for path in common_paths {
        let nvcc_path = Path::new(path).join("bin/nvcc");
        if nvcc_path.exists() {
            println!("cargo:warning=Found CUDA at: {}", path);
            return true;
        }
    }

    false
}

/// Detect if an NVIDIA GPU is present in the system
fn detect_nvidia_gpu() -> bool {
    // Check nvidia-smi
    if Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        println!("cargo:warning=NVIDIA GPU detected via nvidia-smi");
        return true;
    }

    // Check /proc/driver/nvidia
    if Path::new("/proc/driver/nvidia/version").exists() {
        println!("cargo:warning=NVIDIA driver detected via /proc");
        return true;
    }

    false
}

//! FFI usage example (demonstrating C API from Rust)
//!
//! Run with: cargo run --example 03_ffi_usage

use ghostwave_core::ffi::*;
use std::ffi::CStr;

fn main() {
    println!("GhostWave FFI Example");
    println!("=====================");

    // Get version
    let version = unsafe { CStr::from_ptr(ghostwave_version()) };
    println!("Library version: {}", version.to_str().unwrap());

    // Create processor
    let mut handle = GhostWaveHandle::null();
    let err = ghostwave_create(&mut handle, 48000, 1, 256);

    if err.code != 0 {
        let msg = String::from_utf8_lossy(&err.message);
        eprintln!("Error creating processor: {}", msg.trim_end_matches('\0'));
        return;
    }

    println!("Processor created successfully");

    // Check RTX
    let has_rtx = ghostwave_has_rtx(handle);
    println!("RTX available: {}", has_rtx);

    // Get processing mode
    let mut mode_buf = [0u8; 64];
    let err = ghostwave_get_processing_mode(
        handle,
        mode_buf.as_mut_ptr() as *mut i8,
        mode_buf.len(),
    );

    if err.code == 0 {
        let mode = String::from_utf8_lossy(&mode_buf);
        println!("Processing mode: {}", mode.trim_end_matches('\0'));
    }

    // Process audio
    let input = vec![0.1f32; 256];
    let mut output = vec![0.0f32; 256];

    let err = ghostwave_process(handle, input.as_ptr(), output.as_mut_ptr(), 256);

    if err.code == 0 {
        println!("Processed 256 samples successfully");
    } else {
        let msg = String::from_utf8_lossy(&err.message);
        eprintln!("Processing error: {}", msg.trim_end_matches('\0'));
    }

    // Cleanup
    ghostwave_destroy(handle);
    println!("Processor destroyed");
}

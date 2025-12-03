/**
 * GhostWave Core - C FFI Header
 *
 * NVIDIA RTX Voice-style AI noise cancellation for Linux
 *
 * Usage:
 *   #include "ghostwave.h"
 *
 *   GhostWaveHandle handle;
 *   GhostWaveError err = ghostwave_create(&handle, 48000, 1, 256);
 *   if (err.code != 0) {
 *       fprintf(stderr, "Error: %s\n", err.message);
 *       return 1;
 *   }
 *
 *   float input[256], output[256];
 *   // ... fill input with audio samples ...
 *
 *   err = ghostwave_process(handle, input, output, 256);
 *   if (err.code != 0) {
 *       fprintf(stderr, "Processing error: %s\n", err.message);
 *   }
 *
 *   ghostwave_destroy(handle);
 *
 * Link with: -lghostwave_core
 *
 * Copyright (c) 2025 Christopher Kelley
 * License: MIT OR Apache-2.0
 */

#ifndef GHOSTWAVE_H
#define GHOSTWAVE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Types
 * ============================================================================ */

/**
 * Opaque handle to GhostWave processor
 * Must be created with ghostwave_create() and destroyed with ghostwave_destroy()
 */
typedef struct {
    void* ptr;
} GhostWaveHandle;

/**
 * Error structure returned by API functions
 * code = 0 means success
 */
typedef struct {
    int32_t code;           /* Error code (0 = success) */
    uint8_t message[256];   /* Null-terminated error message */
} GhostWaveError;

/**
 * Processing profile presets
 */
typedef enum {
    GHOSTWAVE_PROFILE_BALANCED = 0,   /* Balanced noise reduction (default) */
    GHOSTWAVE_PROFILE_STREAMING = 1,  /* Aggressive for streaming */
    GHOSTWAVE_PROFILE_STUDIO = 2      /* Minimal for professional recording */
} GhostWaveProfile;

/**
 * GPU acceleration information
 */
typedef struct {
    bool available;         /* GPU acceleration available */
    int32_t compute_major;  /* CUDA compute capability major */
    int32_t compute_minor;  /* CUDA compute capability minor */
    float memory_gb;        /* GPU memory in GB */
    uint8_t name[64];       /* GPU name (null-terminated) */
} GhostWaveGpuInfo;

/**
 * Processing statistics
 */
typedef struct {
    uint64_t frames_processed;  /* Total frames processed */
    uint64_t xrun_count;        /* Buffer underrun count */
    uint64_t latency_us;        /* Average latency in microseconds */
    float cpu_usage_pct;        /* CPU usage percentage */
} GhostWaveStats;

/* ============================================================================
 * Error Codes
 * ============================================================================ */

/* Success */
#define GHOSTWAVE_OK                     0

/* Initialization errors (1xx) */
#define GHOSTWAVE_ERR_GPU_NOT_AVAILABLE  101
#define GHOSTWAVE_ERR_INVALID_CONFIG     102
#define GHOSTWAVE_ERR_MODEL_NOT_FOUND    103
#define GHOSTWAVE_ERR_MODEL_LOAD_FAILED  104
#define GHOSTWAVE_ERR_UNSUPPORTED_RATE   105
#define GHOSTWAVE_ERR_UNSUPPORTED_BUFFER 106

/* Runtime errors (2xx) */
#define GHOSTWAVE_ERR_PROCESSING         201
#define GHOSTWAVE_ERR_BUFFER_MISMATCH    202
#define GHOSTWAVE_ERR_BUFFER_TOO_SMALL   203
#define GHOSTWAVE_ERR_LOCK               204
#define GHOSTWAVE_ERR_NOT_INITIALIZED    205

/* Device errors (3xx) */
#define GHOSTWAVE_ERR_AUDIO_DEVICE       301
#define GHOSTWAVE_ERR_DEVICE_NOT_FOUND   302
#define GHOSTWAVE_ERR_DEVICE_BUSY        303

/* Resource errors (4xx) */
#define GHOSTWAVE_ERR_INSUFFICIENT_MEM   401
#define GHOSTWAVE_ERR_INSUFFICIENT_VRAM  402
#define GHOSTWAVE_ERR_ALLOCATION_FAILED  403

/* IO errors (5xx) */
#define GHOSTWAVE_ERR_IO                 501
#define GHOSTWAVE_ERR_NETWORK            502

/* FFI errors (6xx) */
#define GHOSTWAVE_ERR_NULL_POINTER       601
#define GHOSTWAVE_ERR_INVALID_HANDLE     602
#define GHOSTWAVE_ERR_STRING             603
#define GHOSTWAVE_ERR_PANIC              604

/* Unknown (9xx) */
#define GHOSTWAVE_ERR_UNKNOWN            999

/* ============================================================================
 * Core API
 * ============================================================================ */

/**
 * Create a new GhostWave processor
 *
 * @param handle_out    Output pointer for the created handle
 * @param sample_rate   Audio sample rate (44100, 48000, 96000, 192000)
 * @param channels      Number of audio channels (1 = mono, 2 = stereo)
 * @param buffer_size   Processing buffer size in frames (32-4096, power of 2)
 * @return              Error code (GHOSTWAVE_OK on success)
 *
 * Example:
 *   GhostWaveHandle handle;
 *   GhostWaveError err = ghostwave_create(&handle, 48000, 1, 256);
 */
GhostWaveError ghostwave_create(
    GhostWaveHandle* handle_out,
    uint32_t sample_rate,
    uint32_t channels,
    uint32_t buffer_size
);

/**
 * Create processor with a specific profile preset
 *
 * @param handle_out    Output pointer for the created handle
 * @param sample_rate   Audio sample rate
 * @param channels      Number of audio channels
 * @param buffer_size   Processing buffer size
 * @param profile       Processing profile preset
 * @return              Error code
 */
GhostWaveError ghostwave_create_with_profile(
    GhostWaveHandle* handle_out,
    uint32_t sample_rate,
    uint32_t channels,
    uint32_t buffer_size,
    GhostWaveProfile profile
);

/**
 * Destroy a GhostWave processor and free resources
 *
 * @param handle    Handle to destroy (may be invalid/null)
 *
 * Note: Safe to call with null/invalid handle
 */
void ghostwave_destroy(GhostWaveHandle handle);

/**
 * Process audio through noise suppression
 *
 * @param handle    Processor handle
 * @param input     Input audio samples (f32, -1.0 to 1.0 range)
 * @param output    Output buffer for processed audio (must be same size as input)
 * @param frames    Number of samples to process
 * @return          Error code
 *
 * Note: For stereo, frames is total samples (e.g., 256 stereo frames = 512 samples)
 *
 * Example:
 *   float input[256], output[256];
 *   GhostWaveError err = ghostwave_process(handle, input, output, 256);
 */
GhostWaveError ghostwave_process(
    GhostWaveHandle handle,
    const float* input,
    float* output,
    size_t frames
);

/**
 * Process audio in-place (input buffer is modified)
 *
 * @param handle    Processor handle
 * @param buffer    Audio buffer (input and output)
 * @param frames    Number of samples
 * @return          Error code
 */
GhostWaveError ghostwave_process_inplace(
    GhostWaveHandle handle,
    float* buffer,
    size_t frames
);

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * Set noise suppression strength
 *
 * @param handle    Processor handle
 * @param strength  Strength value (0.0 = disabled, 1.0 = maximum)
 * @return          Error code
 */
GhostWaveError ghostwave_set_noise_strength(
    GhostWaveHandle handle,
    float strength
);

/**
 * Enable or disable noise suppression processing
 *
 * @param handle    Processor handle
 * @param enabled   true to enable, false for passthrough
 * @return          Error code
 */
GhostWaveError ghostwave_set_enabled(
    GhostWaveHandle handle,
    bool enabled
);

/* ============================================================================
 * Query Functions
 * ============================================================================ */

/**
 * Get GPU acceleration status and info
 *
 * @param handle    Processor handle
 * @param info_out  Output pointer for GPU info
 * @return          Error code
 */
GhostWaveError ghostwave_get_gpu_info(
    GhostWaveHandle handle,
    GhostWaveGpuInfo* info_out
);

/**
 * Get current processing mode string
 *
 * @param handle        Processor handle
 * @param mode_out      Output buffer for mode string
 * @param mode_out_len  Size of output buffer
 * @return              Error code
 */
GhostWaveError ghostwave_get_processing_mode(
    GhostWaveHandle handle,
    char* mode_out,
    size_t mode_out_len
);

/**
 * Get library version string
 *
 * @return  Version string (e.g., "0.2.0")
 */
const char* ghostwave_version(void);

/**
 * Check if RTX GPU acceleration is available
 *
 * @param handle    Processor handle
 * @return          true if RTX acceleration is active
 */
bool ghostwave_has_rtx(GhostWaveHandle handle);

/* ============================================================================
 * Utility Macros
 * ============================================================================ */

/** Check if error code indicates success */
#define GHOSTWAVE_SUCCEEDED(err) ((err).code == GHOSTWAVE_OK)

/** Check if error code indicates failure */
#define GHOSTWAVE_FAILED(err) ((err).code != GHOSTWAVE_OK)

#ifdef __cplusplus
}
#endif

#endif /* GHOSTWAVE_H */

//! CPAL backend integration
//!
//! Provides cross-platform audio support through CPAL.

#[cfg(feature = "cpal-backend")]
use cpal::traits::HostTrait;

/// Check if CPAL is available (should always be true)
#[cfg(feature = "cpal-backend")]
pub fn check_cpal_availability() -> bool {
    let host = cpal::default_host();
    match host.default_input_device() {
        Some(_) => true,
        None => false,
    }
}

#[cfg(not(feature = "cpal-backend"))]
pub fn check_cpal_availability() -> bool {
    false
}
//! ALSA backend integration
//!
//! Provides direct ALSA hardware access for minimal latency.

#[cfg(feature = "alsa-backend")]
use alsa::pcm::PCM;

/// Check if ALSA is available on the system
#[cfg(feature = "alsa-backend")]
pub fn check_alsa_availability() -> bool {
    match PCM::new("default", alsa::Direction::Capture, false) {
        Ok(_) => true,
        Err(_) => false,
    }
}

#[cfg(not(feature = "alsa-backend"))]
pub fn check_alsa_availability() -> bool {
    false
}
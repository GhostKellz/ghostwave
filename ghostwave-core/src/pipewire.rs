//! PipeWire backend integration
//!
//! Provides modern Linux audio system integration through PipeWire.

#[cfg(not(feature = "pipewire-backend"))]
pub fn check_pipewire_availability() -> bool {
    false
}

#[cfg(feature = "pipewire-backend")]
pub fn check_pipewire_availability() -> bool {
    // Try to initialize PipeWire (note: pipewire::init() returns ())
    // For now, just assume it's available if the feature is compiled in
    true
}
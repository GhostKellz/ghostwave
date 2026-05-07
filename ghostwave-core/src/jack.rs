//! JACK backend integration
//!
//! Provides professional audio workflow integration through JACK.

#[cfg(feature = "jack-backend")]
use jack::{Client, ClientOptions};

/// Check if JACK server is running
#[cfg(feature = "jack-backend")]
pub fn check_jack_availability() -> bool {
    Client::new("ghostwave-test", ClientOptions::NO_START_SERVER).is_ok()
}

#[cfg(not(feature = "jack-backend"))]
pub fn check_jack_availability() -> bool {
    false
}
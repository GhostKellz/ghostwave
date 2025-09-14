use anyhow::{Result, Context};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tracing::info;

pub struct SystemdService;

impl SystemdService {
    pub fn install() -> Result<()> {
        info!("Installing systemd user service");

        let service_dir = Self::systemd_user_dir()?;
        fs::create_dir_all(&service_dir)
            .context("Failed to create systemd user directory")?;

        let service_content = Self::generate_service_file()?;
        let service_path = service_dir.join("ghostwave.service");

        fs::write(&service_path, &service_content)
            .context("Failed to write service file")?;

        info!("Created systemd service: {:?}", service_path);

        // Reload systemd and enable service
        Self::reload_systemd()?;
        Self::enable_service()?;

        info!("✅ GhostWave systemd service installed and enabled");
        info!("Service will auto-start on login");

        Ok(())
    }

    pub fn uninstall() -> Result<()> {
        info!("Removing systemd user service");

        // Stop and disable service first
        Self::stop_service().ok();
        Self::disable_service().ok();

        let service_path = Self::systemd_user_dir()?.join("ghostwave.service");
        if service_path.exists() {
            fs::remove_file(&service_path)
                .context("Failed to remove service file")?;
            info!("Removed: {:?}", service_path);
        }

        Self::reload_systemd()?;

        info!("✅ GhostWave systemd service uninstalled");
        Ok(())
    }

    pub fn start() -> Result<()> {
        info!("Starting GhostWave service");
        Self::run_systemctl(&["--user", "start", "ghostwave.service"])
    }

    pub fn stop() -> Result<()> {
        info!("Stopping GhostWave service");
        Self::run_systemctl(&["--user", "stop", "ghostwave.service"])
    }

    pub fn status() -> Result<()> {
        info!("GhostWave service status:");
        Self::run_systemctl(&["--user", "status", "ghostwave.service"])
    }

    fn systemd_user_dir() -> Result<PathBuf> {
        let mut config_dir = dirs::config_dir()
            .context("Failed to get config directory")?;
        config_dir.push("systemd");
        config_dir.push("user");
        Ok(config_dir)
    }

    fn generate_service_file() -> Result<String> {
        let ghostwave_binary = Self::find_ghostwave_binary()?;

        Ok(format!(r#"[Unit]
Description=GhostWave - NVIDIA RTX Voice for Linux
Documentation=https://github.com/ghostkellz/ghostwave
After=pipewire.service
Wants=pipewire.service

[Service]
Type=simple
ExecStart={} --pipewire-module --profile balanced
Restart=on-failure
RestartSec=5
Environment=GHOSTWAVE_LOG=info

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes
RestrictRealtime=yes
LockPersonality=yes

[Install]
WantedBy=default.target
"#, ghostwave_binary))
    }

    fn find_ghostwave_binary() -> Result<String> {
        // Try to find the ghostwave binary
        if let Ok(output) = Command::new("which").arg("ghostwave").output() {
            if output.status.success() {
                return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }
        }

        // Fallback to common locations
        let possible_paths = [
            "/usr/local/bin/ghostwave",
            "/usr/bin/ghostwave",
            "~/.cargo/bin/ghostwave",
        ];

        for path in &possible_paths {
            if PathBuf::from(path).exists() {
                return Ok(path.to_string());
            }
        }

        // If not found, use cargo run as fallback for development
        Ok("cargo run --release --bin ghostwave --".to_string())
    }

    fn reload_systemd() -> Result<()> {
        Self::run_systemctl(&["--user", "daemon-reload"])
    }

    fn enable_service() -> Result<()> {
        Self::run_systemctl(&["--user", "enable", "ghostwave.service"])
    }

    fn disable_service() -> Result<()> {
        Self::run_systemctl(&["--user", "disable", "ghostwave.service"])
    }

    fn stop_service() -> Result<()> {
        Self::run_systemctl(&["--user", "stop", "ghostwave.service"])
    }

    fn run_systemctl(args: &[&str]) -> Result<()> {
        let output = Command::new("systemctl")
            .args(args)
            .output()
            .context("Failed to run systemctl")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("systemctl failed: {}", stderr));
        }

        if !output.stdout.is_empty() {
            println!("{}", String::from_utf8_lossy(&output.stdout));
        }

        Ok(())
    }
}

pub fn install_systemd_service() -> Result<()> {
    SystemdService::install()
}

pub fn uninstall_systemd_service() -> Result<()> {
    SystemdService::uninstall()
}

pub fn start_service() -> Result<()> {
    SystemdService::start()
}

pub fn stop_service() -> Result<()> {
    SystemdService::stop()
}

pub fn service_status() -> Result<()> {
    SystemdService::status()
}
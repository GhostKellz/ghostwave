#!/usr/bin/env bash
set -euo pipefail

REPO="GhostKellz/ghostwave"
API_URL="https://api.github.com/repos/${REPO}/releases/latest"
INSTALL_PREFIX="/usr/local"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

log() {
  printf '\033[1;36m==>\033[0m %s\n' "$1"
}

fail() {
  printf '\033[1;31merror:\033[0m %s\n' "$1" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

need_cmd curl
need_cmd tar
need_cmd install

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  fail "required command not found: python3 or python"
fi

if [[ "${EUID}" -ne 0 ]]; then
  fail "run this installer as root (sudo bash install-system.sh)"
fi

log "GhostWave installer"
log "Fetching latest release metadata"

release_json="${TMP_DIR}/release.json"
curl -fsSL "$API_URL" -o "$release_json"

asset_url="$($PYTHON_BIN - <<'PY' "$release_json"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
assets = {asset['name']: asset['browser_download_url'] for asset in data.get('assets', [])}
name = next((n for n in assets if n.startswith('ghostwave-') and n.endswith('-linux-x86_64.tar.gz')), None)
if name:
    print(assets[name])
    raise SystemExit(0)
raise SystemExit(1)
PY
)" || fail "could not locate a downloadable Linux tar.gz release asset"

asset_name="$(basename "$asset_url")"
archive_path="${TMP_DIR}/${asset_name}"

log "Downloading ${asset_name}"
curl -fsSL "$asset_url" -o "$archive_path"

log "Extracting archive"
tar -xzf "$archive_path" -C "$TMP_DIR"

extract_root="$(find "$TMP_DIR" -mindepth 1 -maxdepth 1 -type d -name 'ghostwave-*' | head -n1)"
[[ -n "$extract_root" ]] || fail "failed to find extracted release directory"

# Install binary
install -d "${INSTALL_PREFIX}/bin"
if [[ -f "${extract_root}/ghostwave" ]]; then
  install -m755 "${extract_root}/ghostwave" "${INSTALL_PREFIX}/bin/ghostwave"
  log "Installed ghostwave -> ${INSTALL_PREFIX}/bin/ghostwave"
else
  fail "ghostwave binary not found in release archive"
fi

# Install systemd user service
if [[ -f "${extract_root}/ghostwave.user.service" ]]; then
  install -d "/usr/lib/systemd/user"
  install -m644 "${extract_root}/ghostwave.user.service" "/usr/lib/systemd/user/ghostwave.service"
  log "Installed systemd user service"
fi

# Install default configuration
if [[ -f "${extract_root}/config.toml" ]]; then
  install -d "/etc/ghostwave"
  if [[ ! -f "/etc/ghostwave/config.toml" ]]; then
    install -m644 "${extract_root}/config.toml" "/etc/ghostwave/config.toml"
    log "Installed default configuration"
  else
    log "Existing config preserved at /etc/ghostwave/config.toml"
  fi
fi

log "Done"
printf 'Run: ghostwave --version\n'
printf 'Run: ghostwave --doctor        (system diagnostics)\n'
printf 'Run: ghostwave --pipewire-module (start PipeWire filter)\n'
printf '\nTo enable as a service:\n'
printf '  systemctl --user enable --now ghostwave\n'

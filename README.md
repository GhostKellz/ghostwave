# GhostWave 🎧⚡

<div align="center">
  <img src="assets/icons/ghostwave.png" alt="GhostWave Logo" width="128" height="128">

  **NVIDIA RTX Voice–powered Noise Suppression for Linux**  
  _Wayland-ready · Low-latency · Built for creators & gamers_
</div>

---

## ✨ Overview

**GhostWave** brings **NVIDIA’s RTX Voice–style AI noise cancellation** to Linux.  
It integrates seamlessly with your audio workflow, offering **real-time voice cleanup** for:

- 🎮 Gaming (Discord, Steam, OBS, etc.)  
- 🎤 Streaming & Podcasting  
- 💼 Professional calls (Zoom, Teams, Meet)  

Built on top of **NVIDIA’s open Linux drivers (≥ 580)** with a focus on **Wayland environments**,  
GhostWave delivers **studio-grade voice clarity** without hacky workarounds.

---

## 🚀 Features

- 🔊 **AI Noise Cancellation** – Background noise removal in real time  
- 🎛 **Profiles** – Balanced / Streaming / Studio  
- 🖥 **Wayland-Native** – KDE, GNOME, Hyprland, Sway  
- ⚡ **Low-latency DSP** – Optimized for live gaming & calls  
- 🧩 **Integration Options**  
  - Use standalone via **PipeWire module**  
  - Control directly inside [nvcontrol](https://github.com/ghostkellz/nvcontrol)  
  - Combine with [PhantomLink](https://github.com/ghostkellz/phantomlink) as a virtual WaveXLR device  

---

## 🔧 Installation

### Prerequisites
- NVIDIA Open Driver **≥ 580** (Proprietary also supported)
- PipeWire or PulseAudio backend
- CUDA/cuDNN runtime libraries installed

### From Source
```bash
git clone https://github.com/ghostkellz/ghostwave
cd ghostwave
cargo build --release   # or: zig build -Drelease-fast


# 👁️🐧👂 AI Video Clipper & LoRA Captioner (v4.0)

**The ultimate local dataset preparation tool for Video LoRA training (LTX-2, HunyuanVideo).**
*Now featuring Audio Intelligence, 3-Stage Pipeline, and Advanced Bulk Processing.*

---

## ⚡ What's New in v5.0 (GGUF / llama.cpp Update)

* We have made the engine more modular and faster, allowing more agility in onboarding new models.
* Full support for Qwen3-VL, Gemma3, etc.
* Now, any gguf+mmproj pair in appropriate dir structure will be added to the UI.  This allows users to pick quants for themselves to better suit their hardware.
    -note: This dependency does require Windows users to have the Visual C Runtimes
* BRAND NEW DOWNLOAD ARCHITECTURE
    * Now uses a standard, flat and human-readable dir structure compatible w/ other tools and UIs instead of the HuggingFace format
        - unfortunately, requires a one-time download refresh for users currently using transformers repos
    * Multithreaded, multi-connection downloads that don't depend on HuggingFace libs or logins (though we do appreciate their bandwidth)
* Added UI support for making text appears as it is generated
* Now using uv-managed Python for "portable" installs
* Added an "advanced options" shelf/panel ui w/ additional tuning parameters 
* Models with vision support but no video support are now supported through cutting frames w/ ffmpeg and moviepy

---

## 🎯 Core Features

### 1. 🎥 Video Auto-Clipper
Upload a long video (e.g., a vlog or podcast). The AI will:
* Detect speech using **WhisperX** (Word-level precision).
* Cut the video into segments (e.g., 5-7 seconds).
* Analyze the visual content.
* (Optional) Analyze the background audio.
* Save pairs of `.mp4` and `.txt` files automatically.

### 2. 📝 Bulk Video Captioner
Have a folder full of raw clips? Point the app to it.
* Select **Vision**, **Speech**, or **Both**.
* The app will generate descriptions for every video file in the folder.

### 3. 🖼️ Image Folder Captioner
Standard mode for captioning datasets of images using powerful, vision-capable LLMs.

---

## ⚙️ Installation

### Prerequisites
* **NVIDIA GPU** 
    * Minimum 12GB VRAM recommended for vision-only mode
    * 24GB recommended for audio+vision mode
    * MUST have driver support for CUDA 12.8 or higher
* **Windows 10/11** or **Linux**

### 🪟 Windows (One-Click)
1.  Run `install.bat`.
    * *This script uses `uv` to create an isolated, conflict-free environment and installs all dependencies including FFmpeg and Flash Attention.*
2.  Run `Run.bat` to start the app.

### 🐧 Linux / WSL
1.  Run `./install.sh`.
2.  Run `./run.sh`.

### 🐳 Docker / Container
Ideally suited for headless servers or easy deployment.
1.  **Pull the Image**:
    ```bash
    docker pull ghcr.io/cyberbol/ai-video-clipper-lora:latest
    ```
2.  **Run with GPU Support**:
    ```bash
    docker run --gpus all -p 8501:8501 -v $(pwd):/workspace/projects ghcr.io/cyberbol/ai-video-clipper-lora:latest
    ```
    *(Note: Ensure you have the `nvidia-container-toolkit` installed on your host system).*

### ☁️ Cloud / RunPod
For those who prefer processing datasets on high-VRAM cloud GPUs, an illustrated [RunPod Deployment Guide](docs/RUNPOD-HOWTO.md) is available to walk you through the setup.

### 🛠️ Maintainers
For developers needing to support new CUDA versions or custom model architectures, refer to the [Custom Wheel Build Guide](docs/BUILD_WHEELS_HOWTO.md) for instructions on compiling `llama-cpp-python` from source.

---

## ⚠️ Important Notes

* **VRAM Usage:** Enabling "Audio Analysis" downloads an additional ~15GB model (Qwen2-Audio). The process will be slower as models are swapped in and out of GPU memory.
* **Models:** The app automatically downloads models to the `./models` folder.
* **RTX 5090 Support:** Includes patches for Blackwell architecture compatibility.

---

## 🏆 Credits

* **[Cyberbol](https://github.com/cyberbol):** Original Creator & Logic Architect.
* **[FNGarvin](https://github.com/FNGarvin):** Engine Architect (UV & Linux Systems).
* **[WildSpeaker7315](https://www.reddit.com/user/WildSpeaker7315/):** Hardware Research & Fixes.

---
<div align="center">
  <b>Licensed under MIT - Built for the Community</b>

</div>


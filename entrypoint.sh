#!/bin/bash
# FNGarvin - AI Video Clipper & LoRA Captioner Entrypoint
# MIT License 2026

# --- SSH Setup ---
SSH_DIR="/root/.ssh"
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

if [ -n "$PUBLIC_KEY" ]; then
    echo "[INFO] Injecting SSH public key..."
    echo "$PUBLIC_KEY" > "$SSH_DIR/authorized_keys"
    chmod 600 "$SSH_DIR/authorized_keys"
    chown root:root "$SSH_DIR/authorized_keys"
fi

# ==========================================
# GGUF Backend - Dynamic Library Path Fix
# ==========================================
# We must ensure the `llama-cpp-python` backend native libraries can dynamically find the 12.8 CUDA toolkit provided by PyTorch.
# If these paths are missed, the backend will crash on an `str/str` TypeError internal to `load_shared_library()` or core dump.
export NVIDIA_LIBS=$(python3 -c 'import site, os, glob; paths = [glob.glob(os.path.join(p, "nvidia/*/lib")) for p in site.getsitepackages()]; print(":".join([p for sub in paths for p in sub]))' 2>/dev/null)
export LD_LIBRARY_PATH=$NVIDIA_LIBS:$LD_LIBRARY_PATH
echo "[INFO] Injected PyTorch CUDA paths into LD_LIBRARY_PATH."

# Start SSHD
echo "[INFO] Starting SSHD..."
mkdir -p /run/sshd
/usr/sbin/sshd

# --- GPU diagnostics ---
# The llama-cpp-python wheel baked into this image at build time (see
# docs/BUILD_WHEELS_HOWTO.md) already covers every supported CUDA
# architecture, Turing through Blackwell, in one universal build - there
# is no more runtime hot-swap. This is diagnostic-only.
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    echo "[INFO] Detected NVIDIA GPU Compute Capability: $COMPUTE_CAP"
else
    echo "[INFO] No NVIDIA GPU detected via nvidia-smi. Running in CPU/Fallback mode."
fi

# Test hook: stop right after the GPU diagnostics above, before
# Filebrowser and the (long-running, app-serving) exec below. Used by
# tests/test_gpu_detect.sh / CI to verify this logic inside a real
# container against a stubbed nvidia-smi without booting the full app.
if [ "$CI_DETECT_ONLY" = "true" ]; then
    echo "[DETECT-ONLY] Exiting after GPU hot-swap check (CI mode)."
    exit 0
fi

# --- Filebrowser Setup ---
echo "[INFO] Starting Filebrowser on port 8080..."
# Start filebrowser in background
nohup /usr/local/bin/filebrowser --address 0.0.0.0 --port 8080 --root /workspace --noauth &> /filebrowser.log &

# --- Main Application ---
echo "[INFO] Starting main application..."
# Execute the original run.sh script with passed arguments
exec ./run.sh "$@"

#EOF entrypoint.sh

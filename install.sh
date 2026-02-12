#!/bin/bash
# AI Video Clipper & LoRA Captioner - Linux/WSL Installer

# Exit on error
set -e

# UV Optimizations
export UV_HTTP_TIMEOUT=3600
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
export UV_CACHE_DIR="${HOME}/.cache/uv"

echo "======================================================================"
echo "         AI VIDEO CLIPPER & LORA CAPTIONER - INSTALLER (Linux/WSL)"
echo "======================================================================"

# Check for FFmpeg (Linux)
if ! command -v ffmpeg &> /dev/null; then
    echo "[ERROR] FFmpeg is missing!"
    echo "This tool requires FFmpeg to process video/audio."
    echo "Please install it using your package manager, e.g.:"
    echo "  sudo apt update && sudo apt install -y ffmpeg"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "[INFO] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$PATH:$HOME/.cargo/bin:$HOME/.local/bin"
fi

# Argument parsing
RESET_VENV=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reset) RESET_VENV=true ;;
    esac
    shift
done

echo "[STEP 1/3] Preparing Environment..."
if [ "$RESET_VENV" = true ]; then
    if [ -d ".venv" ]; then
        echo "[INFO] Resetting virtual environment as requested..."
        rm -rf .venv
    fi
fi

if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.10 --seed --managed-python --link-mode hardlink
fi
source .venv/bin/activate

# Privacy Configuration (On-the-fly)
if [ ! -f ".streamlit/config.toml" ]; then
    echo "[INFO] Applying privacy settings (Headless Mode + No Analytics)..."
    mkdir -p .streamlit
    cat > .streamlit/config.toml <<EOL
[browser]
gatherUsageStats = false
[server]
headless = true
maxUploadSize = 4096
EOL
fi

echo "[STEP 2/3] Installing Torch Engine (CUDA 12.8)..."
uv pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    --link-mode hardlink \
    "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128"

echo "[STEP 3/3] Installing AI Stack..."
uv pip install \
    --link-mode hardlink \
    "git+https://github.com/m-bain/whisperX.git" --no-deps

echo "[INFO] Syncing GGUF High-Performance Backend (CUDA 12.8)..."
LINUX_WHEEL_URL="https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.0-deps/llama_cpp_python-0.3.23+cu128-cp310-cp310-linux_x86_64.whl"
LINUX_WHEEL_SHA256="8d8546cd067a4cd9d86639519dd4833974cdc4603b28753c5195deef08f406cf"
WHEEL_FILE="llama_cpp_python-0.3.23+cu128-cp310-cp310-linux_x86_64.whl"

echo "[INFO] Downloading wheel for verification..."
curl -L -o "$WHEEL_FILE" "$LINUX_WHEEL_URL"

echo "[INFO] Verifying checksum..."
echo "$LINUX_WHEEL_SHA256  $WHEEL_FILE" | sha256sum -c -

if [ $? -ne 0 ]; then
    echo "[ERROR] Checksum verification failed!"
    rm "$WHEEL_FILE"
    exit 1
fi

echo "[INFO] Checksum verified! Installing..."
uv pip install "$WHEEL_FILE" --force-reinstall
rm "$WHEEL_FILE"


# Fix for ROCm/Linux compatibility or just general stability matching Windows
echo "[INFO] Ensuring correct CTranslate2 - Pinning <4.7.0..."
uv pip install "ctranslate2<4.7.0" --index-url https://pypi.org/simple --force-reinstall

echo "[INFO] Syncing basic dependencies from pyproject.toml..."
uv pip install \
    --link-mode hardlink \
    -r pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu128

echo ""
echo "[STEP 3.5] Installing Audio Intelligence Stack (Qwen2-Audio Support)..."
echo "[INFO] Adding librosa, soundfile and updating transformers..."
uv pip install librosa soundfile numpy --link-mode hardlink
uv pip install --upgrade transformers accelerate huggingface_hub --link-mode hardlink


echo ""
# [Check] GPU Verification
if [ "$SKIP_GPU_CHECK" != "true" ]; then
    echo "[CHECK] Verifying GPU Acceleration (Llama CPP)..."
    .venv/bin/python -c "from llama_cpp import llama_supports_gpu_offload; print(f'>>> GPU Offload Supported: {llama_supports_gpu_offload()}')" || echo "WARNING: Llama check failed"
else
    echo "[INFO] Skipping GPU Verification (Build Mode)"
fi

echo "======================================================================"
echo "Installation complete!"
echo "Run the app with: ./run.sh"


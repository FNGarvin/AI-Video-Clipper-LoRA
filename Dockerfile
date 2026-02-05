# FNGarvin - AI Video Clipper & LoRA Captioner Container
# MIT License 2026

FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_HTTP_TIMEOUT=3600
ENV UV_LINK_MODE=hardlink
ENV UV_CACHE_DIR="/root/.cache/uv"
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

# Copy only necessary files first to leverage build cache
COPY pyproject.toml .

# Create venv and install dependencies
# We use Python 3.10 as requested in install.sh
RUN uv venv .venv --python 3.10
ENV PATH="/workspace/.venv/bin:$PATH"

# Install Torch Engine (matching install.sh)
# Install Torch Engine (matching install.sh)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.10.0+cu128" \
    "torchvision==0.25.0+cu128" \
    "torchaudio==2.10.0+cu128"

# Install AI Stack (WhisperX)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    "git+https://github.com/m-bain/whisperX.git" --no-deps

# Install CTranslate2 pinned (rocm/linux fix)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "ctranslate2<4.7.0" --index-url https://pypi.org/simple --force-reinstall

# Install Audio Intelligence Stack (Qwen2-Audio Support)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install librosa soundfile numpy && \
    uv pip install --upgrade transformers accelerate huggingface_hub

# Sync remaining dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    -r pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu128


# --- MULTI-SERVICE INTEGRATION (Appended to preserve cache) ---

# Install SSHD and Filebrowser
RUN apt-get update && apt-get install -y openssh-server && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash

# Configure SSHD (Key-based only)
RUN mkdir -p /run/sshd && \
    sed -i 's/^#\?PermitRootLogin .*$/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication .*$/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    echo 'ClientAliveInterval 30' >> /etc/ssh/sshd_config && \
    echo 'ClientAliveCountMax 5' >> /etc/ssh/sshd_config

# Copy entrypoint script
COPY entrypoint.sh /workspace/entrypoint.sh

# --- FINAL APP SETUP ---

# Copy the rest of the application
COPY . .

# Apply privacy settings and Runpod Proxy fixes (CORS/XSRF)
RUN mkdir -p .streamlit && \
    echo "[browser]\ngatherUsageStats = false\n[server]\nheadless = true\nmaxUploadSize = 4096\nenableCORS = false\nenableXsrfProtection = false" > .streamlit/config.toml

# Set up local model paths
ENV HF_HOME="/workspace/models"
ENV TORCH_HOME="/workspace/models"
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Make scripts executable
RUN chmod +x run.sh install.sh entrypoint.sh

# Expose Streamlit (8501), Filebrowser (8080), and SSH (22)
EXPOSE 8501 8080 22

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["-h", "0.0.0.0"]

#EOF Dockerfile

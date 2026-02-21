#!/bin/bash
set -ex

# Setup environment
export PATH=/usr/local/cuda-12.8/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export CUDA_PATH=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Install basic build dependencies
apt-get update && apt-get install -y git build-essential cmake ninja-build 

# Prepare output directories
mkdir -p /workspace/output_wheels
cd /workspace
rm -rf /workspace/llama-cpp-python
git clone --recursive https://github.com/JamePeng/llama-cpp-python.git /workspace/llama-cpp-python

# Build configs
COMMON_CMAKE="-DGGML_CUDA=on -DCMAKE_BUILD_TYPE=Release -DLLAMA_AVX512=OFF -DLLAMA_NATIVE=OFF -DCMAKE_BUILD_PARALLEL_LEVEL=16 -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc"
STD_ARCH="-DCMAKE_CUDA_ARCHITECTURES=80;86;89"
BW_ARCH="-DCMAKE_CUDA_ARCHITECTURES=80;86;89;90a"

build_wheel() {
    local py_version=$1
    local py_env="llama_env_${py_version//./}"
    local is_bw=$2

    echo "--- Starting Python ${py_version} Build (Blackwell=${is_bw}) ---"
    rm -rf /workspace/${py_env}
    uv venv --python ${py_version} /workspace/${py_env}
    source /workspace/${py_env}/bin/activate
    uv pip install build scikit-build-core cmake ninja

    if [ "$is_bw" = "true" ]; then
        export CMAKE_ARGS="${COMMON_CMAKE} ${BW_ARCH}"
        suffix="+cu128_Blackwell"
    else
        export CMAKE_ARGS="${COMMON_CMAKE} ${STD_ARCH}"
        suffix="+cu128"
    fi
    export FORCE_CMAKE=1

    cd /workspace/llama-cpp-python
    rm -rf build/
    /workspace/${py_env}/bin/python -m build --wheel --outdir /workspace/output_wheels_tmp --no-isolation
    
    # Rename wheel
    for f in /workspace/output_wheels_tmp/*.whl; do
        if [[ "$f" == *"llama_cpp_python-0.3.26"* ]]; then
            new_name=$(basename $f | sed "s/-0.3.26-/-0.3.26${suffix}-/")
            mv "$f" "/workspace/output_wheels/$new_name"
        fi
    done
    rm -rf /workspace/output_wheels_tmp
}

# 1. Python 3.10 Standard
build_wheel "3.10" "false"
# 2. Python 3.10 Blackwell
build_wheel "3.10" "true"
# 3. Python 3.12 Standard
build_wheel "3.12" "false"
# 4. Python 3.12 Blackwell
build_wheel "3.12" "true"

echo "--- All Builds Finished ---"
ls -la /workspace/output_wheels

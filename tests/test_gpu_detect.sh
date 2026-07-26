#!/bin/bash
# --------------------------------------------------------------------
# Regression test for the GPU detection / llama-cpp-python wheel
# selection logic in install.sh (Linux/WSL install-time path). Mirrors
# tests/test_gpu_detect.bat for Windows Install.bat; see that file and
# issue #13 for the cmd.exe-specific bug this whole effort started from.
#
# Drives the REAL install.sh via its --detect-only hook (resolves
# PY_VER/IS_MODERN_GPU/WHEEL_FILE/LINUX_WHEEL_URL and exits immediately,
# before touching ffmpeg/uv/the venv/network/installs) against a fake
# nvidia-smi (tests/fixtures/nvidia-smi) so it runs in under a second,
# with or without an NVIDIA GPU, and without ffmpeg/uv installed.
#
# Usage: tests/test_gpu_detect.sh
# --------------------------------------------------------------------
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
export PATH="$SCRIPT_DIR/fixtures:$PATH"

PASS_COUNT=0
FAIL_COUNT=0

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.10")
if [ "$PY_VER" == "3.10" ]; then
    STD_WHEEL="llama_cpp_python-0.3.26+cu128-cp310-cp310-linux_x86_64.whl"
    MODERN_WHEEL="llama_cpp_python-0.3.26+cu128_Blackwell-cp310-cp310-linux_x86_64.whl"
else
    STD_WHEEL="llama_cpp_python-0.3.26+cu128-cp312-cp312-linux_x86_64.whl"
    MODERN_WHEEL="$STD_WHEEL"
fi

echo "======================================================================"
echo "  GPU DETECTION REGRESSION TEST (install.sh --detect-only)"
echo "  local python3 resolves to $PY_VER"
echo "======================================================================"

run_case() {
    local label="$1" cap="$2" expect_modern="$3" expect_wheel="$4"
    local out_file
    out_file="$(mktemp)"

    FAKE_COMPUTE_CAP="$cap" bash "$REPO_ROOT/install.sh" --detect-only > "$out_file" 2>&1

    local got_modern got_wheel
    got_modern=$(grep -m1 "IS_MODERN_GPU=" "$out_file" | cut -d= -f2)
    got_wheel=$(grep -m1 "\] WHEEL_FILE=" "$out_file" | cut -d= -f2)

    local failed=""
    if grep -qi "unbound variable\|command not found\|integer expression expected\|Traceback" "$out_file"; then
        failed="1"
    fi
    if [ "$got_modern" != "$expect_modern" ]; then failed="1"; fi
    if [ "$got_wheel" != "$expect_wheel" ]; then failed="1"; fi

    if [ -n "$failed" ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "[FAIL] $label"
        echo "       IS_MODERN_GPU: expected '$expect_modern', got '$got_modern'"
        echo "       WHEEL_FILE: expected '$expect_wheel', got '$got_wheel'"
        echo "       Full output kept at: $out_file"
    else
        PASS_COUNT=$((PASS_COUNT + 1))
        echo "[PASS] $label"
        rm -f "$out_file"
    fi
}

run_case "RTX 5090 (Blackwell, consumer, cap 12.0)"  "12.0" "true"  "$MODERN_WHEEL"
run_case "B100 (Blackwell, datacenter, cap 10.0)"    "10.0" "true"  "$MODERN_WHEEL"
run_case "H100 (Hopper, cap 9.0)"                    "9.0"  "true"  "$MODERN_WHEEL"
run_case "RTX 4090 (Ada, cap 8.9)"                   "8.9"  "false" "$STD_WHEEL"
run_case "RTX 2080 (Turing, cap 7.5)"                "7.5"  "false" "$STD_WHEEL"
run_case "GTX 1080 (Pascal, cap 6.1)"                "6.1"  "false" "$STD_WHEEL"
run_case "nvidia-smi returns garbage (N/A)"          "N/A"  "false" "$STD_WHEEL"
run_case "No NVIDIA GPU / nvidia-smi missing"        "NONE" "false" "$STD_WHEEL"

echo
echo "======================================================================"
echo "  RESULTS: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "======================================================================"

[ "$FAIL_COUNT" -eq 0 ]

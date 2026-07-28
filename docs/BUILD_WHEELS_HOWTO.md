# Building CUDA-Accelerated llama-cpp-python Wheels

This project depends on custom-built `llama-cpp-python` wheels: CUDA 12.8, and
(as of the rebuild described here) a single universal wheel per platform that
covers every supported GPU architecture from Turing through Blackwell. There
is no more "standard" vs "Blackwell" wheel split - one wheel per platform now
works on all of them.

## The bug that started this rewrite

The original hand-built wheels (referenced by the old `v5.0-deps` release)
had a real architecture-selection bug:

* **Windows Blackwell build:** the build never set `-DCMAKE_CUDA_ARCHITECTURES`
  at all. Without it, CMake/nvcc falls back to auto-detecting the
  architecture of whatever GPU is physically installed on the *build*
  machine - not the target. If the machine doing the build had a different
  GPU installed (e.g. an Ada card) at the time, the resulting "Blackwell"
  wheel would silently contain kernels for the wrong architecture.
* **Linux Blackwell instructions (below, historical):** told you to pass
  `-DCMAKE_CUDA_ARCHITECTURES=90`. That's Hopper (H100), not Blackwell -
  Blackwell is `sm_100`/`sm_101` (datacenter) and `sm_120` (RTX 50-series).
* **The actual build automation** (`scripts/build_universal_linux.sh`, now
  removed): its `BW_ARCH` variable was `-DCMAKE_CUDA_ARCHITECTURES=80;86;89;90a`
  - Ampere/Ada plus Hopper's architecture-specific `90a`, but **still no
  `100`/`101`/`120`**. The currently-shipped `_Blackwell`-suffixed wheels
  were built from this script, so they most likely never contained real
  Blackwell (sm_120) kernels at all - this is almost certainly the actual
  cause of the RTX 5090 misbehavior that prompted this whole rebuild.

Building in CI on a GPU-less runner makes this whole bug class structurally
impossible: there's no local GPU to auto-detect off of, and the architecture
list is spelled out explicitly in the workflow. If you ever forget to set it,
the build fails loudly instead of silently miscompiling.

---

## Primary path: the CI workflow (recommended)

`.github/workflows/build-llama-wheels.yml` builds all 3 wheels this repo
needs, on demand (`workflow_dispatch` only - it never runs automatically):

| Wheel | Platform | Python | Used by |
|---|---|---|---|
| Windows | `win_amd64` | 3.10 | `Install.bat` |
| Linux (native) | `linux_x86_64` | 3.10 | `install.sh` |
| Linux (container) | `linux_x86_64` | 3.12 | `Dockerfile` (matches the `pytorch/pytorch` base image's system Python) |

To run it:
```powershell
gh workflow run build-llama-wheels.yml -f release_tag=v5.4-llama-deps
```

(The first real run of this workflow published `v5.3-llama-deps` on the old
v0.3.26 pin - all 3 wheels built successfully; the Linux legs finished in
~1-1.5h each, the Windows leg took ~3.5h because that recipe fell back to
MSBuild instead of Ninja. The `v5.4-llama-deps` bump to v0.3.44 switches the
Windows leg to Ninja + Clang/LLVM specifically to fix that, alongside picking
up current Gemma/Qwen model support - see [Source pin](#source-pin) below.)
Each wheel gets a [SLSA build provenance
attestation](https://docs.github.com/actions/security-for-github-actions/using-artifact-attestations)
via `actions/attest-build-provenance`, binding it to the exact workflow run
and source commit that produced it. All 3 wheels + a `SHA256SUMS.txt` land as
assets on a **prerelease** (so it doesn't interfere with "latest" tagging,
matching how `v5.0-deps` was published).

To verify a downloaded wheel's provenance:
```powershell
gh attestation verify llama_cpp_python-0.3.44+cu128-cp310-cp310-win_amd64.whl --owner cyberbol
```

### Source pin

The workflow builds from the [JamePeng
fork](https://github.com/JamePeng/llama-cpp-python) - see the Acknowledgements
in the main README - pinned to an **immutable commit SHA**, not a branch or
floating tag:

```yaml
LLAMA_CPP_PYTHON_LINUX_REF: "ebf6099b81cf67cfb5eec569466367c9fa04e9d4"
LLAMA_CPP_PYTHON_WIN_REF:   "ebf6099b81cf67cfb5eec569466367c9fa04e9d4"
```

This is JamePeng's own tagged v0.3.44 cu128 release commit - as of this pin
both platform tags (`v0.3.44-cu128-linux-20260721` /
`v0.3.44-cu128-win-20260721`) happen to point at the same commit. Bumped from
the previous v0.3.26 pin specifically to pick up current Gemma/Qwen model
support (Gemma 4 chat handling landed v0.3.35/v0.3.36; Qwen3.5/Qwen3.6/
Qwen3-Next landed v0.3.36/v0.3.39 - both well before this pin, this was just
the first bump that happened to land after them).

JamePeng tags a release per (version, CUDA version, platform), so to bump
the llama-cpp-python version again, find the new commit with:

```bash
git ls-remote --tags https://github.com/JamePeng/llama-cpp-python.git | grep "cu128"
```

and update both `LLAMA_CPP_PYTHON_*_REF` values in the workflow - **do not
just swap the hash**. Also fetch JamePeng's own workflow YAML *at that exact
commit* (not their `main` branch - the recipe drifts over time):

`https://raw.githubusercontent.com/JamePeng/llama-cpp-python/<COMMIT_SHA>/.github/workflows/build-wheels-cu128-{linux,win}.yml`

and diff it against what's currently in `build-llama-wheels.yml`. The
v0.3.26 -> v0.3.44 bump changed the recipe substantially, not just flags:
CPU dispatch moved from a single baked-in "Basic" build to
`GGML_BACKEND_DL`/`GGML_CPU_ALL_VARIANTS` (runtime-selected CPU backends),
and the Windows leg moved off MSVC/MSBuild onto Ninja Multi-Config +
Clang/LLVM via a toolchain file that ships inside the llama.cpp submodule
itself (`vendor/llama.cpp/cmake/x64-windows-llvm.cmake`) - JamePeng's stated
reason is that MSVC skips several GGML CPU all-variant backends. Don't
assume a version bump is just a hash swap; port whatever the new recipe
actually does.

### After a run completes

Download the new wheels' SHA256 hashes from the release's `SHA256SUMS.txt`
(don't hand-transcribe them off scrolling/truncated log output - it's easy to
mistype a hex digit) and update the pinned URLs/hashes in `Install.bat` and
`install.sh`. `entrypoint.sh` no longer references a wheel at all - the
image's baked-in wheel already covers every architecture.

---

## Manual/local build (fallback only)

Only needed if you're iterating on a build locally and don't want to wait on
CI. **Always pass an explicit, correct `CMAKE_CUDA_ARCHITECTURES` list** -
never leave it unset, and never use `=90` expecting it to mean Blackwell.

### Prerequisites

* **NVIDIA CUDA Toolkit 12.8.1**, `nvcc` on PATH.
* **C++ Compiler:** Windows: [Visual Studio 2022 Community with C++
  workloads](https://visualstudio.microsoft.com/visual-cpp-build-tools/),
  plus the **Clang/LLVM** optional component (VS2022 installer -> Individual
  Components -> "C++ Clang Compiler for Windows") and **Ninja** - the
  current pinned recipe builds via `Ninja Multi-Config` + Clang, not MSBuild,
  using a toolchain file that ships inside the llama.cpp submodule
  (`vendor/llama.cpp/cmake/x64-windows-llvm.cmake`).
  Linux: `gcc`, `g++`, `cmake`, `ninja-build`.
* **Python:** 3.10 or 3.12, matching the target wheel.

### Windows (Developer PowerShell for VS 2022)

```powershell
git clone https://github.com/JamePeng/llama-cpp-python.git
cd llama-cpp-python
git checkout ebf6099b81cf67cfb5eec569466367c9fa04e9d4
git submodule update --init --recursive

uv venv .venv --python 3.10 --seed
.\.venv\Scripts\Activate.ps1
uv pip install --upgrade build setuptools wheel packaging ninja

# Run "Setup MSVC environment for nvcc" first (Developer PowerShell already
# does this) - CUDA still needs an MSVC host compiler even though GGML
# itself builds with Clang.
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:VERBOSE = "1"
$env:CMAKE_GENERATOR = "Ninja Multi-Config"

# Turing through Blackwell (consumer + datacenter). Dynamic GGML backends
# (GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS) instead of one baked-in "Basic"
# CPU build - see the matching comment in build-llama-wheels.yml.
$env:CMAKE_ARGS = "-DCMAKE_TOOLCHAIN_FILE=vendor/llama.cpp/cmake/x64-windows-llvm.cmake -DLLAMA_BUILD_BORINGSSL=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF -DGGML_CPU=ON -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;87-real;89-real;90-real;100-real;120-real -DGGML_CUDA_FORCE_MMQ=ON -DCUDA_SEPARABLE_COMPILATION=ON -DCMAKE_CUDA_FLAGS=--diag-suppress=177,221,550 -DENABLE_CCACHE=ON"

python -m build --wheel
# Rename dist\llama_cpp_python-0.3.44-*.whl -> ...+cu128-*.whl to match the pin in Install.bat
```

### Linux

```bash
git clone https://github.com/JamePeng/llama-cpp-python.git
cd llama-cpp-python
git checkout ebf6099b81cf67cfb5eec569466367c9fa04e9d4
git submodule update --init --recursive

uv venv .venv --python 3.10 --seed   # or 3.12 for the container wheel
source .venv/bin/activate
uv pip install --upgrade build setuptools wheel packaging

export CUDA_HOME=/usr/local/cuda
export VERBOSE=1
export CMAKE_ARGS="-G Ninja -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=ON -DGGML_CPU=ON -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES='75-real;80-real;86-real;87-real;89-real;90-real;100-real;120-real' -DGGML_CUDA_FORCE_MMQ=ON -DCUDA_SEPARABLE_COMPILATION=ON -DCMAKE_CUDA_FLAGS=--diag-suppress=177,221,550 -DENABLE_CCACHE=ON"

python -m build --wheel
# Rename dist/llama_cpp_python-0.3.44-*.whl -> ...+cu128-*.whl to match the pin in install.sh/entrypoint.sh
```

---

## Deployment

1. Trigger `build-llama-wheels.yml` (or build manually per above).
2. Verify the wheels' attestations (`gh attestation verify ...`) and, ideally,
   smoke-test the Blackwell path on real Blackwell hardware before trusting it.
3. Update the pinned URLs/SHA256 hashes in `Install.bat` and `install.sh` to
   point at the new release assets (copy hashes straight from the release's
   `SHA256SUMS.txt`, not from log output).

// END OF BUILD_WHEELS_HOWTO.md

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
gh workflow run build-llama-wheels.yml -f release_tag=v5.3-llama-deps
```

(The first real run of this workflow published `v5.3-llama-deps` - all 3 wheels
built successfully; the Linux legs finished in ~1-1.5h each, the Windows leg
took ~3.5h, most likely because the pinned recipe below doesn't force the
Ninja generator on Windows and falls back to MSBuild. Slow, but correct, and
comfortably inside GitHub's 6-hour job timeout.)
Each wheel gets a [SLSA build provenance
attestation](https://docs.github.com/actions/security-for-github-actions/using-artifact-attestations)
via `actions/attest-build-provenance`, binding it to the exact workflow run
and source commit that produced it. All 3 wheels + a `SHA256SUMS.txt` land as
assets on a **prerelease** (so it doesn't interfere with "latest" tagging,
matching how `v5.0-deps` was published).

To verify a downloaded wheel's provenance:
```powershell
gh attestation verify llama_cpp_python-0.3.26+cu128-cp310-cp310-win_amd64.whl --owner cyberbol
```

### Source pin

The workflow builds from the [JamePeng
fork](https://github.com/JamePeng/llama-cpp-python) - see the Acknowledgements
in the main README - pinned to **immutable commit SHAs**, not a branch or
floating tag:

```yaml
LLAMA_CPP_PYTHON_LINUX_REF: "32f2380ec8ebfa0d5f01c22e3ba86d8d5e762882"
LLAMA_CPP_PYTHON_WIN_REF:   "3d0fd1b75ee564361a4babf21f88855225ba1fe0"
```

These are JamePeng's own tagged v0.3.26 cu128 release commits
(`v0.3.26-cu128-Basic-linux-20260219` / `v0.3.26-cu128-Basic-win-20260220`).
This is intentionally **not** an upgrade - it rebuilds the exact version this
project already ships, just correctly and with attestation. JamePeng tags a
release per (version, CUDA version, platform), so if you ever do want to bump
the llama-cpp-python version, find the new commit with:

```bash
git ls-remote --tags https://github.com/JamePeng/llama-cpp-python.git | grep "cu128"
```

and update both `LLAMA_CPP_PYTHON_*_REF` values in the workflow - along with
a quick check that the CMake flags in the workflow still match whatever
JamePeng's own workflow used *at that commit* (fetch
`https://raw.githubusercontent.com/JamePeng/llama-cpp-python/<COMMIT_SHA>/.github/workflows/build-wheels-cu128-{linux,win}.yml`,
not their `main` branch - the build recipe has changed over time, and a
newer recipe isn't guaranteed to apply cleanly to an older pinned commit).

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
  workloads](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
  Linux: `gcc`, `g++`, `cmake`.
* **Python:** 3.10 or 3.12, matching the target wheel.

### Windows (Developer PowerShell for VS 2022)

```powershell
git clone https://github.com/JamePeng/llama-cpp-python.git
cd llama-cpp-python
git checkout 3d0fd1b75ee564361a4babf21f88855225ba1fe0
git submodule update --init --recursive

uv venv .venv --python 3.10 --seed
.\.venv\Scripts\Activate.ps1
uv pip install --upgrade build setuptools wheel packaging ninja

$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:VERBOSE = "1"

# Turing through Blackwell (consumer + datacenter), "Basic" CPU baseline
# (no AVX/AVX2 - GPU offload is what matters here, not CPU SIMD).
$env:CMAKE_ARGS = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;87-real;89-real;90-real;100-real;101-real;120-real -DGGML_CUDA_FORCE_MMQ=on -DCUDA_SEPARABLE_COMPILATION=on -DENABLE_CCACHE=on -DLLAMA_CURL=off -DGGML_NATIVE=off -DGGML_AVX=off -DGGML_AVX2=off -DGGML_AVX_VNNI=off -DGGML_AVX512=off -DGGML_AVX512_VBMI=off -DGGML_AVX512_VNNI=off -DGGML_AVX512_BF16=off -DGGML_FMA=off -DGGML_F16C=off"

python -m build --wheel
# Rename dist\llama_cpp_python-0.3.26-*.whl -> ...+cu128-*.whl to match the pin in Install.bat
```

### Linux

```bash
git clone https://github.com/JamePeng/llama-cpp-python.git
cd llama-cpp-python
git checkout 32f2380ec8ebfa0d5f01c22e3ba86d8d5e762882
git submodule update --init --recursive

uv venv .venv --python 3.10 --seed   # or 3.12 for the container wheel
source .venv/bin/activate
uv pip install --upgrade build setuptools wheel packaging

export CUDA_HOME=/usr/local/cuda
export VERBOSE=1
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES='75-real;80-real;86-real;87-real;89-real;90-real;100-real;101-real;120-real' -DGGML_CUDA_FORCE_MMQ=on -DLLAMA_CURL=off -DLLAMA_OPENSSL=on -DGGML_NATIVE=off -DGGML_AVX=off -DGGML_AVX2=off -DGGML_AVX_VNNI=off -DGGML_AVX512=off -DGGML_AVX512_VBMI=off -DGGML_AVX512_VNNI=off -DGGML_AVX512_BF16=off -DGGML_FMA=off -DGGML_F16C=off"

python -m build --wheel
# Rename dist/llama_cpp_python-0.3.26-*.whl -> ...+cu128-*.whl to match the pin in install.sh/entrypoint.sh
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

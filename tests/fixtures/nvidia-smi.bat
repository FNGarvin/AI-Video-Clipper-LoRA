@echo off
REM Fake nvidia-smi for tests\test_gpu_detect.bat. Emulates just enough of
REM `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` for the GPU
REM detection logic in Install.bat to be tested without real GPU hardware.
REM
REM Controlled entirely via the FAKE_COMPUTE_CAP env var set by the test
REM harness: set it to "NONE" to emulate no NVIDIA GPU present (nvidia-smi
REM missing/erroring), or to a compute_cap string (e.g. "12.0", "8.9", "N/A")
REM to emulate nvidia-smi's real stdout for a given GPU.
if "%FAKE_COMPUTE_CAP%"=="NONE" exit /b 9009
echo %FAKE_COMPUTE_CAP%
exit /b 0

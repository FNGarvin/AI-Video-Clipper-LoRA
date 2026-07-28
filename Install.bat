@echo off
setlocal EnableDelayedExpansion
:: AI Video Clipper & LoRA Captioner - Windows Installer (v5.0 Staging)

TITLE AI Clipper Installer - UV Edition
color 0B

:: UV Optimizations
set UV_HTTP_TIMEOUT=3600
set UV_LINK_MODE=hardlink
set UV_CACHE_DIR=%USERPROFILE%\.cache\uv

echo ======================================================================
echo          AI VIDEO CLIPPER ^& LORA CAPTIONER - INSTALLER
echo ======================================================================
echo.

:: Check for uv
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] uv not found. Installing via winget...
    winget install astral-sh.uv --accept-source-agreements --accept-package-agreements
    
    REM SEARCH STRATEGY:
    REM 1. Standard Winget Links (Symlinks)
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" (
        set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
    )
    
    REM 2. Dynamic Winget Package Folder (finding the folder ending in ...uv_... source)
    for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv*") do (
        if exist "%%D\uv.exe" set "PATH=%%D;%PATH%"
    )
    
    REM 3. Cargo Bin (fallback)
    if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    )
    
    REM 4. User-specific valid path from previous runs?
    if exist "%LOCALAPPDATA%\uv\uv.exe" (
         set "PATH=%LOCALAPPDATA%\uv;%PATH%"
    )

    REM Verify installation
    uv --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] uv installed but still not found in PATH for this session.
        echo [INFO] Detected location via 'where uv' might require a restart.
        echo [INFO] Please close this window and run install.bat again.
        pause
        exit /b 0
    )
)

:: Argument parsing
set "RESET_VENV=false"
set "NO_PAUSE=false"
set "DETECT_ONLY=false"

:parse_args
if "%~1"=="" goto end_parse_args
if /i "%~1"=="--reset" set "RESET_VENV=true"
if /i "%~1"=="--no-pause" set "NO_PAUSE=true"
if /i "%~1"=="--detect-only" set "DETECT_ONLY=true"
shift
goto parse_args
:end_parse_args

:: Test hook: resolve the wheel to fetch and exit immediately, before
:: touching the venv, network, or any installs. Used by tests\test_gpu_detect.bat
:: to verify this doesn't crash cmd.exe's parser (see issue #13).
if not "%DETECT_ONLY%"=="true" goto after_detect_only
call :resolve_wheel
echo [DETECT-ONLY] WHEEL_FILE=%WHEEL_FILE%
echo [DETECT-ONLY] WIN_WHEEL_URL=%WIN_WHEEL_URL%
echo [DETECT-ONLY] WIN_WHEEL_SHA256=%WIN_WHEEL_SHA256%
exit /b 0
:after_detect_only

echo.
echo [STEP 1/3] Preparing isolated environment (uv)...

if "%RESET_VENV%"=="true" (
    if exist ".venv" (
        echo [INFO] Resetting virtual environment as requested...
        rd /s /q ".venv"
    )
)

if not exist ".venv" (
    uv venv .venv --python 3.10 --seed --managed-python --link-mode hardlink
)

:: --------------------------------------------------------------------
:: [STEP 1.5] FFmpeg Installation & Path Refresh
:: --------------------------------------------------------------------
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] FFmpeg not found. Installing via winget...
    winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    
    REM ATTEMPT DYNAMIC PATH REFRESH
    REM 1. Standard Winget Links (Symlinks)
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" (
        set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
        echo [INFO] Added Winget Links to PATH for this session.
    )

    REM 2. Robust Search for Gyan.FFmpeg Package (Deep Search)
    REM User Path: %LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_...\ffmpeg-*-full_build\bin
    for /d %%P in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*") do (
        echo [INFO] Found Legacy Package: %%~nxP
        for /d %%B in ("%%P\ffmpeg-*-full_build") do (
             if exist "%%B\bin\ffmpeg.exe" (
                 set "PATH=%%B\bin;%PATH%"
                 echo [INFO] Added Deep FFmpeg Path: %%B\bin
             )
        )
    )
    
    REM Verify
    where ffmpeg >nul 2>&1
    if %errorlevel% neq 0 (
        echo [WARNING] FFmpeg installed but not detected in current session.
        echo [IMPORTANT] You may need to RESTART your terminal/PC before running the app.
    ) else (
        echo [SUCCESS] FFmpeg detected!
    )
)

:: Privacy Configuration (On-the-fly)
if not exist ".streamlit\config.toml" (
    echo [INFO] Applying privacy settings ^(Headless Mode + No Analytics^)...
    if not exist ".streamlit" mkdir .streamlit
    (
        echo [browser]
        echo gatherUsageStats = false
        echo.
        echo [server]
        echo headless = true
        echo maxUploadSize = 4096
    ) > .streamlit\config.toml
)

echo .
echo [STEP 2/3] Installing Torch Engine (CUDA 12.8)...
call .venv\Scripts\activate.bat
uv pip install ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    --link-mode hardlink ^
    "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128"

echo [INFO] Syncing GGUF High-Performance Backend (CUDA 12.8)...
call :resolve_wheel

echo [INFO] Downloading wheel for verification...
curl -L -o "%WHEEL_FILE%" "%WIN_WHEEL_URL%"
if %errorlevel% equ 35 (
    echo.
    echo [WARNING] Standard download failed: TLS certificate revocation check error.
    echo ======================================================================
    echo   SECURITY NOTICE: RETRYING DOWNLOAD WITH REVOCATION CHECK DISABLED
    echo ======================================================================
    echo   This is common on school/corporate networks and VPNs that intercept
    echo   or block certificate revocation checks - it does NOT mean this file
    echo   or your connection is compromised.
    echo.
    echo   The download will still be verified against a SHA256 checksum
    echo   pinned in this script before anything is installed. If that check
    echo   fails, installation stops immediately.
    echo ======================================================================
    echo.
    curl -L --ssl-no-revoke -o "%WHEEL_FILE%" "%WIN_WHEEL_URL%"
)
if %errorlevel% neq 0 (
    echo [ERROR] Download failed.
    pause
    exit /b 1
)

echo [INFO] Verifying checksum...
certutil -hashfile "%WHEEL_FILE%" SHA256 | findstr /i "%WIN_WHEEL_SHA256%" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Checksum verification failed!
    del "%WHEEL_FILE%"
    pause
    exit /b 1
)

echo [SUCCESS] Checksum verified! Installing...
uv pip install "%WHEEL_FILE%" --force-reinstall
del "%WHEEL_FILE%"


echo.
echo [STEP 3/3] Installing AI Stack...
uv pip install "git+https://github.com/m-bain/whisperX.git@6ec4a020489d904c4f2cd1ed097674232d2692d4" --no-deps --link-mode hardlink

echo [INFO] Ensuring correct CTranslate2 (Windows) - Pinning <4.7.0 to avoid ROCm bug...
uv pip install "ctranslate2<4.7.0" --index-url https://pypi.org/simple --force-reinstall

echo [INFO] Syncing remaining dependencies from pyproject.toml...
uv pip install -r pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu128 --link-mode hardlink

:: --- NOWA SEKCJA v4.0 ---
echo.
echo [STEP 3.5] Installing Audio Intelligence Stack (Qwen2-Audio Support)...
echo [INFO] Adding librosa, soundfile and updating transformers...
uv pip install librosa soundfile numpy --link-mode hardlink
uv pip install transformers accelerate huggingface_hub --link-mode hardlink
:: ------------------------

echo.
echo [CHECK] Verifying GPU Acceleration...
call .venv\Scripts\python -c "from llama_cpp import llama_supports_gpu_offload; print(f'>>> GPU Offload Supported: {llama_supports_gpu_offload()}')"

echo.
echo ======================================================================
echo                    INSTALLATION COMPLETE!
echo ======================================================================
echo You can now run the program using "Run.bat".
echo.
echo.
if "%NO_PAUSE%"=="false" pause
exit /b 0

:: --------------------------------------------------------------------
:: llama-cpp-python wheel resolution.
::
:: As of the wheel rebuild in docs/BUILD_WHEELS_HOWTO.md, a single wheel
:: covers every supported CUDA architecture (Turing through Blackwell) -
:: there is no more per-GPU-family branching here. The compute-cap probe
:: below is diagnostic-only (printed for troubleshooting) and does not
:: affect which wheel gets fetched.
::
:: Written with only single-line "if" statements - no multi-line
:: parenthesized if/else blocks. cmd.exe's parser miscounts parentheses
:: that appear in echoed text *inside* a parenthesized block (even
:: balanced ones), which is what broke GPU detection before:
:: https://github.com/cyberbol/AI-Video-Clipper-LoRA/issues/13
:: Keeping this routine block-free avoids that entire bug class.
::
:: Sets: WHEEL_FILE, WIN_WHEEL_URL, WIN_WHEEL_SHA256
:: --------------------------------------------------------------------
:resolve_wheel
set "WIN_WHEEL_URL=https://github.com/cyberbol/AI-Video-Clipper-LoRA/releases/download/v5.3-llama-deps/llama_cpp_python-0.3.26+cu128-cp310-cp310-win_amd64.whl"
set "WIN_WHEEL_SHA256=f3e8512dd6e80f847189420bdeba657cc45d38d42cf025e2bababaa9f5188013"
set "WHEEL_FILE=llama_cpp_python-0.3.26+cu128-cp310-cp310-win_amd64.whl"

set "MAJOR_CAP="
for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu=compute_cap --format=csv^,noheader 2^>nul') do set "MAJOR_CAP=%%a"
if defined MAJOR_CAP echo [INFO] Detected NVIDIA GPU Compute Capability Major: %MAJOR_CAP%
goto :eof

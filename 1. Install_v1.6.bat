@echo off
TITLE AI Clipper Installer v2.3 (Stable & Blackwell)
color 0B

echo ======================================================================
echo       AI VIDEO CLIPPER ^& LORA CAPTIONER - INSTALLER v2.3
echo             Optimized for Stability ^& RTX 5090 Support
echo ======================================================================
echo.

:MENU
echo ======================================================================
echo    SELECT YOUR GRAPHICS CARD (GPU)
echo ======================================================================
echo.
echo  [1] STANDARD (RTX 4090, 3090, 20xx)
echo      - Installs PyTorch Stable (CUDA 12.6)
echo.
echo  [2] NEXT-GEN (RTX 5090 / Blackwell)
echo      - Installs PyTorch Nightly (CUDA 13.0)
echo.
set /p gpu_choice="Type 1 or 2 and press ENTER: "

if "%gpu_choice%"=="1" goto SETUP_STABLE
if "%gpu_choice%"=="2" goto SETUP_EXPERIMENTAL
goto MENU

:SETUP_STABLE
cls
color 0A
echo [INFO] Selected: STANDARD MODE (Stable / CUDA 12.6)
:: Usunalem sztywne wersje torchvision/audio - pip dobierze je sam idealnie pod torch 2.8.0
set "TORCH_CMD=pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
goto START_INSTALL

:SETUP_EXPERIMENTAL
cls
color 0D
echo [WARNING] Selected: RTX 5090 MODE (Nightly / CUDA 13.0)
:: Dla 5090 uzywamy najnowszego kanalu nightly dla CUDA 13.0
set "TORCH_CMD=pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130"
goto START_INSTALL

:START_INSTALL
echo.
echo [STEP 1/5] Creating isolated environment (venv)...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [STEP 2/5] Upgrading PIP...
python -m pip install --upgrade pip

echo.
echo [STEP 3/5] Installing GPU Engine...
:: Najpierw fundament: PyTorch
%TORCH_CMD%

echo.
echo [STEP 4/5] Installing WhisperX ^& Vision Components...
:: Instalujemy WhisperX oraz wszystkie zaleznosci w jednym kroku, 
:: aby pip mogl rozwiazac konflikty miedzy pyannote a reszta.
pip install git+https://github.com/m-bain/whisperX.git qwen-vl-utils accelerate transformers streamlit moviepy "pillow<11.0"

echo.
echo [STEP 5/5] Final GPU Engine Sync...
:: Upewniamy sie, ze poprzedni krok nie podmienil nam Torcha na starszy
%TORCH_CMD%

echo.
echo ======================================================================
echo                    INSTALLATION COMPLETE!
echo ======================================================================
echo You can now run the program using "3. Run.bat".
echo.
pause
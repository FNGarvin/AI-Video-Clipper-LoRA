@echo off
setlocal EnableDelayedExpansion

:: --------------------------------------------------------------------
:: Regression test for the GPU detection / llama-cpp-python wheel
:: selection logic in Install.bat (see issue #13: RTX 5090/Blackwell was
:: never getting the Blackwell wheel, and a later fix attempt crashed
:: cmd.exe's parser outright with ". was unexpected at this time.").
::
:: This drives the REAL Install.bat via its "--detect-only" hook (which
:: resolves IS_MODERN_GPU/WHEEL_FILE/WIN_WHEEL_URL and exits immediately,
:: before touching the venv, network, or doing any installs) against a
:: fake nvidia-smi.bat (tests\fixtures\nvidia-smi.bat) so it can be run
:: on any machine, with or without an NVIDIA GPU, in a few seconds.
::
:: Usage: tests\test_gpu_detect.bat
:: Requires: uv on PATH (Install.bat checks for it before anything else).
:: --------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
set "PATH=%SCRIPT_DIR%fixtures;%PATH%"

set "PASS_COUNT=0"
set "FAIL_COUNT=0"

set "STD_WHEEL=llama_cpp_python-0.3.26+cu128-cp310-cp310-win_amd64.whl"
set "MODERN_WHEEL=llama_cpp_python-0.3.26+cu128_Blackwell-cp310-cp310-win_amd64.whl"

echo ======================================================================
echo   GPU DETECTION REGRESSION TEST (Install.bat --detect-only)
echo ======================================================================

call :run_case "RTX 5090 (Blackwell, consumer, cap 12.0)"  "12.0" "true"  "!MODERN_WHEEL!"
call :run_case "B100 (Blackwell, datacenter, cap 10.0)"     "10.0" "true"  "!MODERN_WHEEL!"
call :run_case "H100 (Hopper, cap 9.0)"                     "9.0"  "true"  "!MODERN_WHEEL!"
call :run_case "RTX 4090 (Ada, cap 8.9)"                    "8.9"  "false" "!STD_WHEEL!"
call :run_case "RTX 2080 (Turing, cap 7.5)"                 "7.5"  "false" "!STD_WHEEL!"
call :run_case "GTX 1080 (Pascal, cap 6.1)"                 "6.1"  "false" "!STD_WHEEL!"
call :run_case "nvidia-smi returns garbage (N/A)"           "N/A"  "false" "!STD_WHEEL!"
call :run_case "No NVIDIA GPU / nvidia-smi missing"         "NONE" "false" "!STD_WHEEL!"

echo.
echo ======================================================================
echo   RESULTS: !PASS_COUNT! passed, !FAIL_COUNT! failed
echo ======================================================================

if !FAIL_COUNT! GTR 0 exit /b 1
exit /b 0

:: ---------------------------------------------------------------
:: %1 = human label, %2 = FAKE_COMPUTE_CAP value, %3 = expected
:: IS_MODERN_GPU (true/false), %4 = expected WHEEL_FILE
:: ---------------------------------------------------------------
:run_case
set "CASE_LABEL=%~1"
set "FAKE_COMPUTE_CAP=%~2"
set "EXPECT_MODERN=%~3"
set "EXPECT_WHEEL=%~4"

set "OUT_FILE=%TEMP%\gpu_detect_test_%RANDOM%.log"
call "%REPO_ROOT%\Install.bat" --detect-only --no-pause > "%OUT_FILE%" 2>&1

set "GOT_MODERN="
set "GOT_WHEEL="
for /f "tokens=2 delims==" %%v in ('findstr /c:"IS_MODERN_GPU=" "%OUT_FILE%"') do set "GOT_MODERN=%%v"
for /f "tokens=2 delims==" %%v in ('findstr /c:"] WHEEL_FILE=" "%OUT_FILE%"') do set "GOT_WHEEL=%%v"

set "PARSE_ERROR="
findstr /c:"was unexpected at this time" "%OUT_FILE%" >nul
if not errorlevel 1 set "PARSE_ERROR=1"
findstr /c:"is not recognized as an internal or external command" "%OUT_FILE%" >nul
if not errorlevel 1 set "PARSE_ERROR=1"
findstr /c:"The syntax of the command is incorrect" "%OUT_FILE%" >nul
if not errorlevel 1 set "PARSE_ERROR=1"

set "CASE_FAILED="
if defined PARSE_ERROR set "CASE_FAILED=1"
if not "!GOT_MODERN!"=="!EXPECT_MODERN!" set "CASE_FAILED=1"
if not "!GOT_WHEEL!"=="!EXPECT_WHEEL!" set "CASE_FAILED=1"

if defined CASE_FAILED goto case_fail

set /a PASS_COUNT+=1
echo [PASS] !CASE_LABEL!
del "%OUT_FILE%" >nul 2>&1
goto :eof

:case_fail
set /a FAIL_COUNT+=1
echo [FAIL] !CASE_LABEL!
if defined PARSE_ERROR echo        cmd.exe parser error detected in output - see !OUT_FILE!
if not "!GOT_MODERN!"=="!EXPECT_MODERN!" echo        IS_MODERN_GPU: expected "!EXPECT_MODERN!", got "!GOT_MODERN!"
if not "!GOT_WHEEL!"=="!EXPECT_WHEEL!" echo        WHEEL_FILE: expected "!EXPECT_WHEEL!", got "!GOT_WHEEL!"
echo        Full output kept at: !OUT_FILE!
goto :eof

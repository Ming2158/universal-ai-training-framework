@echo off
setlocal
cd /d "%~dp0"
title UATF Local Training UI

set "PYTHON=venv\Scripts\python.exe"
set "CUDA_PYTHON=.venv\Scripts\python.exe"
set "PYVER="

if exist "%CUDA_PYTHON%" (
    "%CUDA_PYTHON%" -c "import gradio, torch, torchvision, PIL, yaml, matplotlib; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON=%CUDA_PYTHON%"
)

if exist "%PYTHON%" (
    "%PYTHON%" --version >nul 2>&1
    if errorlevel 1 goto recreate_venv
    goto check_dependencies
)

:create_venv
call :find_python
if errorlevel 1 goto finish
echo Creating the UATF Python environment...
py -%PYVER% -m venv venv
if errorlevel 1 goto setup_failed
goto check_dependencies

:recreate_venv
call :find_python
if errorlevel 1 goto finish
echo Recreating the invalid UATF Python environment...
py -%PYVER% -m venv --clear venv
if errorlevel 1 goto setup_failed

:check_dependencies
"%PYTHON%" -c "import gradio, torch, torchvision, PIL, yaml, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Installing CPU dependencies. This may take several minutes...
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto setup_failed
    "%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto setup_failed
    "%PYTHON%" -m pip install -e .
    if errorlevel 1 goto setup_failed
)

echo Starting UATF. Close this window or press Ctrl+C to stop it.
"%PYTHON%" -m app.app
if errorlevel 1 echo UATF stopped with an error. Review the messages above.
goto finish

:find_python
where py >nul 2>&1
if errorlevel 1 (
    echo Python Launcher was not found. Install Python 3.10, 3.11, or 3.12, then run this file again.
    start "" "https://www.python.org/downloads/windows/"
    exit /b 1
)

py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PYVER=3.12"
if defined PYVER exit /b 0
py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYVER=3.11"
if defined PYVER exit /b 0
py -3.10 --version >nul 2>&1
if not errorlevel 1 set "PYVER=3.10"
if defined PYVER exit /b 0

echo Python 3.10, 3.11, or 3.12 was not found. Install one, then run this file again.
start "" "https://www.python.org/downloads/windows/"
exit /b 1

:setup_failed
echo Setup failed. Check your internet connection and the error above, then run this file again.

:finish
echo.
pause
endlocal

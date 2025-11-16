@echo off
cls
echo ======================================================
echo    AI Image Detection System - TensorFlow Setup
echo ======================================================
echo.

echo This script will help you enable long path support and install TensorFlow.
echo.

:MENU
echo Please select an option:
echo 1. Enable Long Path Support (requires Administrator)
echo 2. Install TensorFlow CPU (shorter paths)
echo 3. Create Virtual Environment in Short Path
echo 4. Exit
echo.
choice /c 1234 /m "Select an option"

if errorlevel 4 goto END
if errorlevel 3 goto VENV
if errorlevel 2 goto INSTALL
if errorlevel 1 goto ENABLE

:ENABLE
cls
echo Enabling Long Path Support...
echo =============================
echo.
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
if %errorlevel% == 0 (
    echo.
    echo SUCCESS: Long path support enabled!
    echo Please restart your computer for changes to take effect.
) else (
    echo.
    echo ERROR: Could not enable long path support.
    echo Please run this script as Administrator.
)
echo.
pause
goto MENU

:INSTALL
cls
echo Installing TensorFlow CPU...
echo ===========================
echo.
pip install tensorflow-cpu
if %errorlevel% == 0 (
    echo.
    echo SUCCESS: TensorFlow CPU installed!
) else (
    echo.
    echo ERROR: Failed to install TensorFlow.
    echo You may need to enable long path support first.
)
echo.
pause
goto MENU

:VENV
cls
echo Creating Virtual Environment in Short Path...
echo ===========================================
echo.
echo This will create a virtual environment in C:\ai_model
echo.
echo Creating directory...
mkdir C:\ai_model 2>nul
if exist C:\ai_model (
    echo Directory created successfully.
    echo.
    echo Creating virtual environment...
    cd C:\
    python -m venv ai_model
    echo.
    echo Activating virtual environment...
    C:\ai_model\Scripts\activate
    echo.
    echo Installing TensorFlow in virtual environment...
    pip install tensorflow
    echo.
    echo Virtual environment setup complete!
    echo To use: C:\ai_model\Scripts\activate
) else (
    echo Failed to create directory.
    echo Please run as Administrator.
)
echo.
pause
goto MENU

:END
echo.
echo Thank you for using the AI Image Detection System setup!
echo.
pause
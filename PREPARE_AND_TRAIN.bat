@echo off
echo ============================================================
echo  AI IMAGE DETECTION - PREPARATION AND TRAINING
echo  Dell Latitude E5470 Optimized
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Step 1: Installing required packages...
echo.
pip install pillow-heif psutil --quiet
if errorlevel 1 (
    echo WARNING: Some packages may not have installed correctly
    echo Continuing anyway...
)

echo.
echo Step 2: Preparing dataset...
echo.
python prepare_for_training.py

echo.
echo ============================================================
echo  PREPARATION COMPLETE!
echo ============================================================
echo.
echo Next: Review the training commands above and run one manually
echo Example: python train_optimized.py --max_images 156 --epochs 30 --batch_size 8
echo.
pause

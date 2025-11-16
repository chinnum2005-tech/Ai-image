@echo off
echo Enabling Long Path Support on Windows
echo ====================================
echo.

REM Try to enable long path support
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f

if %errorlevel% == 0 (
    echo Successfully enabled long path support!
    echo Please restart your computer for changes to take effect.
) else (
    echo Failed to enable long path support.
    echo Please run this script as Administrator.
    echo.
    echo Alternative solutions:
    echo 1. Manually enable via Registry Editor:
    echo    - Press Win + R, type "regedit" and press Enter
    echo    - Navigate to: HKLM\SYSTEM\CurrentControlSet\Control\FileSystem
    echo    - Create or modify DWORD "LongPathsEnabled" and set value to 1
    echo.
    echo 2. Use Group Policy Editor (Windows Pro/Enterprise):
    echo    - Press Win + R, type "gpedit.msc" and press Enter
    echo    - Navigate to: Computer Configuration ^> Administrative Templates ^> System ^> Filesystem
    echo    - Enable "Enable Win32 long paths"
)

echo.
pause
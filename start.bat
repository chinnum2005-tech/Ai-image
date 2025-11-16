@echo off
echo ========================================
echo  AI-GENERATED IMAGE DETECTION SYSTEM
echo ========================================
echo.

echo [1] Starting Flask Backend...
start cmd /k "cd /d %~dp0 && python app.py"
timeout /t 3 >nul

echo [2] Starting React Frontend...
start cmd /k "cd /d %~dp0\frontend && npm run dev"
timeout /t 3 >nul

echo.
echo ========================================
echo  System Starting...
echo ========================================
echo.
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Opening browser in 5 seconds...
timeout /t 5 >nul

start http://localhost:3000

echo.
echo ========================================
echo  AI Detection System is Running!
echo ========================================
echo.
echo Press any key to stop all services...
pause >nul

echo.
echo Stopping services...
taskkill /IM python.exe /F >nul 2>&1
taskkill /IM node.exe /F >nul 2>&1

echo Services stopped.
echo.

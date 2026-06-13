@echo off
title CLS Finder

echo ================================================================
echo   CLS Finder  --  Compact Localized State Calculator
echo   Native Python Backend (Full CPU Performance)
echo ================================================================
echo.
echo  Server: http://localhost:8765/
echo  Press Ctrl+C to stop.
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

:: Check fastapi is installed
python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo  [INFO] FastAPI / uvicorn not found. Installing...
    pip install fastapi uvicorn[standard]
    echo.
)

:: Open browser after server starts (3-second delay)
start "" cmd /c "ping -n 4 127.0.0.1 >nul & start http://localhost:8765/"

:: Start FastAPI server
python -m uvicorn server:app --host localhost --port 8765

pause

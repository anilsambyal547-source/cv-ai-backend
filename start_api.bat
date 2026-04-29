@echo off
echo ============================================
echo   CV AI Backend — FastAPI Server
echo   http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo ============================================
echo.

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [WARN] No venv found — using system Python
)

echo [INFO] Starting FastAPI server...
echo.
venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
pause

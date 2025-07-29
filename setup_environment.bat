@echo off
echo ========================================
echo Environment Setup for Optimization Toolkit
echo ========================================

echo.
echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [3/4] Installing dependencies...
pip install -r requirements.txt -r requirements-dev.txt

echo.
echo [4/4] Installing package in development mode...
pip install -e .

echo.
echo ========================================
echo Environment setup completed!
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the full system:
echo   run_system.bat
echo ========================================
pause

@echo off
rem setup.bat - Automated Environment Configuration Script for Windows

echo ==========================================================
echo           ResearchSphere AI Environment Setup
echo ==========================================================
echo.

rem Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in system PATH.
    echo Please install Python 3.11+ and try again.
    pause
    exit /b 1
)

echo [1/4] Creating Virtual Environment in '.venv'...
python -m venv .venv
if %errorlevel% neq 0 (
    echo Error occurred while creating the virtual environment.
    pause
    exit /b %errorlevel%
)

echo [2/4] Activating Virtual Environment and Upgrading Pip...
call .venv\Scripts\activate
python -m pip install --upgrade pip

echo [3/4] Installing Required Dependencies from 'requirements.txt'...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error occurred during dependency installation.
    pause
    exit /b %errorlevel%
)

echo [4/4] Creating local environment variables config template if not present...
if not exist .env (
    copy .env.example .env
    echo   -^> Created .env from template. Remember to add your API keys!
) else (
    echo   -^> .env file already exists. Skipping.
)

echo.
echo ==========================================================
echo           SETUP COMPLETED SUCCESSFULLY!
echo ==========================================================
echo.
echo To activate your virtual environment, run:
echo     .venv\Scripts\activate
echo.
echo To run the configuration health check, execute:
echo     python main.py
echo.
echo ==========================================================
pause

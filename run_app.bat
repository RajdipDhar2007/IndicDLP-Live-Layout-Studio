@echo off
rem Double-click to start IndicDLP Live Layout Studio.
rem Uses .venv if one exists in this folder, otherwise the Python on PATH.
cd /d "%~dp0"
title IndicDLP Live Layout Studio

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer, or create a .venv in this folder.
    pause
    exit /b 1
)

echo Starting the app. A browser tab will open; close this window to stop it.
"%PY%" -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo The app stopped with an error. If a package is missing, run:  "%PY%" -m pip install -r requirements.txt
    pause
)

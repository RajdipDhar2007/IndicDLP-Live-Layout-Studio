@echo off
rem One-click start: Label Studio (9091) + OCR backend (9090) + OCR driver for annotations 1-100.
rem Safe to re-run any time: regions that already have text are skipped, so only blank ones use Gemini requests.
setlocal
rem Paths are relative to this bat file (backend folder is a sibling of this one), so moving the project won't break it.
set "WORK_DIR=%~dp0"
if "%WORK_DIR:~-1%"=="\" set "WORK_DIR=%WORK_DIR:~0,-1%"
set "BACKEND_DIR=%WORK_DIR%\..\newspaper-ocr-backend-master"

rem A machine-level LABEL_STUDIO_API_KEY (old token) overrides .env, so clear it for the processes started here.
set "LABEL_STUDIO_API_KEY="
rem OCR every region (including Image), pace requests under the free-tier 15/min, save every 10 regions.
set "OCR_SKIP_LABELS="
set "GEMINI_MIN_INTERVAL=4.5"
set "OCR_CHECKPOINT_EVERY=10"

echo [1/3] Label Studio...
curl -s -m 3 -o nul http://localhost:9091/health
if errorlevel 1 (
  start "Label Studio" /min /d "%BACKEND_DIR%" cmd /c "ls-env\Scripts\python.exe -m label_studio.server start --port 9091 --no-browser"
  call :wait http://localhost:9091/health 90
) else (
  echo       already running
)

echo [2/3] OCR backend...
curl -s -m 3 -o nul http://localhost:9090/health
if errorlevel 1 (
  start "OCR backend" /min /d "%BACKEND_DIR%" cmd /c ""backend-env\Scripts\python.exe" app.py > "%WORK_DIR%\backend.log" 2>&1"
  call :wait http://localhost:9090/health 60
) else (
  echo       already running - NOTE: it keeps the settings it was started with
)

start "" http://localhost:9091/projects/2/data

echo [3/3] OCR driver - leave this window open. Progress: ocr_progress.json
cd /d "%WORK_DIR%"
"%BACKEND_DIR%\backend-env\Scripts\python.exe" ocr_driver.py
echo.
echo Driver stopped. See the last lines above (daily quota / error / finished).
pause
exit /b 0

:wait
set /a tries=0
:waitloop
curl -s -m 3 -o nul %1
if not errorlevel 1 exit /b 0
set /a tries+=1
if %tries% geq %2 (echo       %1 did not come up & exit /b 1)
ping -n 2 127.0.0.1 >nul
goto waitloop

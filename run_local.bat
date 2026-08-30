@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

where docker >nul 2>nul
if errorlevel 1 (
    echo Docker is not installed or not on PATH.
    echo Install Docker Desktop, then run this file again.
    pause
    exit /b 1
)

echo Building local image...
docker build -t aerialsegmentation-local .
if errorlevel 1 (
    echo Docker build failed.
    pause
    exit /b 1
)

echo Starting local container on http://localhost:7860 ...
docker run --rm -p 7860:7860 aerialsegmentation-local

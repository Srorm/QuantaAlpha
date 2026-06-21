# QuantaAlpha Web Dashboard launcher for Windows (replaces the bash start.sh which uses lsof/hostname).
# Starts the FastAPI backend (port 8000) and the Vite frontend (port 3000) in the quantaalpha conda env.
$ErrorActionPreference = "Stop"
$ENVDIR = "C:\Users\Storm\miniconda3\envs\quantaalpha"
$REPO   = "C:\Users\Storm\Desktop\QuantaAlpha\QuantaAlpha"
$FE     = Join-Path $REPO "frontend-v2"
$py     = Join-Path $ENVDIR "python.exe"
$npm    = Join-Path $ENVDIR "npm.cmd"
$env:Path = "$ENVDIR;$ENVDIR\Scripts;$ENVDIR\Library\bin;" + $env:Path

Write-Host "Starting QuantaAlpha web dashboard..." -ForegroundColor Cyan
# Backend (FastAPI on :8000) in a new window
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$FE'; & '$py' backend/app.py" -WindowStyle Normal
Start-Sleep -Seconds 3
# Frontend (Vite on :3000) in a new window
Start-Process powershell -ArgumentList "-NoExit","-Command","cd '$FE'; & '$npm' run dev" -WindowStyle Normal

Write-Host ""
Write-Host "  Frontend : http://localhost:3000" -ForegroundColor Green
Write-Host "  Backend  : http://localhost:8000   (API docs: /docs)" -ForegroundColor Green
Write-Host "Two PowerShell windows opened (backend + frontend). Close them to stop." -ForegroundColor Yellow

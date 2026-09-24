@echo off
setlocal

cd /d "%~dp0"

set "PATH=%~dp0.venv\Lib\site-packages\nvidia\cublas\bin;%~dp0.venv\Lib\site-packages\nvidia\cudnn\bin;%PATH%"

uv run --frozen python -m local_phone_ai_v2
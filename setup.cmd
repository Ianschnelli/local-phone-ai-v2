@echo off

set "HF_HUB_DISABLE_XET=1"

uv sync --frozen
if errorlevel 1 exit /b 1

uv run --frozen -- python -m local_phone_ai_v2.setup
if errorlevel 1 exit /b 1

uv run --frozen -- python -m local_phone_ai_v2.doctor
if errorlevel 1 exit /b 1

echo.
echo Setup complete and system ready.
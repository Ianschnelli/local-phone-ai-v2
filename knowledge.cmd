@echo off
uv run --no-sync python -m local_phone_ai_v2.knowledge.ingest
if errorlevel 1 exit /b 1

echo.
echo Knowledge updated.
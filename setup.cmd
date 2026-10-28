@echo off

set "HF_HUB_DISABLE_XET=1"

rem PyTorch-Variante: gpu bei NVIDIA-Karte (nvidia-smi vorhanden), sonst cpu.
rem Erzwingen mit: set TORCH_VARIANT=cpu  bzw.  set TORCH_VARIANT=gpu
if not defined TORCH_VARIANT (
    where nvidia-smi >nul 2>nul && (set "TORCH_VARIANT=gpu") || (set "TORCH_VARIANT=cpu")
)
echo PyTorch variant: %TORCH_VARIANT%

uv sync --frozen --extra %TORCH_VARIANT%
if errorlevel 1 exit /b 1

uv run --no-sync -- python -m local_phone_ai_v2.setup
if errorlevel 1 exit /b 1

uv run --no-sync -- python -m local_phone_ai_v2.doctor
if errorlevel 1 exit /b 1

echo.
echo Setup complete and system ready.
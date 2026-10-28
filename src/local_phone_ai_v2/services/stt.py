
from __future__ import annotations

import os
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING

from local_phone_ai_v2.config.languages import get_language

if TYPE_CHECKING:
    from pipecat.services.whisper.stt import WhisperSTTService


def _prepare_cuda_path() -> None:
    """Make CUDA DLLs available to CTranslate2 on Windows."""
    if os.name != "nt":
        return

    site = Path(sysconfig.get_path("purelib"))

    folders = [
        site / "nvidia" / "cublas" / "bin",
        site / "nvidia" / "cudnn" / "bin",
    ]

    for folder in folders:
        if not folder.is_dir():
            raise RuntimeError(
                f"Required CUDA DLL directory not found: {folder}"
            )

    current_path = os.environ.get("PATH", "")
    existing = {
        os.path.normcase(os.path.normpath(entry))
        for entry in current_path.split(os.pathsep)
        if entry
    }

    additions = [
        str(folder.resolve())
        for folder in folders
        if os.path.normcase(str(folder.resolve())) not in existing
    ]

    if additions:
        os.environ["PATH"] = os.pathsep.join(
            additions + ([current_path] if current_path else [])
        )


def create_stt(config: dict) -> WhisperSTTService:
    device = config["device"]
    compute_type = "int8" if device == "cpu" else "float16"

    if device == "cuda":
        _prepare_cuda_path()

    # Import after CUDA PATH initialization.
    from pipecat.services.whisper.stt import WhisperSTTService

    return WhisperSTTService(
        device=device,
        compute_type=compute_type,
        settings=WhisperSTTService.Settings(
            model=config["model"],
            language=get_language(config["language"]),
        ),
    )

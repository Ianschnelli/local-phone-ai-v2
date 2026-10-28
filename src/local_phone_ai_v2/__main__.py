import asyncio
from pathlib import Path

from loguru import logger

from local_phone_ai_v2.audio.local_transport import create_local_transport
from local_phone_ai_v2.config.loader import load_config
from local_phone_ai_v2.config.validate import validate_config
from local_phone_ai_v2.conversation.customer_loader import (
    load_customer_config,
)
from local_phone_ai_v2.conversation.session import CallSession
from local_phone_ai_v2.runtime.resources import RuntimeResources
from local_phone_ai_v2.runtime.warmup import warmup_qwen


async def main() -> None:
    # Zusätzlich zur Konsole in eine Datei loggen, damit Läufe nachvollziehbar
    # sind. Achtung: enthält Gesprächstexte (B-13), nur für die Entwicklung.
    logger.add(
        "logs/app.log",
        level="DEBUG",
        encoding="utf-8",
        rotation="10 MB",
        retention=5,
    )

    config = load_config(
        Path("config.yaml")
    )
    validate_config(config)

    resources = RuntimeResources(config)

    try:
        resources.warmup()

        customer_dir = Path(config["customer"]["facts_path"]).parent

        customer = load_customer_config(
            customer_dir.parent,
            customer_dir.name,
        )

        transport = create_local_transport()

        session = CallSession(
            config,
            transport,
            customer["instructions"],
            customer["greeting"],
            resources,
        )

        warmup_qwen(
            config["llm"],
        )

        print("System ready.")

        await session.run()

    finally:
        resources.close()


if __name__ == "__main__":
    asyncio.run(main())
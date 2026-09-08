import asyncio
from pathlib import Path

from local_phone_ai_v2.audio.local_transport import create_local_transport
from local_phone_ai_v2.config.loader import load_config
from local_phone_ai_v2.config.validate import validate_config
from local_phone_ai_v2.conversation.customer_loader import (
    load_customer_config,
)
from local_phone_ai_v2.conversation.instructions import (
    build_instructions,
)
from local_phone_ai_v2.conversation.session import CallSession
from local_phone_ai_v2.runtime.resources import RuntimeResources
from local_phone_ai_v2.runtime.warmup import warmup_qwen


async def main() -> None:
    config = load_config(
        Path("config.yaml")
    )
    validate_config(config)

    resources = RuntimeResources(config)

    try:
        resources.warmup()

        customer = load_customer_config(
            Path("customers"),
            "test_customer",
        )

        system_instruction = build_instructions(
            customer["instructions"]
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
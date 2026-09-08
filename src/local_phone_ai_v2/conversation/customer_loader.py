from pathlib import Path

import yaml


def load_customer_config(
    customers_directory: Path,
    customer_id: str,
) -> dict:
    path = customers_directory / customer_id / "customer.yaml"

    if not path.is_file():
        raise FileNotFoundError(
            f"Customer configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid customer configuration: {path}"
        )

    instructions = data.get("instructions")
    greeting = data.get("greeting")

    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(
            "Customer instructions must be a non-empty string."
        )

    if not isinstance(greeting, str) or not greeting.strip():
        raise ValueError(
            "Customer greeting must be a non-empty string."
        )

    return {
        "instructions": instructions.strip(),
        "greeting": greeting.strip(),
    }
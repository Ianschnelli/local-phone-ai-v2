def validate_config(config: dict) -> None:
    required_sections = (
        "audio",
        "vad",
        "turn",
        "stt",
        "llm",
        "tts",
        "rag",
    )

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing config section: {section}"
            )

    valid_devices = {
        "cpu",
        "cuda",
    }

    for section in (
        "turn",
        "stt",
        "tts",
    ):
        device = config[section].get("device")

        if device not in valid_devices:
            raise ValueError(
                f"Invalid device for {section}: "
                f"{device}. "
                "Use 'cpu' or 'cuda'."
            )

    thinking = config["llm"].get(
        "thinking"
    )

    if (
        thinking is not None
        and not isinstance(
            thinking,
            bool,
        )
    ):
        raise ValueError(
            "llm.thinking must be true or false."
        )

    temperature = config["llm"].get(
        "temperature"
    )

    top_p = config["llm"].get(
        "top_p"
    )

    max_tokens = config["llm"].get(
        "max_tokens"
    )

    if not isinstance(
        temperature,
        (int, float),
    ):
        raise ValueError(
            "llm.temperature must be a number."
        )

    if not 0 <= temperature <= 2:
        raise ValueError(
            "llm.temperature must be "
            "between 0 and 2."
        )

    if not isinstance(
        top_p,
        (int, float),
    ):
        raise ValueError(
            "llm.top_p must be a number."
        )

    if not 0 < top_p <= 1:
        raise ValueError(
            "llm.top_p must be greater "
            "than 0 and at most 1."
        )

    if (
        not isinstance(
            max_tokens,
            int,
        )
        or max_tokens < 1
    ):
        raise ValueError(
            "llm.max_tokens must be "
            "a positive integer."
        )

    rag = config["rag"]

    if not isinstance(
        rag,
        dict,
    ):
        raise ValueError(
            "rag must be a section."
        )

    enabled = rag.get(
        "enabled"
    )

    if not isinstance(
        enabled,
        bool,
    ):
        raise ValueError(
            "rag.enabled must be true or false."
        )

    embedding_model = rag.get(
        "embedding_model"
    )

    if (
        not isinstance(
            embedding_model,
            str,
        )
        or not embedding_model.strip()
    ):
        raise ValueError(
            "rag.embedding_model must be "
            "a non-empty string."
        )

    sparse_model = rag.get(
        "sparse_model"
    )

    if (
        not isinstance(
            sparse_model,
            str,
        )
        or not sparse_model.strip()
    ):
        raise ValueError(
            "rag.sparse_model must be "
            "a non-empty string."
        )

    candidate_k = rag.get(
        "candidate_k"
    )

    if (
        not isinstance(
            candidate_k,
            int,
        )
        or candidate_k < 1
    ):
        raise ValueError(
            "rag.candidate_k must be "
            "an integer greater than 0."
        )

    final_k = rag.get(
        "final_k"
    )

    if (
        not isinstance(
            final_k,
            int,
        )
        or final_k < 1
    ):
        raise ValueError(
            "rag.final_k must be "
            "an integer greater than 0."
        )

    if final_k > candidate_k:
        raise ValueError(
            "rag.final_k must not be "
            "greater than rag.candidate_k."
        )

    qdrant_path = rag.get(
        "qdrant_path"
    )

    if (
        not isinstance(
            qdrant_path,
            str,
        )
        or not qdrant_path.strip()
    ):
        raise ValueError(
            "rag.qdrant_path must be "
            "a non-empty string."
        )

    collection = rag.get(
        "collection"
    )

    if (
        not isinstance(
            collection,
            str,
        )
        or not collection.strip()
    ):
        raise ValueError(
            "rag.collection must be "
            "a non-empty string."
        )

    query_timeout_secs = rag.get(
        "query_timeout_secs"
    )

    if (
        not isinstance(
            query_timeout_secs,
            (int, float),
        )
        or query_timeout_secs <= 0
    ):
        raise ValueError(
            "rag.query_timeout_secs must be "
            "greater than 0."
        )
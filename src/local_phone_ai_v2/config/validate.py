from local_phone_ai_v2.config.devices import VALID_TORCH_DEVICES


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

    reranker_device = config["rag"]["reranker"].get("device")

    if reranker_device not in VALID_TORCH_DEVICES:
        raise ValueError(
            f"Invalid device for rag.reranker: {reranker_device}. "
            "Use 'auto', 'cpu' or 'cuda'."
        )

    llm = config["llm"]

    if not isinstance(
        llm,
        dict,
    ):
        raise TypeError(
            "llm must be a section."
        )

    provider = llm.get(
        "provider"
    )

    if provider != "lm_studio":
        raise ValueError(
            "llm.provider must be 'lm_studio'."
        )

    base_url = llm.get(
        "base_url"
    )

    if (
        not isinstance(
            base_url,
            str,
        )
        or not base_url.strip()
    ):
        raise ValueError(
            "llm.base_url must be "
            "a non-empty string."
        )

    if not base_url.startswith(
        ("http://", "https://")
    ):
        raise ValueError(
            "llm.base_url must start with "
            "'http://' or 'https://'."
        )

    api_key = llm.get(
        "api_key"
    )

    if (
        not isinstance(
            api_key,
            str,
        )
        or not api_key.strip()
    ):
        raise ValueError(
            "llm.api_key must be "
            "a non-empty string."
        )

    model = llm.get(
        "model"
    )

    if (
        not isinstance(
            model,
            str,
        )
        or not model.strip()
    ):
        raise ValueError(
            "llm.model must be "
            "a non-empty string."
        )

    tokenizer_model = llm.get(
        "tokenizer_model"
    )

    if (
        not isinstance(
            tokenizer_model,
            str,
        )
        or not tokenizer_model.strip()
    ):
        raise ValueError(
            "llm.tokenizer_model must be "
            "a non-empty string."
        )

    context_window = llm.get(
        "context_window"
    )

    if (
        not isinstance(
            context_window,
            int,
        )
        or context_window < 1
    ):
        raise ValueError(
            "llm.context_window must be "
            "a positive integer."
        )

    response_reserve_tokens = llm.get(
        "response_reserve_tokens"
    )

    if (
        not isinstance(
            response_reserve_tokens,
            int,
        )
        or response_reserve_tokens < 1
    ):
        raise ValueError(
            "llm.response_reserve_tokens must be "
            "a positive integer."
        )

    if response_reserve_tokens >= context_window:
        raise ValueError(
            "llm.response_reserve_tokens must be "
            "smaller than llm.context_window."
        )

    thinking = llm.get(
        "thinking"
    )

    if (
        thinking is not None
        and not isinstance(
            thinking,
            bool,
        )
    ):
        raise TypeError(
            "llm.thinking must be true or false."
        )

    temperature = llm.get(
        "temperature"
    )

    top_p = llm.get(
        "top_p"
    )

    max_tokens = llm.get(
        "max_tokens"
    )

    if not isinstance(
        temperature,
        (int, float),
    ):
        raise TypeError(
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
        raise TypeError(
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
        raise TypeError(
            "rag must be a section."
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
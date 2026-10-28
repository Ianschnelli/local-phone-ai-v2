from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .enrichment import PROMPT_VERSION


INDEX_SCHEMA_VERSION = 1


def build_index_fingerprint(
    config: dict,
    knowledge_directory: Path,
) -> str:
    rag = config["rag"]
    enrichment = rag.get("enrichment") or {}

    parameters = {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "embedding_backend": rag.get(
            "embedding_backend", "fastembed"
        ),
        "embedding_model": rag["embedding_model"],
        "embedding_query_prefix": rag.get(
            "embedding_query_prefix", ""
        ),
        "embedding_passage_prefix": rag.get(
            "embedding_passage_prefix", ""
        ),
        "sparse_model": rag["sparse_model"],
        "chunk_tokenizer": rag.get(
            "chunk_tokenizer", rag["embedding_model"]
        ),
        "enrichment_enabled": enrichment.get(
            "enabled", False
        ),
        "enrichment_prompt_version": (
            PROMPT_VERSION
            if enrichment.get("enabled")
            else None
        ),
    }

    digest = hashlib.sha256()

    digest.update(
        json.dumps(
            parameters,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    )

    files = sorted(
        path
        for path in knowledge_directory.rglob("*")
        if path.is_file()
    )

    if not files:
        raise ValueError(
            f"No knowledge documents found: {knowledge_directory}"
        )

    for path in files:
        relative_path = path.relative_to(
            knowledge_directory
        ).as_posix()

        digest.update(b"\0FILE\0")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

    return digest.hexdigest()


MANIFEST_FILENAME = "index_manifest.json"


def write_index_manifest(
    index_directory: Path,
    fingerprint: str,
) -> None:
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "fingerprint": fingerprint,
    }

    (index_directory / MANIFEST_FILENAME).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

def index_is_current(
    index_directory: Path,
    expected_fingerprint: str,
) -> bool:
    manifest_path = index_directory / MANIFEST_FILENAME

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False

    return (
        isinstance(manifest, dict)
        and manifest.get("schema_version") == INDEX_SCHEMA_VERSION
        and manifest.get("fingerprint") == expected_fingerprint
    )
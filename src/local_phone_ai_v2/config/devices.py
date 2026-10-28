from loguru import logger

VALID_TORCH_DEVICES = (
    "auto",
    "cpu",
    "cuda",
)


def resolve_torch_device(
    requested: str | None,
) -> str:
    """Gerät für PyTorch-Modelle (Reranker, Embeddings) auflösen.

    "auto" nimmt CUDA, wenn PyTorch eine GPU sieht, sonst die CPU. So läuft
    derselbe Code auf Rechnern mit und ohne NVIDIA-Karte; welche
    PyTorch-Variante installiert ist, entscheidet setup.cmd.
    "cuda" erzwingt die GPU und bricht mit klarer Meldung ab, wenn keine da ist.
    """
    device = (requested or "auto").lower()

    if device not in VALID_TORCH_DEVICES:
        raise ValueError(
            f"Invalid torch device: {requested}. "
            "Use 'auto', 'cpu' or 'cuda'."
        )

    if device == "cpu":
        return "cpu"

    import torch

    cuda_available = torch.cuda.is_available()

    if device == "cuda" and not cuda_available:
        raise RuntimeError(
            "device 'cuda' requested, but PyTorch sees no CUDA GPU "
            f"(torch {torch.__version__}). Install the GPU variant "
            "via setup.cmd or use device 'auto' or 'cpu'."
        )

    resolved = "cuda" if cuda_available else "cpu"

    logger.info(
        "Torch device: requested={}, using={} (torch {})",
        device,
        resolved,
        torch.__version__,
    )

    return resolved

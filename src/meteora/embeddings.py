from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence, Union

import numpy as np


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any:
        ...


EncoderLike = Union[Encoder, Callable[[Sequence[str]], Any]]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class HashingEncoder:
    """A tiny deterministic encoder for tests, demos, and dependency-free smoke runs.

    For production evidence selection, use `SentenceTransformerEncoder` or pass any encoder
    exposing an `encode(list[str])` method.
    """

    def __init__(self, n_features: int = 384, lowercase: bool = True) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive.")
        self.n_features = n_features
        self.lowercase = lowercase

    def encode(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(self.n_features, dtype=np.float64)
            normalized = text.lower() if self.lowercase else text
            for token in _TOKEN_RE.findall(normalized):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                raw = int.from_bytes(digest, byteorder="big", signed=False)
                index = raw % self.n_features
                sign = 1.0 if (raw >> 63) == 0 else -1.0
                vector[index] += sign
            rows.append(vector)
        return normalize_rows(np.vstack(rows)) if rows else np.empty((0, self.n_features))


class SentenceTransformerEncoder:
    """Lazy wrapper around `sentence_transformers.SentenceTransformer`."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        **model_kwargs: Any,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers support with "
                '`pip install "meteora-rag[sentence-transformers]"`.'
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device, **model_kwargs)

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        options = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        options.update(kwargs)
        return np.asarray(self.model.encode(list(texts), **options), dtype=np.float64)


def encode_texts(encoder: EncoderLike, texts: Iterable[str]) -> np.ndarray:
    items = list(texts)
    if not items:
        return np.empty((0, 0), dtype=np.float64)

    if hasattr(encoder, "encode"):
        try:
            encoded = encoder.encode(items, convert_to_numpy=True, show_progress_bar=False)
        except TypeError:
            encoded = encoder.encode(items)
    else:
        encoded = encoder(items)

    array = _to_numpy(encoded)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[0] != len(items):
        raise ValueError(
            f"Encoder returned {array.shape[0]} embeddings for {len(items)} input texts."
        )
    return array.astype(np.float64, copy=False)


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim == 1:
        left = left.reshape(1, -1)
    if right.ndim == 1:
        right = right.reshape(1, -1)
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"Embedding dimensions do not match: {left.shape[1]} vs {right.shape[1]}."
        )
    return normalize_rows(left) @ normalize_rows(right).T


def normalize_rows(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return array / norms


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)

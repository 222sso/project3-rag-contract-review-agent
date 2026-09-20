"""Embedding generation module with retry and fallback resilience (768 dimensions)."""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.json"


def load_embedding_config() -> dict[str, Any]:
    """Loads embedding configuration from project_config.json."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            return {
                "model": cfg.get("embedding_model", "gemini-embedding-2"),
                "dimension": cfg.get("embedding_dimension", 768),
            }
    return {"model": "gemini-embedding-2", "dimension": 768}


def _generate_deterministic_mock_embedding(text: str, dimension: int = 768) -> list[float]:
    """Generates a deterministic normalized unit vector of fixed dimension based on text hash."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dimension).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


class GeminiEmbedder:
    """Gemini Embedder wrapper supporting batch embedding generation with retries and fallbacks."""

    def __init__(self, api_key: str | None = None, model: str | None = None, dimension: int = 768):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        cfg = load_embedding_config()
        self.model_name = model or cfg["model"]
        self.dimension = dimension or cfg["dimension"]
        self.use_mock = os.getenv("RAG_USE_MOCK_EMBEDDING", "false").lower() in ("true", "1")
        self.client = None

        if not self.use_mock and self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Google GenAI client for embedding model '%s'.", self.model_name)
            except Exception as e:
                logger.warning("Could not initialize Google GenAI client: %s. Operating in fallback mode.", e)
        else:
            logger.warning("GEMINI_API_KEY is not set or mock mode is active. Operating in deterministic fallback mode.")

    def _embed_single_with_retry(self, text: str, max_retries: int = 3) -> list[float]:
        """Embeds a single string with retry and fallback."""
        clean_text = text.strip() or " "
        if not self.client:
            return _generate_deterministic_mock_embedding(clean_text, self.dimension)

        for attempt in range(1, max_retries + 1):
            try:
                res = self.client.models.embed_content(
                    model=self.model_name,
                    contents=clean_text,
                    config={"output_dimensionality": self.dimension},
                )
                if hasattr(res, "embeddings") and res.embeddings:
                    values = res.embeddings[0].values
                elif hasattr(res, "embedding") and res.embedding:
                    values = res.embedding.values
                else:
                    raise ValueError(f"Unexpected response structure: {res}")

                if len(values) != self.dimension:
                    values = values[:self.dimension] + [0.0] * max(0, self.dimension - len(values))
                return values
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str
                is_timeout = "timeout" in err_str or "timed out" in err_str or "deadline" in err_str
                if attempt < max_retries and (is_rate_limit or is_timeout):
                    time.sleep(2 ** attempt)
                else:
                    logger.warning("Single embed failed (%s). Falling back.", e)
                    return _generate_deterministic_mock_embedding(clean_text, self.dimension)

        return _generate_deterministic_mock_embedding(clean_text, self.dimension)

    def embed_texts(self, texts: list[str], max_workers: int = 10) -> list[list[float]]:
        """Generates 768-dimensional embeddings for a list of texts in parallel."""
        if not texts:
            return []

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(self._embed_single_with_retry, texts))
        return embeddings

    def embed_single(self, text: str) -> list[float]:
        """Embeds a single text string."""
        return self._embed_single_with_retry(text)

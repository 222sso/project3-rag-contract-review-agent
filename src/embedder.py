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
                "model": cfg.get("embedding_model", "text-embedding-004"),
                "dimension": cfg.get("embedding_dimension", 768),
            }
    return {"model": "text-embedding-004", "dimension": 768}


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

    def embed_texts(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        """Generates 768-dimensional embeddings for a list of texts with retry and fallback handling."""
        if not texts:
            return []

        if self.client:
            for attempt in range(1, max_retries + 1):
                try:
                    embeddings = []
                    for text in texts:
                        clean_text = text.strip() or " "
                        res = self.client.models.embed_content(
                            model=self.model_name,
                            contents=clean_text,
                            config={"output_dimensionality": self.dimension},
                        )
                        values = res.embedding.values
                        if len(values) != self.dimension:
                            logger.warning(
                                "Dimension mismatch from API (%d != %d). Adjusting.",
                                len(values),
                                self.dimension,
                            )
                            values = values[:self.dimension] + [0.0] * max(0, self.dimension - len(values))
                        embeddings.append(values)
                    return embeddings
                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str
                    is_timeout = "timeout" in err_str or "timed out" in err_str or "deadline" in err_str
                    
                    logger.warning(
                        "[Attempt %d/%d] Embedding API error (%s).",
                        attempt,
                        max_retries,
                        e,
                    )
                    if attempt < max_retries and (is_rate_limit or is_timeout):
                        backoff = 2 ** attempt
                        logger.info("Applying exponential backoff: sleeping %ds before retry...", backoff)
                        time.sleep(backoff)
                    else:
                        logger.error("All %d retries failed or non-retryable error. Falling back to deterministic embedding.", max_retries)
                        break

        # Deterministic fallback
        return [_generate_deterministic_mock_embedding(t, self.dimension) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        """Embeds a single text string."""
        return self.embed_texts([text])[0]

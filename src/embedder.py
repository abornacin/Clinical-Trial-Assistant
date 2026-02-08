import os
import sys
import warnings
import logging
import time
from contextlib import redirect_stdout, redirect_stderr

# 🔕 Kill all known noise sources
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TQDM_DISABLE"] = "1"

warnings.filterwarnings("ignore")

# Silence all relevant loggers
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # 🔇 Hard mute stdout/stderr during model load
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            self.model = SentenceTransformer(model_name)

    def _embed_batch_safe(self, texts, retries=5, base_delay=2):
        """
        Embed a single batch with retry + exponential backoff.
        """
        for attempt in range(retries):
            try:
                return self.model.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            except Exception as e:
                sleep_time = base_delay * (2 ** attempt)
                print(f"Batch attempt {attempt+1} failed: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)

        raise RuntimeError(f"Failed to embed batch after {retries} retries.")

    def embed(self, texts, batch_size=32, retries=5, base_delay=2):
        """
        Embed a list of texts in batches with retry safety.
        """
        if not isinstance(texts, list):
            texts = [texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._embed_batch_safe(
                batch,
                retries=retries,
                base_delay=base_delay
            )
            all_embeddings.extend(embeddings)

        return all_embeddings

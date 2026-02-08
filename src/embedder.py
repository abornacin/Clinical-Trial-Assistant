import os
import sys
import warnings
import logging
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

    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=False)


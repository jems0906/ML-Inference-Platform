import logging
import time

import requests

from app.config import settings

logger = logging.getLogger(__name__)

HF_API_BASE = "https://api-inference.huggingface.co/models"


def hf_post(model: str, payload: dict, retries: int = 3) -> list | dict:
    """POST to HuggingFace Inference API with automatic retry on model cold-start."""
    url = f"{HF_API_BASE}/{model}"
    headers = {"Authorization": f"Bearer {settings.hf_api_token}"}
    for attempt in range(retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 503:
            # Model is warming up — HF returns estimated_time in the body
            wait = min(resp.json().get("estimated_time", 20), 30)
            logger.warning("Model %r is loading, retrying in %ss (attempt %d/%d)",
                           model, wait, attempt + 1, retries)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Model {model!r} unavailable after {retries} retries")
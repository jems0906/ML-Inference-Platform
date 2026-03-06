import logging

from transformers import pipeline

from app.config import settings

logger = logging.getLogger(__name__)


def _load(task: str, model: str, **kwargs):
    logger.info("Loading model %r for task %r ...", model, task)
    try:
        return pipeline(task, model=model, **kwargs)
    except Exception as exc:
        logger.error("Failed to load model %r: %s", model, exc)
        raise RuntimeError(f"Model '{model}' could not be loaded") from exc


model_v1 = _load("sentiment-analysis", settings.model_name_v1)
model_v2 = _load("sentiment-analysis", settings.model_name_v2)
chat_model = _load(
    "text-generation",
    "microsoft/DialoGPT-small",
)

MODELS = {
    "v1": model_v1,
    "v2": model_v2,
    "chat": chat_model,
}
import random

from app.config import settings
from app.models import hf_post


def perform_inference(text: str, tenant_id: str) -> dict:
    # A/B routing: send ab_test_percentage fraction of traffic to v2
    version = "v2" if random.random() < settings.ab_test_percentage else "v1"
    model = settings.model_name_v2 if version == "v2" else settings.model_name_v1
    result = hf_post(model, {"inputs": text})
    return {"result": result, "version": version, "tenant": tenant_id}


def perform_chat(text: str, tenant_id: str) -> dict:
    result = hf_post(
        "microsoft/DialoGPT-small",
        {"inputs": text, "parameters": {"max_new_tokens": 100, "temperature": 0.75}},
    )
    response = result[0]["generated_text"] if isinstance(result, list) else str(result)
    return {"response": response, "tenant": tenant_id}


def perform_batch_inference(texts: list[str], tenant_id: str) -> dict:
    return {
        "results": [perform_inference(t, tenant_id) for t in texts],
        "tenant": tenant_id,
    }
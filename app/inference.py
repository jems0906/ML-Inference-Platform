import random

from app.config import settings
from app.models import MODELS


def perform_inference(text: str, tenant_id: str) -> dict:
    # A/B routing: send ab_test_percentage fraction of traffic to v2
    if random.random() < settings.ab_test_percentage:
        version = "v2"
    else:
        version = "v1"
    result = MODELS[version](text)
    return {"result": result, "version": version, "tenant": tenant_id}


def perform_chat(text: str, tenant_id: str) -> dict:
    output = MODELS["chat"](text, max_new_tokens=100, do_sample=True, temperature=0.75)
    response = output[0]["generated_text"][len(text):].strip()
    return {"response": response, "tenant": tenant_id}


def perform_batch_inference(texts: list[str], tenant_id: str) -> dict:
    return {
        "results": [perform_inference(t, tenant_id) for t in texts],
        "tenant": tenant_id,
    }
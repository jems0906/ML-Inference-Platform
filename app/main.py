import logging
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from app.inference import perform_batch_inference, perform_chat, perform_inference
from app.rate_limit import is_rate_limited
from app.tenants import create_tenant, get_tenant_by_api_key, list_tenants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-tenant ML Inference Platform",
    description="Scalable, multi-tenant API for serving ML/GenAI models.",
    version="1.0.0",
)

# --- Prometheus metrics ---
inference_counter = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["tenant", "version"],
)
inference_duration = Histogram(
    "inference_duration_seconds",
    "Inference request duration in seconds",
    ["tenant"],
)


# --- Request schemas ---

class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)


class BatchInferenceRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)


class TenantCreateRequest(BaseModel):
    tenant_id: str = Field(..., min_length=3, max_length=32)
    rate_limit: int = Field(10, ge=1, le=1000)
    plan: str = Field("standard", pattern="^(free|standard|premium)$")


# --- Auth dependency ---

def get_tenant(api_key: str = Header(..., alias="X-API-Key")) -> str:
    tenant_id = get_tenant_by_api_key(api_key)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if is_rate_limited(tenant_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return tenant_id


# --- Info routes ---

@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Multi-tenant ML Inference Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok"}


@app.get("/metrics", tags=["Observability"])
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Tenant management ---

@app.get("/tenants", tags=["Tenants"])
def get_tenants():
    return list_tenants()


@app.post("/tenants", status_code=201, tags=["Tenants"])
def register_tenant(req: TenantCreateRequest):
    try:
        result = create_tenant(req.tenant_id, req.rate_limit, req.plan)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=409, detail="Tenant already exists")
    return result


# --- Inference routes ---

@app.post("/inference", tags=["Inference"])
def inference(request: InferenceRequest, tenant_id: str = Depends(get_tenant)):
    logger.info("inference tenant=%s text_len=%d", tenant_id, len(request.text))
    start = time.time()
    try:
        result = perform_inference(request.text, tenant_id)
    except Exception as exc:
        logger.error("inference error tenant=%s error=%s", tenant_id, exc)
        raise HTTPException(status_code=500, detail="Inference failed")
    inference_counter.labels(tenant=tenant_id, version=result["version"]).inc()
    inference_duration.labels(tenant=tenant_id).observe(time.time() - start)
    return result


@app.post("/batch_inference", tags=["Inference"])
def batch_inference(request: BatchInferenceRequest, tenant_id: str = Depends(get_tenant)):
    logger.info("batch_inference tenant=%s count=%d", tenant_id, len(request.texts))
    start = time.time()
    try:
        result = perform_batch_inference(request.texts, tenant_id)
    except Exception as exc:
        logger.error("batch_inference error tenant=%s error=%s", tenant_id, exc)
        raise HTTPException(status_code=500, detail="Batch inference failed")
    for res in result["results"]:
        inference_counter.labels(tenant=tenant_id, version=res["version"]).inc()
    inference_duration.labels(tenant=tenant_id).observe(time.time() - start)
    return result


@app.post("/chat", tags=["GenAI"])
def chat(request: InferenceRequest, tenant_id: str = Depends(get_tenant)):
    logger.info("chat tenant=%s text_len=%d", tenant_id, len(request.text))
    try:
        return perform_chat(request.text, tenant_id)
    except Exception as exc:
        logger.error("chat error tenant=%s error=%s", tenant_id, exc)
        raise HTTPException(status_code=500, detail="Chat failed")
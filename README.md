# Multi-tenant ML Inference Platform

A scalable API for serving ML models with multi-tenancy support.

## Features

- Multi-tenancy via API keys
- Rate limiting per tenant using Redis
- Model versioning and A/B testing
- Logging and metrics (Prometheus)
- GenAI chat endpoint
- Batch inference pipeline

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run locally: `uvicorn app.main:app --reload`
3. For Docker: `docker build -t ml-inference ./docker`
4. For Kubernetes: Use Minikube/Kind, apply manifests in k8s/

## Usage

Send POST to /inference with X-API-Key header and JSON body {"text": "your text"}

For batch: POST to /batch_inference with {"texts": ["text1", "text2"]}

For chat: POST to /chat with {"text": "Hello"}

## Monitoring

Metrics at /metrics, integrate with Prometheus/Grafana.
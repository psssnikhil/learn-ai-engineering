---
title: Deploying Fine-Tuned Models
description: >-
  Learn to serve your fine-tuned models in production with vLLM, Text Generation
  Inference, and cloud platforms
duration: 45 min
difficulty: advanced
has_code: false
module: module-15
youtube: 'https://www.youtube.com/watch?v=080pyz_vTRo'
objectives:
  - Deploy a fine-tuned model with vLLM
  - Set up an API server for model inference
  - Optimize inference with quantization and batching
  - Choose the right deployment platform for your use case
---
# Deploying Fine-Tuned Models

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Deploy models with vLLM and TGI | 45 min | Advanced |
| Optimize inference performance | | |
| Quantize models for production | | |
| Choose deployment platforms | | |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Fine-tuned model** | Merged model or LoRA adapter from Lessons 3-5 |
| **GPU server** | Minimum 8 GB VRAM for 7B quantized, 16 GB for fp16 |
| **Docker** (optional) | For TGI containerized deployment |
| **Python 3.10+** | With CUDA-compatible environment |

**Courses required:**
- Module 15, Lesson 3-5: Fine-tuning (OpenAI and open-source)
- Module 13: LLMOps and deployment fundamentals

```bash
pip install vllm openai  # For vLLM deployment
# OR
docker pull ghcr.io/huggingface/text-generation-inference:latest  # For TGI
```

---

## What You'll Build

By the end of this lesson, you will have:

- [ ] A running inference server with OpenAI-compatible API
- [ ] Successful test requests from a Python client
- [ ] Quantized model deployed (4-bit) for 2-3x throughput improvement
- [ ] Health check endpoint for load balancer integration
- [ ] Prometheus metrics accessible for monitoring
- [ ] A deployment decision documented for your use case

---

## Architecture

```
[Client Application]
    |  OpenAI SDK / HTTP requests
    v
[Load Balancer]  (optional)
    |  Health checks, SSL termination
    v
[Inference Server]
    |
    +-- vLLM (recommended)
    |       |-- PagedAttention memory management
    |       |-- Continuous batching
    |       |-- OpenAI-compatible /v1/chat/completions
    |       +-- LoRA adapter hot-swapping
    |
    +-- TGI (Docker alternative)
    |       |-- Hugging Face native API
    |       +-- /generate endpoint
    |
    +-- Cloud Managed
            |-- HF Inference Endpoints
            |-- Together AI / Fireworks
            +-- AWS SageMaker
    v
[GPU]  (quantized model in VRAM)
    |
    v
[Monitoring]
    - Prometheus metrics (vLLM)
    - Latency, throughput, queue depth
    - GPU memory utilization
```

---

## Deployment Options Overview

| Option | Best For | Cost | Complexity |
|--------|----------|------|------------|
| **vLLM** | Self-hosted, high throughput | GPU rental ($1-5/hr) | Medium |
| **TGI (Hugging Face)** | Docker-based deployment | GPU rental | Medium |
| **Hugging Face Inference Endpoints** | Managed deployment | $0.60-8/hr | Low |
| **Together AI / Fireworks** | Serverless fine-tuned models | Per-token | Low |
| **AWS SageMaker** | Enterprise, auto-scaling | Per-instance | High |
| **Ollama** | Local development, prototyping | Free (local) | Very Low |

---

## Step 1: Prepare Your Model for Serving

Before deploying, decide whether to serve a merged model or a LoRA adapter.

### Option A: Merged Model (single file)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "./my-finetuned-model")
merged = model.merge_and_unload()
merged.save_pretrained("./my-merged-model")
tokenizer = AutoTokenizer.from_pretrained("./my-finetuned-model")
tokenizer.save_pretrained("./my-merged-model")
```

### Option B: LoRA Adapter (smaller, swappable)

Keep the adapter separate — vLLM can load it at runtime without merging.

---

## Step 2: Deploy with vLLM (Recommended)

vLLM is the fastest open-source LLM serving engine with PagedAttention for efficient memory management.

### Installation

```bash
pip install vllm
```

### Serving a Merged Model

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./my-merged-model \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --dtype auto \
    --gpu-memory-utilization 0.9
```

### Serving a LoRA Adapter

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules support-bot=./my-finetuned-model \
    --max-lora-rank 64 \
    --host 0.0.0.0 \
    --port 8000
```

### Making Requests

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="support-bot",  # LoRA module name, or "./my-merged-model"
    messages=[
        {"role": "system", "content": "You are a customer support agent."},
        {"role": "user", "content": "What is your return policy for electronics?"},
    ],
    temperature=0.3,
    max_tokens=256,
)

print(response.choices[0].message.content)
print(f"Tokens: {response.usage.total_tokens}")
```

---

## Step 3: Deploy with Text Generation Inference (TGI)

Hugging Face's production-grade inference server, deployed via Docker.

```bash
docker run --gpus all -p 8080:80 \
    -v $(pwd)/my-merged-model:/model \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id /model \
    --max-input-length 2048 \
    --max-total-tokens 4096 \
    --max-batch-prefill-tokens 4096
```

```python
import requests

response = requests.post(
    "http://localhost:8080/generate",
    json={
        "inputs": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is your return policy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.3,
            "do_sample": True,
        },
    },
)
print(response.json()["generated_text"])
```

---

## Step 4: Quantize for Production

Quantization reduces model size and speeds up inference. Apply before deploying to production.

### AWQ Quantization (for vLLM)

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "./my-merged-model"
quant_path = "./my-model-awq"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
    },
)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

Serve the quantized model:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./my-model-awq \
    --quantization awq \
    --host 0.0.0.0 \
    --port 8000
```

### GGUF Format (for Ollama / llama.cpp)

```bash
# Convert for local deployment with Ollama
python convert_hf_to_gguf.py ./my-merged-model --outfile model.gguf --outtype q4_k_m
ollama create my-support-bot -f Modelfile
```

### Quantization Comparison

| Method | Size (7B model) | Speed | Quality |
|--------|-----------------|-------|---------|
| fp16 (original) | 14 GB | Baseline | Best |
| GPTQ (4-bit) | 4 GB | 1.5-2x faster | Very good |
| AWQ (4-bit) | 4 GB | 1.5-2x faster | Very good |
| GGUF Q4_K_M | 4.4 GB | Good (CPU-friendly) | Good |
| GGUF Q5_K_M | 5.3 GB | Good | Better |

---

## Step 5: Production Hardening

Wrap your inference server with production essentials:

```python
# health_check.py — verify model is loaded and responding
import requests
import sys

def check_health(base_url: str = "http://localhost:8000") -> bool:
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("Health check: OK")
            return True
    except requests.ConnectionError:
        pass

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "support-bot",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
            timeout=30,
        )
        if response.status_code == 200:
            print("Inference check: OK")
            return True
    except Exception as e:
        print(f"Inference check failed: {e}")

    return False

if __name__ == "__main__":
    sys.exit(0 if check_health() else 1)
```

### Production Dockerfile

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip
RUN pip install vllm

COPY ./my-model-awq /model
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/model", "--quantization", "awq", \
     "--host", "0.0.0.0", "--port", "8000"]
```

---

## Testing Your Build

### Verification Checklist

- [ ] Server starts without CUDA OOM errors
- [ ] Health check endpoint returns 200
- [ ] Chat completion request returns valid response
- [ ] Fine-tuned model produces domain-specific output (not generic)
- [ ] Latency under 2s for 256-token response
- [ ] Multiple concurrent requests handled (continuous batching)
- [ ] Prometheus metrics accessible at `/metrics` (vLLM)

### Load Testing

```python
import concurrent.futures
import time
import requests

def send_request(i: int) -> float:
    start = time.time()
    requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "support-bot",
            "messages": [{"role": "user", "content": f"Test query {i}"}],
            "max_tokens": 100,
        },
        timeout=60,
    )
    return time.time() - start

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    latencies = list(executor.map(send_request, range(50)))

print(f"Avg latency: {sum(latencies)/len(latencies):.2f}s")
print(f"P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
```

---

## Deployment Notes

### Production Checklist

**Performance:**
- [ ] Quantize model to 4-bit for 2-3x throughput improvement
- [ ] Enable continuous batching (vLLM does this automatically)
- [ ] Set appropriate `max_model_len` to avoid wasting VRAM
- [ ] Profile with realistic traffic patterns before deploying

**Reliability:**
- [ ] Health check endpoint for load balancer
- [ ] Graceful shutdown handling (SIGTERM → drain requests)
- [ ] Request timeout configuration (30-60s)
- [ ] Error handling for OOM and CUDA errors with automatic restart

**Monitoring:**

```python
# vLLM exposes Prometheus metrics at /metrics
# Key metrics to monitor:
# - vllm:num_requests_running (current load)
# - vllm:num_requests_waiting (queue depth)
# - vllm:avg_generation_throughput_toks_per_s (throughput)
# - vllm:gpu_cache_usage_perc (memory pressure)
```

**Cost Optimization:**

| Strategy | Impact |
|----------|--------|
| 4-bit quantization | 2-3x more throughput per GPU |
| Continuous batching | 5-10x throughput vs naive serving |
| Spot/preemptible instances | 60-70% cost reduction |
| Auto-scaling to zero | Pay only when processing requests |

### Platform Selection Guide

| Scenario | Recommended Platform |
|----------|---------------------|
| Prototyping locally | Ollama |
| Self-hosted production | vLLM on GPU cloud (Lambda, RunPod) |
| Team without ML ops | Hugging Face Inference Endpoints |
| High-traffic API | vLLM + Kubernetes + auto-scaling |
| Fine-tuned OpenAI model | OpenAI API (no self-hosting needed) |

---

## Extensions and Challenges

- **Multi-LoRA serving**: Serve multiple task-specific adapters on one base model
- **Speculative decoding**: Use a small draft model to speed up generation 2-3x
- **Request caching**: Cache identical prompts with Redis for repeated queries
- **A/B testing**: Route traffic between base and fine-tuned models, compare metrics
- **Auto-scaling**: Kubernetes HPA based on `vllm:num_requests_waiting` metric

---

## Key Takeaways

- vLLM is the go-to for high-throughput self-hosted serving with an OpenAI-compatible API
- Quantize models to 4-bit (AWQ or GPTQ) for 2-3x throughput improvement with minimal quality loss
- vLLM can serve LoRA adapters directly without merging — swap adapters per request
- Use Docker + TGI for containerized deployments in Kubernetes
- Monitor GPU memory, queue depth, and throughput in production
- Always run load tests before going live — continuous batching behavior differs from single-request testing

---

## Next Lesson

**Lesson 9: Model Distillation and Compression** — Learn to transfer knowledge from large models to small, fast models that are cheaper to run in production.

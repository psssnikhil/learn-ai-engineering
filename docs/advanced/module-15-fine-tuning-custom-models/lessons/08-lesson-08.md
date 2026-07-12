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

## Deployment Options Overview

| Option | Best For | Cost | Complexity |
|--------|----------|------|------------|
| **vLLM** | Self-hosted, high throughput | GPU rental ($1-5/hr) | Medium |
| **TGI (Hugging Face)** | Docker-based deployment | GPU rental | Medium |
| **Hugging Face Inference Endpoints** | Managed deployment | $0.60-8/hr | Low |
| **Together AI / Fireworks** | Serverless fine-tuned models | Per-token | Low |
| **AWS SageMaker** | Enterprise, auto-scaling | Per-instance | High |

---

## Option 1: vLLM (Recommended)

vLLM is the fastest open-source LLM serving engine with PagedAttention for efficient memory management.

### Installation

```bash
pip install vllm
```

### Serving a Merged Model

```bash
# Start vLLM server (OpenAI-compatible API)
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
# vLLM supports serving LoRA adapters directly
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules my-adapter=./my-lora-adapter \
    --max-lora-rank 64 \
    --host 0.0.0.0 \
    --port 8000
```

### Making Requests

```python
from openai import OpenAI

# vLLM exposes an OpenAI-compatible API
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="./my-merged-model",  # or "my-adapter" for LoRA
    messages=[
        {"role": "user", "content": "Explain our return policy"}
    ],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

---

## Option 2: Text Generation Inference (TGI)

Hugging Face's production-grade inference server, deployed via Docker.

```bash
# Pull and run TGI with your model
docker run --gpus all -p 8080:80 \
    -v ./my-merged-model:/model \
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
        "inputs": "What is your return policy?",
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
        }
    }
)
print(response.json()["generated_text"])
```

---

## Model Quantization for Production

Quantization reduces model size and speeds up inference. Use it before deploying.

### GGUF Format (for llama.cpp / Ollama)

```bash
# Install llama.cpp conversion tools
pip install llama-cpp-python

# Convert to GGUF (use the llama.cpp convert script)
python convert_hf_to_gguf.py ./my-merged-model --outfile model.gguf --outtype q4_k_m
```

### AWQ Quantization (for vLLM)

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "./my-merged-model"
quant_path = "./my-model-awq"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
    }
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
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

## Production Checklist

### Performance

- [ ] Quantize model to 4-bit for 2-3x throughput improvement
- [ ] Enable continuous batching (vLLM does this automatically)
- [ ] Set appropriate `max_model_len` to avoid wasting VRAM
- [ ] Profile with realistic traffic patterns before deploying

### Reliability

- [ ] Health check endpoint for load balancer
- [ ] Graceful shutdown handling
- [ ] Request timeout configuration
- [ ] Error handling for OOM and CUDA errors

### Monitoring

```python
# vLLM exposes Prometheus metrics
# Key metrics to monitor:
# - vllm:num_requests_running (current load)
# - vllm:num_requests_waiting (queue depth)
# - vllm:avg_generation_throughput_toks_per_s (throughput)
# - vllm:gpu_cache_usage_perc (memory pressure)
```

### Cost Optimization

| Strategy | Impact |
|----------|--------|
| 4-bit quantization | 2-3x more throughput per GPU |
| Continuous batching | 5-10x throughput vs naive serving |
| Spot/preemptible instances | 60-70% cost reduction |
| Auto-scaling to zero | Pay only when processing requests |

---

## Resources

- **vLLM Documentation**: [docs.vllm.ai](https://docs.vllm.ai)
- **TGI Documentation**: [huggingface.co/docs/text-generation-inference](https://huggingface.co/docs/text-generation-inference)
- **Ollama**: Run models locally with one command — [ollama.com](https://ollama.com)
- **Blog: Deploying LLMs in Production**: Practical guide to inference optimization

---

## Key Takeaways

- vLLM is the go-to for high-throughput self-hosted serving with an OpenAI-compatible API
- Quantize models to 4-bit (AWQ or GPTQ) for 2-3x throughput improvement with minimal quality loss
- vLLM can serve LoRA adapters directly without merging
- Use Docker + TGI for containerized deployments
- Monitor GPU memory, queue depth, and throughput in production

---

## Next Lesson

**Lesson 9: Model Distillation and Compression** — Learn to transfer knowledge from large models to small, fast models that are cheaper to run in production.

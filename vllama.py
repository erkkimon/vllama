#!/usr/bin/env python3.12
"""
vllama - vLLM + Ollama hybrid server
Fast inference with Ollama model management
OpenAI-compatible API on port 11435
"""

import os
import time
import requests
import subprocess
import json
from threading import Thread, Lock
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vllm import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
import uvicorn

app = FastAPI(title="vllama", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1

# Global vLLM management
engine = None
engine_lock = Lock()
last_activity = time.time()
IDLE_TIMEOUT = 300  # 5 minutes
OLLAMA_URL = "http://localhost:11434"

def unload_vLLM():
    """Background thread to unload vLLM after idle timeout"""
    global engine, last_activity
    while True:
        time.sleep(60)
        with engine_lock:
            if engine and time.time() - last_activity > IDLE_TIMEOUT:
                print(f"[{time.strftime('%H:%M:%S')}] Unloading vLLM due to idle timeout")
                del engine
                engine = None

# Start unload monitor
Thread(target=unload_vLLM, daemon=True).start()

def get_ollama_model_path(model_id: str) -> str:
    """Extract GGUF path from Ollama model"""
    try:
        # Use ollama show to get model info
        result = subprocess.run(
            ["ollama", "show", model_id, "--modelfile"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        
        # Parse FROM line to get blob path
        for line in result.stdout.split('\n'):
            if line.strip().startswith('FROM'):
                blob_path = line.strip().split()[-1]
                if blob_path.startswith('/var/lib/ollama/blobs/'):
                    return blob_path
                elif blob_path.startswith('sha256-'):
                    # Convert hash to full path
                    return f"/var/lib/ollama/blobs/{blob_path}"
        
        raise ValueError(f"Could not parse model path from: {model_id}")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in Ollama")
    except Exception as e:
        print(f"Error getting model path for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model path: {e}")

def load_vLLM(model_id: str):
    """Load vLLM engine for model (lazy loading)"""
    global engine, last_activity
    with engine_lock:
        if not engine:
            print(f"[{time.strftime('%H:%M:%S')}] Loading vLLM for model: {model_id}")
            model_path = get_ollama_model_path(model_id)
            
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            engine_args = AsyncEngineArgs(
                model=model_path,
                tokenizer=model_path,  # Use GGUF tokenizer
                max_model_len=47856,
                gpu_memory_utilization=0.95,
                enforce_eager=True,
                disable_log_stats=True,
            )
            
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            print(f"[{time.strftime('%H:%M:%S')}] vLLM loaded successfully")
        
        last_activity = time.time()
        return engine

@app.get("/v1/models")
async def list_models_endpoint():
    """Proxy to Ollama's /api/tags endpoint"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        
        ollama_models = response.json().get("models", [])
        
        # Convert to OpenAI format
        openai_models = []
        for model in ollama_models:
            model_id = model.get("name", "")
            # Extract size from modelfile if available
            size_info = model.get("modelfile", "").get("parameter", {}).get("size", "unknown")
            
            openai_models.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllama",
                "size": size_info if size_info != "unknown" else None
            })
        
        return {
            "object": "list",
            "data": openai_models
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {e}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """vLLM inference with Ollama model management"""
    try:
        # Lazy load vLLM engine
        engine = load_vLLM(request.model)
        
        # Convert messages to Devstral prompt format
        prompt = build_devstral_prompt(request.messages)
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            repetition_penalty=request.repetition_penalty,
            stop=["</s>", "[/INST]"]
        )
        
        # Generate with vLLM
        results = await engine.generate(prompt, sampling_params, request.model)
        
        if not results or not results[0].outputs:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        generated_text = results[0].outputs[0].text.strip()
        
        # Estimate token counts (rough)
        prompt_tokens = len(prompt.split()) // 4  # Rough estimate
        completion_tokens = len(generated_text.split()) // 4
        
        return {
            "id": f"chatcmpl-{int(time.time())}-{hash(request.model) % 10000}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "system_fingerprint": None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

def build_devstral_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI messages to Devstral chat format"""
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            prompt += f"[SYSTEM_PROMPT]{content}[/SYSTEM_PROMPT]\n"
        elif role == "user":
            prompt += f"[INST]{content}[/INST]\n"
        elif role == "assistant":
            prompt += f"{content}</s>\n"
    
    return prompt.strip()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global engine
    with engine_lock:
        engine_status = "loaded" if engine else "unloaded"
    return {
        "status": "healthy",
        "engine": engine_status,
        "last_activity": last_activity,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

@app.get("/v1")
async def openai_root():
    """OpenAI compatibility root"""
    return {"message": "vllama OpenAI-compatible API", "port": 11435}

if __name__ == "__main__":
    print("Starting vllama server on http://0.0.0.0:11435")
    print("Model management via Ollama (port 11434)")
    print("Lazy loading: vLLM unloads after 5min idle")
    uvicorn.run(app, host="0.0.0.0", port=11435, log_level="info")
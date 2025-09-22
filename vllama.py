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
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vllm import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import uvicorn
import traceback
import torch
from fastapi.responses import StreamingResponse

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup VRAM cleanup - unload any existing engine
    global engine
    async with engine_lock:
        if engine:
            print(f"[{time.strftime('%H:%M:%S')}] Startup cleanup: Unloading existing vLLM engine")
            del engine
            engine = None
            torch.cuda.empty_cache()
    
    # Start unload monitor
    asyncio.create_task(unload_vLLM_task())
    
    print("Starting vllama server on http://0.0.0.0:11435")
    print("Model management via Ollama (port 11434)")
    print(f"Lazy loading: vLLM unloads after {IDLE_TIMEOUT}s idle")
    yield

app = FastAPI(title="vllama", version="0.1.0", lifespan=lifespan)

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
    stream: bool = False

# Global vLLM management
engine = None
engine_lock = asyncio.Lock()
last_activity = time.time()
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", 300))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


async def unload_vLLM_task():
    """Background task to unload vLLM after idle timeout"""
    global engine, last_activity
    while True:
        await asyncio.sleep(60)
        async with engine_lock:
            if engine and time.time() - last_activity > IDLE_TIMEOUT:
                print(f"[{time.strftime('%H:%M:%S')}] Unloading vLLM due to idle timeout")
                del engine
                engine = None
                torch.cuda.empty_cache()


def get_ollama_model_path(model_id: str) -> str:
    """Extract GGUF path from Ollama model using ollama show"""
    try:
        print(f"[DEBUG] Getting path for model: {model_id}")
        
        # Use ollama show to get model info
        result = subprocess.run(
            ["ollama", "show", model_id, "--modelfile"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        
        print(f"[DEBUG] ollama show output: {result.stdout[:500]}...")  # Expanded to 500 chars for full FROM line
        
        # Parse FROM line to get blob path
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith('FROM'):
                blob_path = line.split(maxsplit=1)[1].strip()
                print(f"[DEBUG] Found FROM line: {blob_path}")
                
                # Normalize path - handle different Ollama installations
                if os.path.exists(blob_path):
                    print(f"[DEBUG] Direct path exists: {blob_path}")
                    return blob_path
                
                # Handle relative or hashed paths
                if blob_path.startswith('sha256:') or blob_path.startswith('sha256-'):
                    blob_hash = blob_path.split('-')[-1] if '-' in blob_path else blob_path.split(':')[-1]
                    print(f"[DEBUG] Extracted hash: {blob_hash}")
                    
                    # Common Ollama blob locations (generalized for different OS/users)
                    possible_dirs = [
                        '/var/lib/ollama/models/blobs',
                        '/usr/share/ollama/.ollama/models/blobs',
                        os.path.expanduser('~/.ollama/models/blobs'),
                        '/home/ollama/.ollama/models/blobs',
                        '/opt/ollama/models/blobs'
                    ]
                    
                    for dir_path in possible_dirs:
                        candidate = os.path.join(dir_path, f"sha256-{blob_hash}" if not blob_hash.startswith('sha256-') else blob_hash)
                        if os.path.exists(candidate):
                            print(f"[DEBUG] Found model at: {candidate}")
                            return candidate
                        
                    print(f"[ERROR] Could not find blob file for hash: {blob_hash}")
                    print(f"[DEBUG] Searched directories: {possible_dirs}")
                    raise FileNotFoundError(f"Blob file not found for hash: {blob_hash}")
                
                print(f"[DEBUG] Unexpected path format: {blob_path}")
        
        # Fallback: Use ollama inspect for digest
        print("[DEBUG] Trying ollama inspect as fallback")
        inspect_result = subprocess.run(
            ["ollama", "inspect", model_id],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        
        try:
            inspect_data = json.loads(inspect_result.stdout)
            model_data = inspect_data.get(model_id, {})
            digest = model_data.get("digest", "")
            print(f"[DEBUG] Inspect digest: {digest}")
            
            if digest:
                # Search for digest in common locations
                blob_hash = digest.split(':')[-1] if ':' in digest else digest
                possible_dirs = [
                    '/var/lib/ollama/models/blobs',
                    '/usr/share/ollama/.ollama/models/blobs',
                    os.path.expanduser('~/.ollama/models/blobs'),
                    '/home/ollama/.ollama/models/blobs',
                    '/opt/ollama/models/blobs'
                ]
                
                for dir_path in possible_dirs:
                    candidate = os.path.join(dir_path, f"sha256-{blob_hash}")
                    if os.path.exists(candidate):
                        print(f"[DEBUG] Found model at: {candidate}")
                        return candidate
                
                print(f"[ERROR] Could not find model file for digest: {digest}")
                raise FileNotFoundError(f"Model file not found for digest: {digest}")
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse ollama inspect JSON: {e}")
            print(f"[DEBUG] Inspect output: {inspect_result.stdout[:500]}...")
        
        raise ValueError(f"Could not locate GGUF file for model: {model_id}")
    
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ollama show failed: {e}")
        print(f"[DEBUG] Return code: {e.return_code}, stderr: {e.stderr}")
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in Ollama: {e.stderr}")
    except FileNotFoundError as e:
        print(f"[ERROR] Model file not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model file not found: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error getting model path: {e}")
        import traceback
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load model path: {e}")

async def load_vLLM(model_id: str):
    """Load vLLM engine for model (lazy loading)"""
    global engine, last_activity
    async with engine_lock:
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
        
        ollama_data = response.json()
        
        # Debug: Print raw Ollama response
        print(f"[DEBUG] Ollama /api/tags response: {json.dumps(ollama_data, indent=2)}")
        
        # Handle both old and new Ollama response formats
        if "models" in ollama_data:
            ollama_models = ollama_data["models"]
        else:
            # Fallback for older Ollama versions
            ollama_models = ollama_data.get("models", [])
        
        print(f"[DEBUG] Found {len(ollama_models)} models")
        
        # Convert to OpenAI format
        openai_models = []
        for model in ollama_models:
            model_id = model.get("name", "")
            if not model_id:
                print(f"[DEBUG] Skipping model with no name: {model}")
                continue
                
            # Get size if available
            size_bytes = model.get("size", 0)
            size_gb = round(size_bytes / (1024**3), 1) if size_bytes else None
            
            model_info = {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllama",
            }
            
            if size_gb:
                model_info["size"] = f"{size_gb}GB"
            
            openai_models.append(model_info)
            print(f"[DEBUG] Added model: {model_id} ({size_gb}GB if available)")
        
        response_data = {
            "object": "list",
            "data": openai_models
        }
        
        print(f"[DEBUG] Returning {len(openai_models)} models")
        return response_data
        
    except requests.RequestException as e:
        print(f"[ERROR] Ollama request failed: {e}")
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {e}")
    except Exception as e:
        import traceback
        print(f"[ERROR] Unexpected error in list_models_endpoint: {e}")
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {e}")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """vLLM inference with Ollama model management"""
    try:
        # Lazy load vLLM engine
        engine = await load_vLLM(request.model)
        
        # Convert messages to model-specific prompt format
        prompt = build_model_prompt(request.messages, request.model)
        print(f"[DEBUG] Generated prompt: {prompt[:200]}...")
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            repetition_penalty=request.repetition_penalty,
            stop=["</s>", "[/INST]"]
        )
        
        # Generate with vLLM
        results_generator = engine.generate(prompt, sampling_params, request_id=random_uuid())
        
        if request.stream:
            async def stream_results():
                async for result in results_generator:
                    delta = result.outputs[0].text
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': delta}}]})}\n\n"
            return StreamingResponse(stream_results(), media_type="text/event-stream")
        else:
            final_output = None
            async for result in results_generator:
                final_output = result

            if not final_output or not final_output.outputs:
                raise HTTPException(status_code=500, detail="Generation failed - no output")
            
            generated_text = final_output.outputs[0].text.strip()
            print(f"[DEBUG] Generated {len(generated_text)} characters")
            
            # Estimate token counts (rough)
            prompt_tokens = len(prompt.split()) // 4  # Rough estimate
            completion_tokens = len(generated_text.split()) // 4
            
            return {
                "id": f"chatcmpl-{int(time.time())}-{random_uuid()[:8]}",
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
                    "finish_reason": final_output.outputs[0].finish_reason if final_output.outputs[0].finish_reason else "stop"
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
        print(f"[ERROR] Generation error: {e}")
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Generalized prompt builder
def build_model_prompt(messages: List[Dict[str, str]], model_id: str) -> str:
    """Generalized prompt builder - detect model family from Ollama"""
    try:
        # Get model family from Ollama inspect
        inspect_result = subprocess.run(
            ["ollama", "inspect", model_id],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        inspect_data = json.loads(inspect_result.stdout)
        model_data = inspect_data.get(model_id, {})
        family = model_data.get("details", {}).get("family", "llama").lower()
        print(f"[DEBUG] Detected model family: {family}")
    except Exception as e:
        print(f"[DEBUG] Could not detect family, using default 'llama': {e}")
        family = "llama"
    
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if family in ["llama", "mistral", "devstral"]:
            # ChatML format for Llama/Mistral family
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif family in ["gpt", "openai"]:
            # GPT-style format
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        else:
            # Default simple format
            role_prefix = {"system": "System", "user": "User", "assistant": "Assistant"}
            prompt += f"{role_prefix.get(role, role)}: {content}\n"
    
    return prompt.strip()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global engine
    async with engine_lock:
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

# Manual unload endpoint
@app.post("/unload")
async def manual_unload():
    """Manual unload of vLLM engine (cleans VRAM)"""
    global engine
    async with engine_lock:
        if engine:
            print(f"[{time.strftime('%H:%M:%S')}] Manual unload triggered via /unload")
            del engine
            engine = None
            torch.cuda.empty_cache()
            return {"status": "unloaded", "message": "vLLM engine unloaded from VRAM"}
        return {"status": "already unloaded", "message": "No engine to unload"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11435, log_level="info")
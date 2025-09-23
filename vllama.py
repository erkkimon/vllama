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
import re
from typing import List, Dict, Any, Optional, Tuple
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
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = None  # configurable default applied later
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    stream: bool = False
    stream_options: dict | None = None
    reasoning_effort: str | None = None
    tools: List[Dict[str, Any]] | None = None
    tool_choice: str | Dict[str, Any] | None = None

# Global vLLM management
engine = None
engine_lock = asyncio.Lock()
last_activity = time.time()
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", 300))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_TOKENS_DEFAULT = int(os.environ.get("MAX_TOKENS_DEFAULT", 1024))  # default completion budget
DEFAULT_CONTEXT_WINDOW = int(os.environ.get("DEFAULT_CONTEXT_WINDOW", 65536))  # fallback context window (64k default)
REPORTED_CONTEXT_WINDOW = int(os.environ.get("REPORTED_CONTEXT_WINDOW", 65536))  # context window reported to clients (64k default)
DEVSTRAL_CONTEXT_WINDOW = int(os.environ.get("DEVSTRAL_CONTEXT_WINDOW", 65536))  # context window for Devstral models (64k default)
current_max_model_len = None

# ---------- Output sanitization to strip meta/thoughts ----------
# We remove blocks even if they are UN-CLOSED, stopping at the next bracket tag, a blank line, or end.
THINK_TAGS = r"(?:thinking|plan|scratchpad|analysis|internal|note|notes|reflection|deliberate)"
THINKY_PATTERNS = [
    # Properly closed blocks: [thinking] ... [/thinking]
    rf"(?is)\[\s*{THINK_TAGS}\s*\](.*?)\[/\s*{THINK_TAGS}\s*\]",
    # Open block until next bracket tag like [ASSISTANT] or [TOOL_CALLS] or [/something]
    rf"(?is)\[\s*{THINK_TAGS}\s*\][\s\S]*?(?=(\n\[[A-Za-z_\/]+)|\Z)",
    # Open block until the next blank line (two consecutive newlines)
    rf"(?is)\[\s*{THINK_TAGS}\s*\][^\S\r\n]*[\s\S]*?(?=\n\s*\n|\Z)",
    # Standalone tags on their own line
    rf"(?im)^\s*\[\s*{THINK_TAGS}\s*\]\s*$",
    # Boilerplate we’ve seen in DevStral-ish dumps
    r"(?is)\bTask\s+Completed\b.*$",
    r"(?im)^\s*Roo has a question.*?(?=\n\s*\n|\Z)",
]

def sanitize_visible_text(txt: str, *, for_stream: bool = False) -> str:
    """
    Remove hidden-thought/meta blocks and tidy whitespace.
    IMPORTANT: when for_stream=True, DO NOT trim leading/trailing whitespace,
    otherwise we destroy spaces that span chunk boundaries.
    """
    if not txt:
        return txt
    for pat in THINKY_PATTERNS:
        txt = re.sub(pat, "", txt, flags=re.IGNORECASE | re.DOTALL)
    # remove stray tag-only lines
    txt = re.sub(r"(?im)^\s*\[[A-Za-z_\/]+\]\s*$", "", txt)
    # collapse only excessive blank lines; keep single newlines and spaces
    txt = re.sub(r"\n{3,}", "\n\n", txt, flags=re.DOTALL)
    if not for_stream:
        txt = txt.strip()
    return txt

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
        result = subprocess.run(
            ["ollama", "show", model_id, "--modelfile"],
            capture_output=True, text=True, check=True, timeout=10
        )
        print(f"[DEBUG] ollama show output: {result.stdout[:500]}...")
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith('FROM'):
                blob_path = line.split(maxsplit=1)[1].strip()
                print(f"[DEBUG] Found FROM line: {blob_path}")
                if os.path.exists(blob_path):
                    print(f"[DEBUG] Direct path exists: {blob_path}")
                    return blob_path
                if blob_path.startswith('sha256:') or blob_path.startswith('sha256-'):
                    blob_hash = blob_path.split('-')[-1] if '-' in blob_path else blob_path.split(':')[-1]
                    print(f"[DEBUG] Extracted hash: {blob_hash}")
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
        print("[DEBUG] Trying ollama inspect as fallback")
        inspect_result = subprocess.run(
            ["ollama", "inspect", model_id],
            capture_output=True, text=True, check=True, timeout=10
        )
        try:
            inspect_data = json.loads(inspect_result.stdout)
            model_data = inspect_data.get(model_id, {})
            digest = model_data.get("digest", "")
            print(f"[DEBUG] Inspect digest: {digest}")
            if digest:
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
    global engine, last_activity, current_max_model_len
    async with engine_lock:
        if not engine:
            print(f"[{time.strftime('%H:%M:%S')}] Loading vLLM for model: {model_id}")
            model_path = get_ollama_model_path(model_id)
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            # Use environment variable for max_model_len, default to 53840 for VRAM compatibility
            max_model_len = int(os.environ.get("MAX_MODEL_LEN", 53840))
            if max_model_len == 0:
                max_model_len = None  # Let vLLM auto-detect
            
            engine_args = AsyncEngineArgs(
                model=model_path,
                tokenizer=model_path,  # Use GGUF tokenizer
                gpu_memory_utilization=0.95,
                enforce_eager=True,
                disable_log_stats=True,
                max_model_len=max_model_len,
            )
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Get the actual max_model_len from the engine args or use the configured value
            if hasattr(engine_args, 'max_model_len') and engine_args.max_model_len:
                current_max_model_len = engine_args.max_model_len
            else:
                current_max_model_len = max_model_len or DEFAULT_CONTEXT_WINDOW
            
            print(f"[{time.strftime('%H:%M:%S')}] vLLM loaded successfully with max_model_len: {current_max_model_len}")
        last_activity = time.time()
        return engine

def get_model_specific_context_window(model_id: str) -> int:
    """Get context window for specific model types with environment variable override"""
    model_lower = model_id.lower()
    
    # Devstral model detection (covers huihui_ai/devstral-abliterated:latest and variants)
    if any(keyword in model_lower for keyword in [
        "devstral",
        "huihui_ai/devstral",
        "devstral-abliterated"
    ]):
        print(f"[DEBUG] Detected Devstral model: {model_id}, using context window: {DEVSTRAL_CONTEXT_WINDOW}")
        return DEVSTRAL_CONTEXT_WINDOW
    
    # Fallback to global setting
    return REPORTED_CONTEXT_WINDOW

def get_context_window_for_model(model_id: str) -> int:
    """Get context window for a model, using actual vLLM engine if loaded, or fallback"""
    global current_max_model_len
    
    # If engine is loaded and matches the requested model, use actual context
    if engine and current_max_model_len:
        return current_max_model_len
    
    # Otherwise, use model-specific context window detection
    return get_model_specific_context_window(model_id)

def get_reported_context_window_for_model(model_id: str) -> int:
    """Get context window to report to clients (can be different from actual vLLM context)"""
    # Use model-specific context window detection instead of generic REPORTED_CONTEXT_WINDOW
    return get_model_specific_context_window(model_id)

@app.get("/v1/models")
async def list_models_endpoint():
    """Proxy to Ollama's /api/tags endpoint, formatted for OpenAI and LiteLLM compatibility"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        ollama_data = response.json()
        
        if "models" not in ollama_data:
            print("[WARN] No 'models' key in Ollama response.")
            return {"object": "list", "data": []}

        openai_models = []
        for model in ollama_data["models"]:
            model_id = model.get("name")
            if not model_id:
                continue

            context_window = get_reported_context_window_for_model(model_id)
            
            # This format is for standard OpenAI API compatibility
            openai_models.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllama",
                "context_window": context_window,
            })

        return {"object": "list", "data": openai_models}

    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {e}")

def _parse_tool_choice(tool_choice: Any, tools_present: bool) -> Tuple[bool, Optional[str]]:
    """
    Returns (tool_required, required_function_name)
    Policy:
      - If tool_choice == "none": not required
      - If tool_choice == "required": required
      - If tool_choice is {"type":"function", "function":{"name":...}}: required that function
      - Else: if tools are present and env FORCE_TOOL_FIRST_WHEN_TOOLS!=0, treat as required
    """
    if tool_choice == "none":
        return False, None
    if tool_choice == "required":
        return True, None
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return True, tool_choice.get("function", {}).get("name")

    # Heuristic default for Roo/Cline: if tools are present but tool_choice omitted, require tool-first.
    force_when_tools = os.environ.get("FORCE_TOOL_FIRST_WHEN_TOOLS", "1").lower() not in ("0", "false", "no", "off")
    if tools_present and force_when_tools:
        return True, None
    return False, None

@app.post("/chat/completions")
async def chat_completions_root(request: ChatCompletionRequest):
    """LiteLLM compatibility endpoint for chat completions at root path"""
    return await chat_completions(request)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """vLLM inference with Ollama model management"""
    try:
        # Determine tool requirement
        tool_required, required_function_name = _parse_tool_choice(request.tool_choice, bool(request.tools))
        print(f"[DEBUG] tools_present={bool(request.tools)} tool_choice={request.tool_choice!r} "
              f"tool_required={tool_required} required_function={required_function_name}")

        # Lazy load vLLM engine
        engine = await load_vLLM(request.model)

        # Convert messages to model-specific prompt format
        prompt, seeded_tool_prefix = build_model_prompt(
            request.messages,
            request.model,
            request.tools,
            force_tools=tool_required,
            required_function_name=required_function_name
        )
        print(f"[DEBUG] Generated prompt: {prompt[:200]}...")
        if tool_required and seeded_tool_prefix:
            print("[DEBUG] Prompt seeded with [ASSISTANT][TOOL_CALLS] prefix for tool-first enforcement.")

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or MAX_TOKENS_DEFAULT,  # use default if client omitted
            repetition_penalty=request.repetition_penalty,
            stop=[
                "</s>", "[/INST]", "<|im_start|>assistant",
                "[/ASSISTANT]",  # helps DevStral-like tags
                "[thinking]", "[/thinking]", "[plan]", "[/plan]",
                "[scratchpad]", "[/scratchpad]", "[internal]", "[/internal]",
                "Roo has a question",
                "Roo wants to read this file",
                "[USER][ERROR]",
                "Roo is having trouble..."
            ]
        )

        # Generate with vLLM
        results_generator = engine.generate(prompt, sampling_params, request_id=random_uuid())

        if request.stream:
            async def stream_results():
                previous_text = ""
                request_id_ = random_uuid()
                created_time = int(time.time())

                def sse_delta_content(text: str) -> str:
                    return (
                        "data: "
                        + json.dumps({
                            'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                            'object':'chat.completion.chunk',
                            'created': created_time,
                            'model': request.model,
                            'choices':[{'index':0,'delta': {'content': text},'finish_reason': None}]
                        })
                        + "\n\n"
                    )

                def sse_delta_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
                    return (
                        "data: "
                        + json.dumps({
                            'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                            'object':'chat.completion.chunk',
                            'created': created_time,
                            'model': request.model,
                            'choices':[{'index':0,'delta': {'tool_calls': tool_calls},'finish_reason': None}]
                        })
                        + "\n\n"
                    )

                # Initial role chunk
                yield "data: " + json.dumps({
                    'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': request.model,
                    'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]
                }) + "\n\n"

                buffer = ""
                tool_calls_sent = False

                async for result in results_generator:
                    new_text = result.outputs[0].text
                    delta = new_text[len(previous_text):]
                    previous_text = new_text
                    buffer += delta

                    # Branch A: classic [TOOL_CALLS][ ... ] in generated text
                    match_tagged = re.search(r"\[TOOL_CALLS\]\[(.*?)\]", buffer, re.DOTALL)

                    # Branch B: seeded prefix -> generation starts with just a JSON array
                    match_seeded = None
                    if (not match_tagged) and (tool_required and seeded_tool_prefix):
                        # Look for a leading [...] array
                        m = re.search(r"^\s*\[(.*?)\]", buffer, re.DOTALL)
                        if m:
                            match_seeded = m

                    if (match_tagged or match_seeded) and not tool_calls_sent:
                        if match_tagged:
                            before = buffer[:match_tagged.start()]
                            json_str = match_tagged.group(1)
                            after = buffer[match_tagged.end():]
                            which = "tagged"
                        else:
                            before = buffer[:match_seeded.start()]
                            json_str = match_seeded.group(1)
                            after = buffer[match_seeded.end():]
                            which = "seeded"
                        print(f"[DEBUG] Detected tool-call block via {which} branch.")

                        # If tools are required, suppress pre-tool prose
                        if not tool_required:
                            clean_before = sanitize_visible_text(before, for_stream=True)
                            if clean_before:
                                yield sse_delta_content(clean_before)

                        # Parse and emit tool_calls
                        try:
                            parsed_tool_calls = json.loads(f"[{json_str}]")
                            if required_function_name:
                                if not any(tc.get("name") == required_function_name for tc in parsed_tool_calls):
                                    print(f"[WARN] Required function '{required_function_name}' not present in TOOL_CALLS.")
                            tool_calls = []
                            for tc in parsed_tool_calls:
                                tool_calls.append({
                                    "id": random_uuid(),
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc.get("arguments", {}))
                                    }
                                })
                            yield sse_delta_tool_calls(tool_calls)
                            tool_calls_sent = True
                        except json.JSONDecodeError:
                            # If JSON is malformed, emit sanitized literal as content only if tools aren't strictly required
                            fallback = sanitize_visible_text("[TOOL_CALLS][" + json_str + "]", for_stream=True)
                            if fallback and not tool_required:
                                yield sse_delta_content(fallback)

                        # Continue after the tool block; DO NOT re-emit the block
                        buffer = after

                    # Emit sanitized content ONLY if tools are not required, or after tool calls were sent
                    clean = sanitize_visible_text(buffer, for_stream=True)
                    if clean and (not tool_required or tool_calls_sent):
                        yield sse_delta_content(clean)
                        buffer = ""

                    if result.finished:
                        # Calculate token usage for final chunk
                        prompt_tokens = len(prompt.split()) // 4  # Rough estimate
                        completion_tokens = len(previous_text.split()) // 4
                        
                        # Try to get more accurate token counts from vLLM
                        try:
                            # For AsyncLLMEngine, we need to access the tokenizer differently
                            if hasattr(engine, 'get_tokenizer'):
                                tokenizer = engine.get_tokenizer()
                                if hasattr(tokenizer, 'encode'):
                                    prompt_tokens = len(tokenizer.encode(prompt))
                                    completion_tokens = len(tokenizer.encode(previous_text))
                        except Exception as e:
                            print(f"[DEBUG] Could not use vLLM tokenizer for streaming token counting: {e}")
                        
                        total_tokens = prompt_tokens + completion_tokens
                        actual_context_window = current_max_model_len or DEFAULT_CONTEXT_WINDOW
                        reported_context_window = get_reported_context_window_for_model(request.model)
                        context_usage_percent = (total_tokens / reported_context_window) * 100 if reported_context_window > 0 else 0
                        
                        print(f"[DEBUG] Streaming context usage: {total_tokens}/{reported_context_window} tokens ({context_usage_percent:.1f}%) [actual: {actual_context_window}]")
                        
                        finish_reason = result.outputs[0].finish_reason if result.outputs[0].finish_reason else "stop"
                        
                        # Final chunk with usage information
                        yield "data: " + json.dumps({
                            'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                            'object': 'chat.completion.chunk',
                            'created': created_time,
                            'model': request.model,
                            'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}],
                            'usage': {
                                'prompt_tokens': prompt_tokens,
                                'completion_tokens': completion_tokens,
                                'total_tokens': total_tokens,
                                'context_window': reported_context_window,
                                'context_usage_percent': round(context_usage_percent, 1)
                            }
                        }) + "\n\n"
                        yield "data: [DONE]\n\n"  # End of stream
            return StreamingResponse(stream_results(), media_type="text/event-stream")
        else:
            final_output = None
            async for result in results_generator:
                final_output = result

            if not final_output or not final_output.outputs:
                raise HTTPException(status_code=500, detail="Generation failed - no output")

            generated_text = final_output.outputs[0].text.strip()

            # Extract tool calls (if any) and sanitize content
            tool_calls: List[Dict[str, Any]] = []
            content = generated_text

            # Branch A: classic [TOOL_CALLS][...]
            tool_calls_pattern = r"\[TOOL_CALLS\]\[(.*?)\]"
            match = re.search(tool_calls_pattern, generated_text, re.DOTALL)

            # Branch B: seeded prefix -> generation is just a JSON array
            if not match and (tool_required and seeded_tool_prefix):
                m2 = re.search(r"^\s*\[(.*?)\]\s*$", generated_text, re.DOTALL)
                if m2:
                    print("[DEBUG] Non-stream detected tool-call block via seeded branch.")
                    try:
                        parsed = json.loads("[" + m2.group(1) + "]")
                        for tc in parsed:
                            tool_calls.append({
                                "id": random_uuid(),
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments", {}))}
                            })
                        content = ""  # no user-visible prose when tools are used
                    except json.JSONDecodeError:
                        pass  # fall through to normal handling

            if match and not tool_calls:
                json_str = match.group(1)
                try:
                    parsed_tool_calls = json.loads(json_str)
                    if required_function_name and not any(tc.get("name") == required_function_name for tc in parsed_tool_calls):
                        print(f"[WARN] Required function '{required_function_name}' not found in TOOL_CALLS (non-stream).")
                    for tc in parsed_tool_calls:
                        tool_calls.append({
                            "id": random_uuid(),
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments", {}))}
                        })
                    # Remove the tool calls from the visible content
                    content = re.sub(tool_calls_pattern, "", generated_text, 1, re.DOTALL).strip()
                except json.JSONDecodeError:
                    print(f"[ERROR] Failed to parse tool calls JSON: {json_str}")
                    # Keep content as-is if parsing fails

            # Enforce tool requirement in non-stream mode
            if tool_required and not tool_calls:
                raise HTTPException(
                    status_code=400,
                    detail="Tool use required by client (tool_choice) but no tool calls were produced."
                )

            content = sanitize_visible_text(content, for_stream=False)

            print(f"[DEBUG] Generated {len(generated_text)} characters")

            # Get actual token counts from vLLM if available, otherwise estimate
            prompt_tokens = len(prompt.split()) // 4  # Rough estimate
            completion_tokens = len(generated_text.split()) // 4
            
            # Try to get more accurate token counts from vLLM
            try:
                # For AsyncLLMEngine, we need to access the tokenizer differently
                if hasattr(engine, 'get_tokenizer'):
                    tokenizer = engine.get_tokenizer()
                    if hasattr(tokenizer, 'encode'):
                        prompt_tokens = len(tokenizer.encode(prompt))
                        completion_tokens = len(tokenizer.encode(generated_text))
                        print(f"[DEBUG] Accurate token count - prompt: {prompt_tokens}, completion: {completion_tokens}")
            except Exception as e:
                print(f"[DEBUG] Could not use vLLM tokenizer for token counting: {e}")
                # Fall back to rough estimate

            total_tokens = prompt_tokens + completion_tokens
            
            # Log context usage for debugging
            actual_context_window = current_max_model_len or DEFAULT_CONTEXT_WINDOW
            reported_context_window = get_reported_context_window_for_model(request.model)
            context_usage_percent = (total_tokens / reported_context_window) * 100 if reported_context_window > 0 else 0
            print(f"[DEBUG] Context usage: {total_tokens}/{reported_context_window} tokens ({context_usage_percent:.1f}%) [actual: {actual_context_window}]")

            return {
                "id": f"chatcmpl-{int(time.time())}-{random_uuid()[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content if content else None, "tool_calls": tool_calls or None},
                    "finish_reason": final_output.outputs[0].finish_reason if final_output.outputs[0].finish_reason else "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "context_window": reported_context_window,  # Add context window info for Roo Code
                    "context_usage_percent": round(context_usage_percent, 1)  # Add usage percentage
                },
                "logprobs": None,
                "system_fingerprint": None
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Generation error: {e}")
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# ---------- Prompt builder (DevStral-aware, with force-tools mode) ----------
def build_model_prompt(
    messages: List[Dict[str, Any]],
    model_id: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    *,
    force_tools: bool = False,
    required_function_name: Optional[str] = None
) -> Tuple[str, bool]:
    """
    Builds a prompt. If the model looks like DevStral, use DevStral tag format:
      [SYSTEM_PROMPT]...[/SYSTEM_PROMPT]
      [AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS]
      [USER]...[/USER]
      [ASSISTANT]...[/ASSISTANT]
    Returns (prompt, seeded_tool_prefix) where seeded_tool_prefix==True
    when the prompt ends with [ASSISTANT][TOOL_CALLS] to force tool-call-first.
    """
    def only_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(part.get("text","") for part in content if part.get("type") == "text")
        return ""

    model_lc = (model_id or "").lower()
    devstral_like = any(x in model_lc for x in ["devstral", "huihui_ai", "abliterated"])

    seeded_tool_prefix = False

    if devstral_like:
        sys_lines = ["You are a fast, helpful assistant."]
        if force_tools:
            sys_lines += [
                "TOOLS ARE REQUIRED for this reply.",
                "Output ONLY a single tool call JSON array.",
                "Do not output any other text."
            ]
            if required_function_name:
                sys_lines.append(
                    f"You MUST call the function named \"{required_function_name}\" in the tool calls."
                )
        else:
            sys_lines += [
                "Do NOT output analysis, plans, or hidden thoughts.",
                "If you use a tool, output ONLY [TOOL_CALLS][…]. Otherwise output ONLY the final answer text.",
            ]

        sys_msg = next((m for m in messages if m.get("role") == "system"), None)
        if sys_msg:
            sys_lines.append(only_text(sys_msg.get("content")))
        system_block = f"[SYSTEM_PROMPT]{'\n'.join(sys_lines)}[/SYSTEM_PROMPT]\n"

        tools_block = ""
        if tools:
            tools_block = f"[AVAILABLE_TOOLS]{json.dumps(tools)}[/AVAILABLE_TOOLS]\n"

        conv = []
        for m in messages:
            role = m.get("role")
            content = only_text(m.get("content"))
            if not content:
                continue
            if role == "system":
                continue  # already included
            elif role == "user":
                conv.append(f"[USER]{content}[/USER]")
            elif role == "assistant":
                conv.append(f"[ASSISTANT]{content}[/ASSISTANT]")
            else:
                conv.append(f"[USER]{content}[/USER]")

        # For tool-required flows, seed the assistant with [TOOL_CALLS] to bias DevStral to emit only the JSON array.
        if force_tools:
            prompt = system_block + tools_block + "\n".join(conv) + "\n[ASSISTANT][TOOL_CALLS]"
            seeded_tool_prefix = True
        else:
            prompt = system_block + tools_block + "\n".join(conv) + "\n[ASSISTANT]"

        return prompt, seeded_tool_prefix
    else:
        # Fallback: Llama chat style with optional force_tools hint baked into system
        prompt_parts = []
        system_instruction = "You are a helpful assistant. You can use tools to assist the user."
        if force_tools:
            system_instruction += "\nTOOLS ARE REQUIRED. Do not output any text except a valid tool call representation."
        if tools:
            tools_json = json.dumps(tools)
            system_instruction += f"\n[AVAILABLE_TOOLS]{tools_json}[/AVAILABLE_TOOLS]"
        processed_messages = []
        if messages and messages[0].get("role") == "system":
            processed_messages.append({"role": "system", "content": system_instruction + "\n" + only_text(messages[0]["content"])})
            processed_messages.extend(messages[1:])
        else:
            processed_messages.append({"role": "system", "content": system_instruction})
            processed_messages.extend(messages)
        for message in processed_messages:
            role = message["role"]
            content = only_text(message["content"])
            if role == "system":
                prompt_parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(f"{content}")
        if force_tools:
            # Seed assistant tool-calls start token for non-devstral too
            return "<s>" + " ".join(prompt_parts) + "</s>" + "\n[TOOL_CALLS]", True
        return "<s>" + " ".join(prompt_parts) + "</s>", False

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global engine, current_max_model_len
    async with engine_lock:
        engine_status = "loaded" if engine else "unloaded"
    return {
        "status": "healthy",
        "engine": engine_status,
        "context_window": get_reported_context_window_for_model("health_check"),
        "last_activity": last_activity,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

@app.get("/v1/context")
async def get_context_info():
    """Get current context window and usage information"""
    global engine, current_max_model_len
    async with engine_lock:
        reported_context = get_reported_context_window_for_model("context_check")
        if engine and current_max_model_len:
            return {
                "context_window": reported_context,
                "actual_context_window": current_max_model_len,
                "engine_loaded": True,
                "model": "loaded_model"
            }
        else:
            return {
                "context_window": reported_context,
                "actual_context_window": DEFAULT_CONTEXT_WINDOW,
                "engine_loaded": False,
                "model": None
            }

@app.get("/v1/model/info")
async def litellm_model_info():
    """LiteLLM-compatible model info endpoint for dynamic context window reporting"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        ollama_data = response.json()
        
        if "models" not in ollama_data:
            print("[WARN] No 'models' key in Ollama response.")
            return {"data": []}

        model_data = []
        for model in ollama_data["models"]:
            model_id = model.get("name")
            if not model_id:
                continue
            
            context_window = get_reported_context_window_for_model(model_id)
            
            model_data.append({
                "model_name": model_id, # This is the ID that Roo Code uses
                "model_info": {
                    "max_tokens": MAX_TOKENS_DEFAULT,
                    "max_input_tokens": context_window,
                    "supports_vision": False,
                    "supports_prompt_caching": False,
                    "supports_computer_use": False,
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0
                },
                "litellm_params": {
                    "model": model_id # This is the model name that LiteLLM uses
                }
            })
        
        return {"data": model_data}
        
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model info: {e}")

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

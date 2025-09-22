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
    max_tokens: int = 150
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
        print(f"[DEBUG] Ollama /api/tags response: {json.dumps(ollama_data, indent=2)}")
        if "models" in ollama_data:
            ollama_models = ollama_data["models"]
        else:
            ollama_models = ollama_data.get("models", [])
        print(f"[DEBUG] Found {len(ollama_models)} models")
        openai_models = []
        for model in ollama_models:
            model_id = model.get("name", "")
            if not model_id:
                print(f"[DEBUG] Skipping model with no name: {model}")
                continue
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
        response_data = {"object": "list", "data": openai_models}
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

def _parse_tool_choice(tool_choice: Any) -> Tuple[bool, Optional[str]]:
    """
    Returns (tool_required, required_function_name or None)
    Supports: "required" or {"type":"function","function":{"name":"..."}}
    """
    if tool_choice == "required":
        return True, None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            name = tool_choice.get("function", {}).get("name")
            return True, name
    return False, None

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """vLLM inference with Ollama model management"""
    try:
        # Determine tool requirement
        tool_required, required_function_name = _parse_tool_choice(request.tool_choice)

        # Lazy load vLLM engine
        engine = await load_vLLM(request.model)

        # Convert messages to model-specific prompt format
        prompt = build_model_prompt(
            request.messages,
            request.model,
            request.tools,
            force_tools=tool_required,
            required_function_name=required_function_name
        )
        print(f"[DEBUG] Generated prompt: {prompt[:200]}...")

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            repetition_penalty=request.repetition_penalty,
            stop=[
                "</s>", "[/INST]", "<|im_start|>assistant",
                "[/ASSISTANT]",  # helps DevStral-like tags
                "[thinking]", "[/thinking]", "[plan]", "[/plan]",
                "[scratchpad]", "[/scratchpad]", "[internal]", "[/internal]",
                "Roo has a question"
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

                    # Detect tool calls
                    match = re.search(r"\[TOOL_CALLS\]\[(.*?)\]", buffer, re.DOTALL)
                    if match and not tool_calls_sent:
                        before = buffer[:match.start()]
                        json_str = match.group(1)
                        after = buffer[match.end():]

                        # If tools are required, DO NOT emit pre-tool content
                        if not tool_required:
                            clean_before = sanitize_visible_text(before, for_stream=True)
                            if clean_before:
                                yield sse_delta_content(clean_before)

                        # Parse and emit tool_calls
                        try:
                            parsed_tool_calls = json.loads(json_str)
                            # If a specific function is required, optionally enforce (filter or just send all)
                            if required_function_name:
                                # Optional enforcement: ensure at least one matches
                                if not any(tc.get("name") == required_function_name for tc in parsed_tool_calls):
                                    print(f"[WARN] Required function '{required_function_name}' not found in TOOL_CALLS; forwarding anyway.")
                            tool_calls = []
                            for tc in parsed_tool_calls:
                                tool_calls.append({
                                    "id": random_uuid(),
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc["arguments"])
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
                        finish_reason = result.outputs[0].finish_reason if result.outputs[0].finish_reason else "stop"
                        yield "data: " + json.dumps({
                            'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                            'object': 'chat.completion.chunk',
                            'created': created_time,
                            'model': request.model,
                            'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]
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

            tool_calls_pattern = r"\[TOOL_CALLS\]\[(.*?)\]"
            match = re.search(tool_calls_pattern, generated_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    parsed_tool_calls = json.loads(json_str)
                    if required_function_name and not any(tc.get("name") == required_function_name for tc in parsed_tool_calls):
                        print(f"[WARN] Required function '{required_function_name}' not found in TOOL_CALLS (non-stream).")
                    for tc in parsed_tool_calls:
                        tool_calls.append({
                            "id": random_uuid(),
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}
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
                    "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                    "finish_reason": final_output.outputs[0].finish_reason if final_output.outputs[0].finish_reason else "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
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
) -> str:
    """
    Builds a prompt. If the model looks like DevStral, use DevStral tag format:
      [SYSTEM_PROMPT]...[/SYSTEM_PROMPT]
      [AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS]
      [USER]...[/USER]
      [ASSISTANT]...[/ASSISTANT]
    Otherwise, fall back to the Llama-chat style you had.
    """
    def only_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(part.get("text","") for part in content if part.get("type") == "text")
        return ""

    model_lc = (model_id or "").lower()
    devstral_like = any(x in model_lc for x in ["devstral", "huihui_ai", "abliterated"])

    if devstral_like:
        sys_lines = ["You are a fast, helpful assistant."]
        if force_tools:
            sys_lines += [
                "TOOLS ARE REQUIRED for this reply.",
                "Output ONLY one tag exactly once: [TOOL_CALLS][<valid JSON array>].",
                "Do not output any other text before or after the tag.",
            ]
            if required_function_name:
                sys_lines.append(
                    f"You MUST call the function named \"{required_function_name}\" in the TOOL_CALLS array."
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

        return system_block + tools_block + "\n".join(conv) + "\n[ASSISTANT]"
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
        return "<s>" + " ".join(prompt_parts) + "</s>"

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

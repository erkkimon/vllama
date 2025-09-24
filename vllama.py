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

app = FastAPI(title="vllama", version="0.1.2", lifespan=lifespan)

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

# FIXED the env var typo; previously looked up "RE REPORTED_CONTEXT_WINDOW"
REPORTED_CONTEXT_WINDOW = int(os.environ.get("REPORTED_CONTEXT_WINDOW", 65536))
DEVSTRAL_CONTEXT_WINDOW = int(os.environ.get("DEVSTRAL_CONTEXT_WINDOW", 65536))
current_max_model_len = None

# ---------- Output sanitization to strip meta/thoughts ----------
# Keep XML tool payloads intact; only remove hidden-thought tags and Roo banners.
THINK_TAGS = r"(?:thinking|plan|scratchpad|analysis|internal|note|notes|reflection|deliberate)"
THINKY_PATTERNS = [
    rf"(?is)\[\s*{THINK_TAGS}\s*\](.*?)\[/\s*{THINK_TAGS}\s*\]",
    rf"(?is)\[\s*{THINK_TAGS}\s*\][\s\S]*?(?=(\n\[[A-Za-z_\/]+)|\Z)",
    rf"(?is)\[\s*{THINK_TAGS}\s*\][^\S\r\n]*[\s\S]*?(?=\n\s*\n|\Z)",
    rf"(?im)^\s*\[\s*{THINK_TAGS}\s*\]\s*$",
    r"(?im)^\s*Roo is having trouble.*?(?=\n\s*\n|\Z)",
    r"(?is)\bTask\s+Completed\b.*$",
]

def sanitize_visible_text(txt: str, *, for_stream: bool = False) -> str:
    """
    Remove hidden-thought/meta blocks and tidy whitespace.
    IMPORTANT: when for_stream=True, DO NOT trim leading/trailing whitespace,
    otherwise we destroy spaces that span chunk boundaries.
    """
    if not txt:
        return txt
    # Do NOT strip XML tool calls; they're not bracket-tagged.
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
            
            max_model_len = int(os.environ.get("MAX_MODEL_LEN", 53840))
            if max_model_len == 0:
                max_model_len = None  # auto-detect
            
            engine_args = AsyncEngineArgs(
                model=model_path,
                tokenizer=model_path,
                gpu_memory_utilization=0.95,
                enforce_eager=True,
                disable_log_stats=True,
                max_model_len=max_model_len,
            )
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            if hasattr(engine_args, 'max_model_len') and engine_args.max_model_len:
                current_max_model_len = engine_args.max_model_len
            else:
                current_max_model_len = max_model_len or DEFAULT_CONTEXT_WINDOW
            
            print(f"[{time.strftime('%H:%M:%S')}] vLLM loaded successfully with max_model_len: {current_max_model_len}")
        last_activity = time.time()
        return engine

def get_model_specific_context_window(model_id: str) -> int:
    model_lower = model_id.lower()
    if any(k in model_lower for k in ["devstral", "huihui_ai/devstral", "devstral-abliterated"]):
        print(f"[DEBUG] Detected Devstral model: {model_id}, using context window: {DEVSTRAL_CONTEXT_WINDOW}")
        return DEVSTRAL_CONTEXT_WINDOW
    return REPORTED_CONTEXT_WINDOW

def get_context_window_for_model(model_id: str) -> int:
    global current_max_model_len
    if engine and current_max_model_len:
        return current_max_model_len
    return get_model_specific_context_window(model_id)

def get_reported_context_window_for_model(model_id: str) -> int:
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
    Returns (tool_required, required_function_name) in OpenAI-compatible way.
    """
    if tool_choice == "none":
        return False, None
    if tool_choice == "required":
        return True, None
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return True, tool_choice.get("function", {}).get("name")
    return False, None  # "auto" or None

def _should_force_tools_from_conversation(messages: List[Dict[str, Any]]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if last.get("role") != "user":
        return False
    txt = ""
    c = last.get("content")
    if isinstance(c, str):
        txt = c
    elif isinstance(c, list):
        txt = "\n".join(
            part.get("text", "")
            for part in c
            if isinstance(part, dict) and part.get("type") == "text"
        )
    triggers = [
        "You did not use a tool",
        "Please retry with a tool use",
        "Tool uses are formatted",
        "<attempt_completion>",
        "<update_todo_list>",
        "<read_file>", "<apply_diff>", "<write_to_file>",
        "<insert_content>", "<search_and_replace>",
        "<list_files>", "<search_files>", "<list_code_definition_names>",
        "<ask_followup_question>", "<switch_mode>", "<new_task>",
    ]
    txt_low = txt.lower()
    return any(t.lower() in txt_low for t in triggers)

def _detect_xml_tool_mode(messages: List[Dict[str, Any]]) -> bool:
    for m in messages:
        if m.get("role") != "system":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            text = "\n".join(part.get("text","") for part in content if isinstance(part, dict) and part.get("type")=="text")
        else:
            text = str(content)
        if ("XML-style tags" in text) or ("<write_to_file>" in text) or ("</write_to_file>" in text):
            return True
    return False

# ---------------- XML tool validation helpers (new) ----------------

# Minimal requirements per Roo's tool schema
REQUIRED_XML_PARAMS: Dict[str, List[str]] = {
    "write_to_file": ["path", "content", "line_count"],
    "attempt_completion": ["result"],
    # ask_followup_question requires <question> and at least one <suggest> inside <follow_up>
    "ask_followup_question": ["question"],  # special-case follow_up/suggest separately
}

_XML_OPEN_CLOSE_RE = re.compile(r"<(?P<name>[a-zA-Z_][\w\-]*)\b[^>]*>(?P<body>[\s\S]*?)</\1>", re.DOTALL)

def extract_first_xml_tool_call(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Returns (tool_name, body, raw_xml) for the first top-level XML tag.
    Very lightweight regex-based extractor; assumes no nested same-named tags.
    """
    if not text:
        return None
    m = _XML_OPEN_CLOSE_RE.search(text)
    if not m:
        return None
    name = m.group("name")
    body = m.group("body") or ""
    raw = m.group(0)
    return name, body, raw

def find_tag_value(body: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}\b[^>]*>([\s\S]*?)</{tag}>", body, re.DOTALL)
    return m.group(1) if m else None

def has_any_suggest(body: str) -> bool:
    # Accept either wrapped in <follow_up>…<suggest>… or direct <suggest> tags
    if re.search(r"<follow_up\b[^>]*>[\s\S]*?<suggest\b[^>]*>[\s\S]*?</suggest>[\s\S]*?</follow_up>", body, re.DOTALL):
        return True
    if re.search(r"<suggest\b[^>]*>[\s\S]*?</suggest>", body, re.DOTALL):
        return True
    return False

def validate_xml_tool_call(raw_xml: str) -> Tuple[bool, Optional[str], List[str]]:
    """
    Validate the first XML tool call found in raw_xml.
    Returns (is_valid, tool_name, missing_params)
    """
    ext = extract_first_xml_tool_call(raw_xml)
    if not ext:
        return False, None, ["<tool>"]
    tool, body, _ = ext
    tool_lc = tool.strip().lower()
    missing: List[str] = []

    if tool_lc in REQUIRED_XML_PARAMS:
        reqs = REQUIRED_XML_PARAMS[tool_lc]
        for r in reqs:
            if find_tag_value(body, r) is None:
                missing.append(r)
        # special case: suggestions for ask_followup_question
        if tool_lc == "ask_followup_question" and not has_any_suggest(body):
            missing.append("suggest")
    else:
        # Unknown tool: let it pass (or treat as missing schema)
        return True, tool, []

    is_valid = len(missing) == 0
    return is_valid, tool_lc, missing

def build_fallback_ask_for_missing(tool: Optional[str], missing: List[str]) -> str:
    """
    Build a valid ask_followup_question XML asking for the minimal missing field(s).
    """
    if not tool:
        ask = "I need a complete XML tool invocation with required parameters."
    else:
        ask = f"The {tool} tool call is missing required parameter(s): {', '.join(missing)}."

    # Provide concrete suggestions that match Roo's schema
    suggestions: List[str] = []
    if tool == "write_to_file":
        if "path" in missing:
            suggestions.append("<suggest>README.md</suggest>")
            suggestions.append("<suggest>docs/README.md</suggest>")
        if "content" in missing:
            suggestions.append("<suggest>Use a basic README template with title, usage, and license</suggest>")
        if "line_count" in missing:
            suggestions.append("<suggest>27</suggest>")
    elif tool == "attempt_completion" and "result" in missing:
        suggestions.append("<suggest>I've completed the requested changes.</suggest>")
        suggestions.append("<suggest>The README.md file has been created and finalized.</suggest>")
    else:
        suggestions.append("<suggest>Proceed with the minimal valid example</suggest>")

    sugg_block = "".join(suggestions) if suggestions else "<suggest>Proceed</suggest>"

    return (
        "<ask_followup_question>"
        f"<question>{ask}</question>"
        "<follow_up>"
        f"{sugg_block}"
        "</follow_up>"
        "</ask_followup_question>"
    )

# ---------------- Chat API ----------------

@app.post("/chat/completions")
async def chat_completions_root(request: ChatCompletionRequest):
    return await chat_completions(request)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        tool_required, required_function_name = _parse_tool_choice(request.tool_choice, bool(request.tools))
        # Pydantic v2-safe
        try:
            print(f"[DEBUG] Incoming ChatCompletionRequest: {request.model_dump_json()}")
        except Exception:
            print("[DEBUG] Incoming ChatCompletionRequest (dump failed)")

        # Try to extract tools from the system message if not provided
        if not request.tools:
            for message in request.messages:
                if message["role"] == "system" and isinstance(message["content"], str):
                    tool_pattern = re.compile(r"## (\w+)\nDescription: ([^\n]+)\nParameters:\n(.+?)(?=\n## |\Z)", re.DOTALL)
                    matches = tool_pattern.findall(message["content"])
                    extracted_tools = []
                    for match in matches:
                        tool_name = match[0]
                        tool_description = match[1]
                        tool_parameters_str = match[2]
                        parameters = {"type": "object", "properties": {}}
                        required_props: List[str] = []
                        for line in tool_parameters_str.splitlines():
                            if not line.strip().startswith("- "):
                                continue
                            try:
                                after_dash = line.strip()[2:]
                                pname, rest = after_dash.split(":", 1)
                                pname = pname.strip()
                                if "(required)" in rest:
                                    required_props.append(pname)
                                parameters["properties"][pname] = {"type": "string"}
                            except Exception:
                                continue
                        if required_props:
                            parameters["required"] = required_props
                        extracted_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": tool_description,
                                "parameters": parameters
                            }
                        })
                    if extracted_tools:
                        request.tools = extracted_tools
                        if not request.tool_choice:
                            request.tool_choice = "auto"
                        break

        force_tools_from_error = _should_force_tools_from_conversation(request.messages)
        xml_tool_mode = _detect_xml_tool_mode(request.messages)

        if not tool_required and force_tools_from_error:
            print("[DEBUG] Forcing tool use due to Roo tool-use error in last user message.")
            tool_required = True

        print(f"[DEBUG] tools_present={bool(request.tools)} tool_choice={request.tool_choice!r} "
              f"tool_required={tool_required} required_function={required_function_name} xml_tool_mode={xml_tool_mode}")

        engine_instance = await load_vLLM(request.model)

        prompt, seeded_tool_prefix = build_model_prompt(
            request.messages,
            request.model,
            request.tools,
            force_tools=tool_required,
            required_function_name=required_function_name,
            xml_tool_mode=xml_tool_mode,
        )
        print(f"[DEBUG] Generated prompt: {prompt[:200]}...")
        if tool_required and seeded_tool_prefix:
            print("[DEBUG] Prompt seeded with [ASSISTANT][TOOL_CALLS] prefix for tool-first enforcement.")

        # Extra stop sequences for XML mode; helps stop right after closing tags
        xml_stops = [
            "</write_to_file>",
            "</attempt_completion>",
            "</ask_followup_question>",
            "</insert_content>",
            "</apply_diff>",
            "</search_and_replace>",
        ] if xml_tool_mode else []

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or MAX_TOKENS_DEFAULT,
            repetition_penalty=request.repetition_penalty,
            stop=[
                "</s>", "[/INST]", "<|im_start|>assistant",
                "[/ASSISTANT]",
                "[thinking]", "[/thinking]", "[plan]", "[/plan]",
                "[scratchpad]", "[/scratchpad]", "[internal]", "[/internal]",
                "Roo has a question",
                "[USER][ERROR]",
                "Roo is having trouble..."
            ] + xml_stops
        )

        results_generator = engine_instance.generate(prompt, sampling_params, request_id=random_uuid())

        if request.stream:
            async def stream_results():
                previous_text = ""
                request_id_ = random_uuid()
                created_time = int(time.time())

                def sse_delta_content(text: str) -> str:
                    return "data: " + json.dumps({
                        'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                        'object':'chat.completion.chunk',
                        'created': created_time,
                        'model': request.model,
                        'choices':[{'index':0,'delta': {'content': text},'finish_reason': None}]
                    }) + "\n\n"

                def sse_delta_role() -> str:
                    return "data: " + json.dumps({
                        'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                        'object': 'chat.completion.chunk',
                        'created': created_time,
                        'model': request.model,
                        'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]
                    }) + "\n\n"

                # Tool-call streaming helpers (non-XML mode only)
                def sse_delta_tool_name(i: int, name: str) -> str:
                    return "data: " + json.dumps({
                        'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                        'object': 'chat.completion.chunk',
                        'created': created_time,
                        'model': request.model,
                        'choices': [{
                            'index': 0,
                            'delta': {'tool_calls': [{
                                'index': i,
                                'id': random_uuid(),
                                'type': 'function',
                                'function': {'name': name}
                            }]},
                            'finish_reason': None
                        }]
                    }) + "\n\n"

                def sse_delta_tool_args(i: int, args_str: str) -> str:
                    return "data: " + json.dumps({
                        'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                        'object': 'chat.completion.chunk',
                        'created': created_time,
                        'model': request.model,
                        'choices': [{
                            'index': 0,
                            'delta': {'tool_calls': [{
                                'index': i,
                                'function': {'arguments': args_str}
                            }]},
                            'finish_reason': None
                        }]
                    }) + "\n\n"

                def sse_finish_tool_calls() -> str:
                    return "data: " + json.dumps({
                        'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                        'object': 'chat.completion.chunk',
                        'created': created_time,
                        'model': request.model,
                        'choices': [{
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'tool_calls'
                        }]
                    }) + "\n\n"

                # Initial assistant role chunk
                yield sse_delta_role()

                buffer = ""
                tool_calls_sent = False

                async for result in results_generator:
                    new_text = result.outputs[0].text
                    delta = new_text[len(previous_text):]
                    previous_text = new_text
                    buffer += delta

                    # XML tool mode: buffer until a complete top-level tool tag appears, then validate
                    if xml_tool_mode and not tool_calls_sent:
                        ext = extract_first_xml_tool_call(buffer)
                        if ext:
                            tool_name, body, raw_xml = ext
                            valid, tname, missing = validate_xml_tool_call(raw_xml)
                            to_emit = raw_xml if valid else build_fallback_ask_for_missing(tname, missing)
                            clean = to_emit  # do NOT sanitize XML tool payload
                            if clean:
                                yield sse_delta_content(clean)
                            # Finish immediately after emitting a single tool invocation
                            prompt_tokens, completion_tokens = _safe_token_usage_counts(engine_instance, prompt, previous_text)
                            total_tokens = prompt_tokens + completion_tokens
                            actual_context_window = current_max_model_len or DEFAULT_CONTEXT_WINDOW
                            reported_context_window = get_reported_context_window_for_model(request.model)
                            context_usage_percent = (total_tokens / reported_context_window) * 100 if reported_context_window > 0 else 0
                            yield "data: " + json.dumps({
                                'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                                'object': 'chat.completion.chunk',
                                'created': created_time,
                                'model': request.model,
                                'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                                'usage': {
                                    'prompt_tokens': prompt_tokens,
                                    'completion_tokens': completion_tokens,
                                    'total_tokens': total_tokens,
                                    'context_window': reported_context_window,
                                    'context_usage_percent': round(context_usage_percent, 1)
                                }
                            }) + "\n\n"
                            yield "data: [DONE]\n\n"
                            return

                    # Non-XML tool streaming path
                    if not xml_tool_mode:
                        match_tagged = re.search(r"\[TOOL_CALLS\]\[(.*?)\]", buffer, re.DOTALL)
                        match_seeded = None
                        if (not match_tagged) and (tool_required and seeded_tool_prefix):
                            m = re.search(r"^\s*\[(.*?)\]", buffer, re.DOTALL)
                            if m:
                                match_seeded = m

                        if (match_tagged or match_seeded) and not tool_calls_sent:
                            if match_tagged:
                                before = buffer[:match_tagged.start()]
                                json_str = match_tagged.group(1)
                                after = buffer[match_tagged.end():]
                            else:
                                before = buffer[:match_seeded.start()]
                                json_str = match_seeded.group(1)
                                after = buffer[match_seeded.end():]

                            if not tool_required:
                                clean_before = sanitize_visible_text(before, for_stream=True)
                                if clean_before:
                                    yield sse_delta_content(clean_before)

                            try:
                                parsed_tool_calls = json.loads(f"[{json_str}]")
                            except json.JSONDecodeError:
                                fallback = sanitize_visible_text("[TOOL_CALLS][" + json_str + "]", for_stream=True)
                                if fallback and not tool_required:
                                    yield sse_delta_content(fallback)
                                buffer = after
                                continue

                            for i, tc in enumerate(parsed_tool_calls):
                                name = tc.get("name", "")
                                args = json.dumps(tc.get("arguments", {}))
                                yield sse_delta_tool_name(i, name)
                                yield sse_delta_tool_args(i, args)

                            tool_calls_sent = True
                            buffer = ""
                            yield sse_finish_tool_calls()
                            yield "data: [DONE]\n\n"
                            return

                    # Prose streaming (sanitized) only if not XML mode
                    if not xml_tool_mode:
                        clean = sanitize_visible_text(buffer, for_stream=True)
                        if clean and not tool_calls_sent:
                            yield sse_delta_content(clean)
                            buffer = ""

                    if result.finished and not tool_calls_sent:
                        # Finalize (no tool emitted)
                        prompt_tokens, completion_tokens = _safe_token_usage_counts(engine_instance, prompt, previous_text)
                        total_tokens = prompt_tokens + completion_tokens
                        actual_context_window = current_max_model_len or DEFAULT_CONTEXT_WINDOW
                        reported_context_window = get_reported_context_window_for_model(request.model)
                        context_usage_percent = (total_tokens / reported_context_window) * 100 if reported_context_window > 0 else 0
                        print(f"[DEBUG] Streaming context usage: {total_tokens}/{reported_context_window} tokens ({context_usage_percent:.1f}%) [actual: {actual_context_window}]")

                        # In XML tool mode and tools required, if nothing valid was produced, emit a fallback ask
                        if xml_tool_mode and tool_required:
                            fallback_xml = build_fallback_ask_for_missing(None, ["<tool>"])
                            yield sse_delta_content(fallback_xml)

                        yield "data: " + json.dumps({
                            'id': f'chatcmpl-{created_time}-{request_id_[:8]}',
                            'object': 'chat.completion.chunk',
                            'created': created_time,
                            'model': request.model,
                            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                            'usage': {
                                'prompt_tokens': prompt_tokens,
                                'completion_tokens': completion_tokens,
                                'total_tokens': total_tokens,
                                'context_window': reported_context_window,
                                'context_usage_percent': round(context_usage_percent, 1)
                            }
                        }) + "\n\n"
                        yield "data: [DONE]\n\n"

            return StreamingResponse(stream_results(), media_type="text/event-stream")
        else:
            final_output = None
            async for result in results_generator:
                final_output = result

            if not final_output or not final_output.outputs:
                raise HTTPException(status_code=500, detail="Generation failed - no output")

            generated_text = (final_output.outputs[0].text or "").strip()
            tool_calls: List[Dict[str, Any]] = []
            content = generated_text

            if xml_tool_mode:
                # Extract first XML tool, validate, and either keep it or replace with a valid ask_followup_question
                ext = extract_first_xml_tool_call(generated_text)
                if ext:
                    _, _, raw_xml = ext
                    valid, tname, missing = validate_xml_tool_call(raw_xml)
                    content = raw_xml if valid else build_fallback_ask_for_missing(tname, missing)
                else:
                    if tool_required:
                        content = build_fallback_ask_for_missing(None, ["<tool>"])
                    else:
                        # If tools not required, keep sanitized prose (avoid empty)
                        content = sanitize_visible_text(generated_text, for_stream=False) or " "
            else:
                # Non-XML tool_calls JSON hook
                tool_calls_pattern = r"\[TOOL_CALLS\]\[(.*?)\]"
                match = re.search(tool_calls_pattern, generated_text, re.DOTALL)

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
                            content = ""
                        except json.JSONDecodeError:
                            pass

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
                        content = re.sub(tool_calls_pattern, "", generated_text, 1, re.DOTALL).strip()
                    except json.JSONDecodeError:
                        print(f"[ERROR] Failed to parse tool calls JSON: {json_str}")

                # sanitize prose if we're not returning tool_calls as content
                content = sanitize_visible_text(content, for_stream=False)

            # Ensure we never return an empty assistant message
            if not tool_calls and (content is None or content == ""):
                content = " "

            # Enforce tool requirement:
            if tool_required and not xml_tool_mode and not tool_calls:
                raise HTTPException(
                    status_code=400,
                    detail="Tool use required by client (tool_choice) but no tool calls were produced."
                )

            # Token usage
            prompt_tokens, completion_tokens = _safe_token_usage_counts(engine_instance, prompt, generated_text)
            total_tokens = prompt_tokens + completion_tokens
            actual_context_window = current_max_model_len or DEFAULT_CONTEXT_WINDOW
            reported_context_window = get_reported_context_window_for_model(request.model)
            context_usage_percent = (total_tokens / reported_context_window) * 100 if reported_context_window > 0 else 0
            print(f"[DEBUG] Context usage: {total_tokens}/{reported_context_window} tokens ({context_usage_percent:.1f}%) [actual: {actual_context_window}]")

            finish_reason = final_output.outputs[0].finish_reason if final_output.outputs[0].finish_reason else "stop"
            if tool_calls and not xml_tool_mode:
                finish_reason = "tool_calls"
                # If tool_calls are present, many OpenAI clients ignore content by spec; keep it None.
                content = None

            return {
                "id": f"chatcmpl-{int(time.time())}-{random_uuid()[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content if content is not None else " ", "tool_calls": tool_calls or None},
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens if not tool_calls else 0,
                    "total_tokens": total_tokens if not tool_calls else prompt_tokens,
                    "context_window": reported_context_window,
                    "context_usage_percent": round(context_usage_percent, 1)
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

def _safe_token_usage_counts(engine_instance, prompt: str, completion_text: str) -> Tuple[int, int]:
    """
    Helper to compute token counts, falling back to whitespace counts if tokenizer access fails.
    """
    prompt_tokens = len(prompt.split())
    completion_tokens = len(completion_text.split())
    try:
        get_tok = getattr(engine_instance, "get_tokenizer", None)
        if get_tok is not None:
            tok = get_tok()
            if asyncio.iscoroutine(tok):
                tok = await_or_none(tok)
            if tok and hasattr(tok, "encode"):
                prompt_tokens = len(tok.encode(prompt))
                completion_tokens = len(tok.encode(completion_text))
                print(f"[DEBUG] Accurate token count - prompt: {prompt_tokens}, completion: {completion_tokens}")
    except Exception as e:
        print(f"[DEBUG] Could not use vLLM tokenizer for token counting: {e}")
    return prompt_tokens, completion_tokens

async def await_or_none(coro):
    try:
        return await coro
    except Exception:
        return None

# ---------- Prompt builder (DevStral-aware, with force-tools mode & XML mode) ----------
def build_model_prompt(
    messages: List[Dict[str, Any]],
    model_id: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    *,
    force_tools: bool = False,
    required_function_name: Optional[str] = None,
    xml_tool_mode: bool = False,
) -> Tuple[str, bool]:
    """
    Builds a prompt. If the model looks like DevStral, use DevStral tag format.
    Returns (prompt, seeded_tool_prefix).
    """
    def only_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(part.get("text","") for part in content if isinstance(part, dict) and part.get("type") == "text")
        return ""

    model_lc = (model_id or "").lower()
    devstral_like = any(x in model_lc for x in ["devstral", "huihui_ai", "abliterated"])

    seeded_tool_prefix = False

    # Mini schema hint to reduce malformed XML
    xml_schema_hint = (
        "When emitting XML tools, include required parameters:\n"
        "- write_to_file: <path>, <content>, <line_count>\n"
        "- attempt_completion: <result>\n"
        "- ask_followup_question: <question> and at least one <suggest> (inside <follow_up>)\n"
        "Emit exactly one tool invocation and nothing else."
    )

    if devstral_like:
        sys_lines = ["You are a fast, helpful assistant."]
        if force_tools:
            if xml_tool_mode:
                sys_lines += [
                    "TOOLS ARE REQUIRED for this reply.",
                    "Output EXACTLY ONE tool invocation using the XML tag schema required by the user (e.g., <write_to_file>...</write_to_file>).",
                    "Do not output any other text before or after the XML.",
                    xml_schema_hint,
                    # Example to anchor the format (not content).
                    "Example format (do NOT reuse these values):",
                    "<write_to_file><path>README.md</path><content>...</content><line_count>42</line_count></write_to_file>",
                ]
                if required_function_name:
                    sys_lines.append(f"When choosing which tool to invoke, prefer the function named '{required_function_name}'.")
            else:
                sys_lines += [
                    "TOOLS ARE REQUIRED for this reply.",
                    "Output ONLY a single tool call JSON array.",
                    "Do not output any other text."
                ]
                if required_function_name:
                    sys_lines.append(f"You MUST call the function named '{required_function_name}'.")
        else:
            if xml_tool_mode:
                sys_lines += [
                    "If you use a tool, output ONLY the XML tool invocation with the required tags.",
                    "Otherwise output ONLY the final answer text.",
                    xml_schema_hint,
                ]
            else:
                sys_lines += [
                    "Do NOT output analysis, plans, or hidden thoughts.",
                    "If you use a tool, output ONLY [TOOL_CALLS][…]. Otherwise output ONLY the final answer text.",
                ]

        sys_msg = next((m for m in messages if m.get("role") == "system"), None)
        if sys_msg:
            sys_lines.append(only_text(sys_msg.get("content")))
        system_block = f"[SYSTEM_PROMPT]{'\n'.join(sys_lines)}[/SYSTEM_PROMPT]\n"

        tools_block = f"[AVAILABLE_TOOLS]{json.dumps(tools)}[/AVAILABLE_TOOLS]\n" if tools else ""

        conv = []
        for m in messages:
            role = m.get("role")
            content = only_text(m.get("content"))
            if not content:
                continue
            if role == "system":
                continue
            elif role == "user":
                conv.append(f"[USER]{content}[/USER]")
            elif role == "assistant":
                conv.append(f"[ASSISTANT]{content}[/ASSISTANT]")
            else:
                conv.append(f"[USER]{content}[/USER]")

        if force_tools:
            if xml_tool_mode:
                # Let the model emit XML directly
                prompt = system_block + tools_block + "\n".join(conv) + "\n[ASSISTANT]"
                seeded_tool_prefix = False
            else:
                prompt = system_block + tools_block + "\n".join(conv) + "\n[ASSISTANT][TOOL_CALLS]"
                seeded_tool_prefix = True
        else:
            prompt = system_block + tools_block + "\n".join(conv) + "\n[ASSISTANT]"

        return prompt, seeded_tool_prefix
    else:
        # Fallback: Llama chat style with optional force_tools hint baked into system
        prompt_parts = []
        if xml_tool_mode:
            system_instruction = (
                "You are a helpful assistant. The user expects literal XML tool invocations.\n"
                "If tools are required, output EXACTLY ONE tool invocation using the XML tag schema and nothing else.\n"
                f"{xml_schema_hint}\n"
                "Example: <write_to_file><path>README.md</path><content>...</content><line_count>42</line_count></write_to_file>"
            )
        else:
            system_instruction = "You are a helpful assistant. You can use tools to assist the user."

        if force_tools and not xml_tool_mode:
            system_instruction += "\nTOOLS ARE REQUIRED. Do not output any text except a valid tool call representation."
        elif force_tools and xml_tool_mode:
            system_instruction += "\nTOOLS ARE REQUIRED. Output only the XML tool invocation; do not include extra prose."

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
        if force_tools and not xml_tool_mode:
            return "<s>" + " ".join(prompt_parts) + "</s>" + "\n[TOOL_CALLS]", True
        return "<s>" + " ".join(prompt_parts) + "</s>", False

@app.get("/health")
async def health_check():
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
                "model_name": model_id,
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
                    "model": model_id
                }
            })
        
        return {"data": model_data}
        
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model info: {e}")

@app.get("/v1")
async def openai_root():
    return {"message": "vllama OpenAI-compatible API", "port": 11435}

@app.post("/unload")
async def manual_unload():
    global engine
    async with engine_lock:
        if engine:
            print(f"[{time.strftime('%H:%M:%S')}] Manual unload triggered via /unload")
            del engine
            engine = None
            torch.cuda.empty_cache()
            return {"status": "unloaded", "message": "vLLM engine unloaded from VRAM"}
        return {"status": "already unloaded", "message": "No engine to unload"}

# ---------------- Main ----------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="vllama - vLLM + Ollama hybrid server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=11435, help="Port to listen on")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level (e.g., debug, info, warning, error)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)

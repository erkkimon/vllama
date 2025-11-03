import asyncio
import httpx
import subprocess
import time
import os
import signal
import threading
import logging
import glob
import json
import dataclasses
import re
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

# --- Logging Configuration ---
PROD_DIR = "/opt/vllama"

# Check if running in production
if os.path.dirname(os.path.abspath(__file__)) == PROD_DIR:
    LOG_DIR = os.path.join(PROD_DIR, "logs")
    MODELS_DIR = os.path.join(PROD_DIR, "models")
else:
    LOG_DIR = "logs"
    MODELS_DIR = "models"

LOG_FILE = f"{LOG_DIR}/vllama.log"

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# --- Configuration ---
@dataclasses.dataclass
class OllamaModel:
    id: str
    name: str
    gguf_path: str
    architecture: str
    max_model_len: int
    quantization: str
    tokenizer_type: str
    hf_tokenizer_path_or_name: str # Hugging Face tokenizer ID or path to generated tokenizer.json

ollama_discovered_models: List[OllamaModel] = []

def discover_ollama_gguf_models():
    logging.info("Discovering Ollama GGUF models...")
    try:
        # 1. Get list of Ollama models
        list_result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True, timeout=30
        )
        lines = list_result.stdout.strip().split('\n')
        if len(lines) <= 1: # Header only or empty
            logging.info("No Ollama models found.")
            return

        for line in lines[1:]: # Skip header
            parts = line.split()
            if not parts:
                continue
            model_id = parts[0] # e.g., huihui_ai/devstral-abliterated:latest

            # 2. For each model, run ollama show --modelfile
            show_result = subprocess.run(
                ["ollama", "show", model_id, "--modelfile"],
                capture_output=True, text=True, check=True, timeout=30
            )
            modelfile_content = show_result.stdout

            # 3. Parse the output to extract GGUF path and metadata
            gguf_path = None
            architecture = "unknown"
            max_model_len = 0
            quantization = "unknown"
            tokenizer_type = "unknown"
            hf_tokenizer_path_or_name = "auto" # Default to auto-detect by vLLM

            # Extract GGUF path from FROM line
            from_match = re.search(r"FROM\s+(/var/lib/ollama/models/blobs/sha256-[a-f0-9]+)", modelfile_content)
            if from_match:
                gguf_path = from_match.group(1)

            # Extract metadata from ollama show (without --modelfile)
            # This is a bit redundant as we already ran ollama show, but it's easier to parse the structured output
            # from the regular 'ollama show' command.
            show_info_result = subprocess.run(
                ["ollama", "show", model_id],
                capture_output=True, text=True, check=True, timeout=30
            )
            info_lines = show_info_result.stdout.split('\n')
            for info_line in info_lines:
                if "architecture" in info_line:
                    architecture = info_line.split("architecture", 1)[1].strip().split()[0]
                elif "context length" in info_line:
                    try:
                        max_model_len = int(info_line.split("context length", 1)[1].strip().split()[0])
                    except ValueError:
                        logging.warning("Could not parse context length for model %s: %s", model_id, info_line)
                elif "quantization" in info_line:
                    quantization = info_line.split("quantization", 1)[1].strip().split()[0]
            
            # Infer tokenizer type and HF path based on architecture
            if "devstral" in model_id.lower(): # Explicitly handle Devstral models
                architecture = "mistral"
                tokenizer_type = "mistral"
                hf_tokenizer_path_or_name = "mistralai/Devstral-Small-2507"
            elif architecture.lower() == "llama":
                tokenizer_type = "llama"
                # Use a well-known Llama tokenizer from Hugging Face
                hf_tokenizer_path_or_name = "hf-internal-testing/llama-tokenizer" 
            elif architecture.lower() == "mistral":
                tokenizer_type = "mistral"
                # Use a well-known Mistral tokenizer from Hugging Face
                hf_tokenizer_path_or_name = "mistralai/Mistral-7B-v0.1" 
            elif architecture.lower() == "qwen": # Added for qwen3:14b
                tokenizer_type = "qwen"
                hf_tokenizer_path_or_name = "Qwen/Qwen-1_8B" # Example Qwen tokenizer
            else:
                tokenizer_type = "auto" # Fallback to auto-detection by vLLM
                hf_tokenizer_path_or_name = "auto" # Fallback to auto-detection by vLLM
                logging.warning("Unknown architecture '%s' for model %s. Using 'auto' tokenizer.", architecture, model_id)

            # Use the minimum of reported and calculated max_model_len
            final_max_model_len = max_model_len
            calculated_len = calculate_max_model_len(gguf_path)
            if calculated_len > 0 and calculated_len < final_max_model_len:
                final_max_model_len = calculated_len

            if gguf_path and os.path.exists(gguf_path):
                ollama_discovered_models.append(OllamaModel(
                    id=model_id,
                    name=model_id,
                    gguf_path=gguf_path,
                    architecture=architecture,
                    max_model_len=final_max_model_len,
                    quantization=quantization,
                    tokenizer_type=tokenizer_type,
                    hf_tokenizer_path_or_name=hf_tokenizer_path_or_name
                ))
                logging.info("Discovered Ollama model: %s at %s with max_model_len %d", model_id, gguf_path, final_max_model_len)
            else:
                logging.warning("Could not find GGUF path for Ollama model: %s", model_id)

    except subprocess.CalledProcessError as e:
        logging.error("Failed to discover Ollama models: %s", e.stderr)
    except Exception as e:
        logging.error("An unexpected error occurred during Ollama model discovery: %s", e)

VLLM_HOST = "0.0.0.0"
VLLM_PORT = 11436
PROXY_HOST = "0.0.0.0"
PROXY_PORT = 11435
IDLE_TIMEOUT = 300  # 5 minutes

def find_gguf_files():
    """Find GGUF files in both local and system model directories."""
    local_models = glob.glob(f"{MODELS_DIR}/*.gguf")
    system_models_dir = "/opt/vllama/models"
    system_models = []
    if os.path.exists(system_models_dir):
        system_models = glob.glob(f"{system_models_dir}/*.gguf")
    
    ollama_gguf_paths = [model.gguf_path for model in ollama_discovered_models]

    # Combine and remove duplicates
    all_models = list(set(local_models + system_models + ollama_gguf_paths))
    return all_models

def get_gpu_memory():
    """Get total GPU memory in MiB."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return int(result.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("nvidia-smi not found. Could not determine GPU memory.")
        return None

def calculate_max_model_len(model_path: str):
    """Calculate max_model_len based on available GPU memory."""
    gpu_memory = get_gpu_memory()
    if gpu_memory:
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        headspace_mb = 2 * 1024 # 2GB
        available_memory_mb = (gpu_memory * 0.95) - model_size_mb - headspace_mb
        
        # Heuristic: 217.6 KB per token
        if available_memory_mb > 0:
            return int(available_memory_mb * 1024 / 217.6)
        else:
            logging.warning("Not enough GPU memory to run the model with headspace.")
    return 74880 # Default value

def get_vllm_model_command(model_name: str):
    """Construct the vLLM command for a specific model."""
    
    ollama_model_config = None
    for om in ollama_discovered_models:
        if om.name == model_name:
            ollama_model_config = om
            break

    if ollama_model_config:
        model_path_for_vllm = ollama_model_config.gguf_path
        served_model_name = ollama_model_config.name
        max_model_len = ollama_model_config.max_model_len
        tokenizer_path = ollama_model_config.hf_tokenizer_path_or_name # Use HF tokenizer ID
        
        # Map our tokenizer_type to vLLM's supported tokenizer-mode
        if ollama_model_config.tokenizer_type == "llama":
            tokenizer_mode = "mistral" # vLLM often uses 'mistral' mode for Llama tokenizers
            tool_call_parser = "llama3_json" # Use a Llama-specific tool-call-parser
        elif ollama_model_config.tokenizer_type == "mistral":
            tokenizer_mode = "mistral"
            tool_call_parser = "mistral"
        elif ollama_model_config.tokenizer_type == "qwen":
            tokenizer_mode = "auto" # Let vLLM auto-detect for Qwen
            tool_call_parser = "qwen3_coder" # Qwen has a specific parser
        else:
            tokenizer_mode = "auto" # Fallback for unknown types
            tool_call_parser = "openai" # Default fallback for tool-call-parser, more generic than mistral

        logging.info("Serving Ollama model %s (vLLM model path: %s) with max_model_len %d, tokenizer %s, tokenizer_mode %s, tool_call_parser %s", 
                     served_model_name, model_path_for_vllm, max_model_len, tokenizer_path, tokenizer_mode, tool_call_parser)
    else:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.gguf")
        if not os.path.exists(model_path):
            # Try to find model in all available locations
            all_gguf_files = find_gguf_files()
            for f in all_gguf_files:
                # Check if the base name (without .gguf) matches
                if os.path.basename(f).replace(".gguf", "") == model_name:
                    model_path = f
                    break
        
        if not os.path.exists(model_path):
            logging.error("Model %s not found.", model_name)
            return None

        # For manually found GGUF, we still pass the GGUF path directly.
        model_path_for_vllm = model_path
        served_model_name = os.path.basename(model_path).replace(".gguf", "")
        max_model_len = calculate_max_model_len(model_path)
        tokenizer_path = "mistralai/Devstral-Small-2507" # Default for non-Ollama GGUF
        tokenizer_mode = "mistral" # Default for non-Ollama GGUF
        tool_call_parser = "mistral" # Default for non-Ollama GGUF
        logging.info("Serving local GGUF model %s (vLLM model path: %s) with max_model_len %d, tokenizer %s, tokenizer_mode %s, tool_call_parser %s", 
                     served_model_name, model_path_for_vllm, max_model_len, tokenizer_path, tokenizer_mode, tool_call_parser)

    return f"""
python -m vllm.entrypoints.openai.api_server \
    --host {VLLM_HOST} \
    --port {VLLM_PORT} \
    --gpu-memory-utilization 0.95 \
    --disable-log-stats \
    --enforce-eager \
    --model {model_path_for_vllm} \
    --served-model-name {served_model_name} \
    --enable-auto-tool-choice \
    --tool-call-parser {tool_call_parser} \
    --tokenizer {tokenizer_path} \
    --tokenizer-mode {tokenizer_mode} \
    --enforce-eager \
    --max-model-len {max_model_len}
"""
# --- Global State ---
vllm_process = None
last_request_time = None
lock = asyncio.Lock()
server_ready = asyncio.Event()
current_model = None

# --- vLLM Process Management ---
def is_vllm_ready():
    """Check if the vLLM server is ready to accept connections."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((VLLM_HOST, VLLM_PORT))
            return True
        except ConnectionRefusedError:
            return False

async def start_vllm_server(model_name: str):
    """Start the vLLM server process and wait for it to be ready."""
    global vllm_process, last_request_time, current_model
    async with lock:
        if vllm_process is None or vllm_process.poll() is not None or current_model != model_name:
            kill_vllm_server() # Kill existing server if model is different
            server_ready.clear()
            
            command = get_vllm_model_command(model_name)
            if not command:
                logging.error("Could not start vLLM server: No model command.")
                return

            logging.info("Starting vLLM server for model %s...", model_name)
            vllm_log_file = open(f"{LOG_DIR}/vllama.log", "w")
            vllm_err_file = open(f"{LOG_DIR}/vllm.err", "w")
            vllm_process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=vllm_log_file, stderr=vllm_err_file)
            current_model = model_name
            
            # Asynchronously wait for the server to be ready
            while not await asyncio.to_thread(is_vllm_ready):
                if vllm_process.poll() is not None:
                    logging.error("vLLM server process terminated unexpectedly.")
                    return
                logging.info("Waiting for vLLM server to be ready...")
                await asyncio.sleep(2)
            
            logging.info("vLLM server is ready.")
            last_request_time = time.time()
            server_ready.set()

def kill_vllm_server():
    """Kill the vLLM server process."""
    global vllm_process, current_model
    if vllm_process:
        server_ready.clear()
        logging.info("Killing vLLM server...")
        try:
            os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
            vllm_process.wait()
        except ProcessLookupError:
            pass # Process already dead
        vllm_process = None
        current_model = None
        logging.info("vLLM server killed.")

# --- Idle Timeout Checker ---
def idle_check():
    """Periodically check for idle timeout."""
    while True:
        time.sleep(60)  # Check every minute
        if last_request_time and vllm_process and server_ready.is_set():
            if time.time() - last_request_time > IDLE_TIMEOUT:
                kill_vllm_server()

# --- App Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    idle_thread = threading.Thread(target=idle_check, daemon=True)
    idle_thread.start()
    discover_ollama_gguf_models() # Call at startup
    yield
    kill_vllm_server()

app = FastAPI(lifespan=lifespan)

# --- Models Endpoint ---
@app.get("/v1/models")
async def list_models():
    """Return a list of available GGUF models."""
    models = []

    # Add Ollama discovered models
    for ollama_model in ollama_discovered_models:
        max_len_str = f"{ollama_model.max_model_len // 1000}k" if ollama_model.max_model_len > 1000 else str(ollama_model.max_model_len)
        models.append({
            "id": f"{ollama_model.name} ({max_len_str})",
            "object": "model",
            "created": int(time.time()), # Use current time as creation time for discovered models
            "owned_by": "vllama",
            "context_window": ollama_model.max_model_len,
        })

    # Add manually found GGUF files (excluding those already discovered by Ollama)
    gguf_files = find_gguf_files()
    ollama_gguf_paths = {model.gguf_path for model in ollama_discovered_models}

    for f in gguf_files:
        if f in ollama_gguf_paths:
            continue # Skip if already added from Ollama discovery

        model_id = os.path.basename(f).replace(".gguf", "")
        max_len = calculate_max_model_len(f)
        max_len_str = f"{max_len // 1000}k" if max_len > 1000 else str(max_len)
        
        models.append({
            "id": f"{model_id} ({max_len_str})",
            "object": "model",
            "created": int(os.path.getctime(f)),
            "owned_by": "vllama",
            "context_window": max_len,
        })
    return JSONResponse(content={"object": "list", "data": models})

# --- Proxy Endpoint ---
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, full_path: str):
    """The main proxy endpoint."""
    global last_request_time

    body = await request.body()
    new_body = body
    model_name = None
    try:
        json_body = json.loads(body)
        model_name_from_request = json_body.get("model")
        if model_name_from_request:
            model_name = model_name_from_request.split(' (')[0]
            json_body["model"] = model_name
            new_body = json.dumps(json_body).encode("utf-8")
    except:
        model_name = None

    # If no model in body, try to get the first available one
    if not model_name:
        gguf_files = find_gguf_files()
        if gguf_files:
            model_name = os.path.basename(gguf_files[0]).replace(".gguf", "")

    if not model_name:
        return JSONResponse(status_code=404, content={"error": "No models available"})

    if not server_ready.is_set() or current_model != model_name:
        await start_vllm_server(model_name)
    
    await server_ready.wait()

    async with lock:
        last_request_time = time.time()

    client = httpx.AsyncClient()
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/{full_path}"
    
    headers = dict(request.headers)
    headers.pop("host", None)
    # Need to update content-length header if body is changed
    if new_body != body:
        headers["content-length"] = str(len(new_body))

    async def stream_response():
        async with client.stream(
            request.method,
            url,
            params=request.query_params,
            content=new_body,
            headers=headers,
            timeout=None
        ) as response:
            # This is a workaround to FastAPI's behavior of not being able to set status_code on StreamingResponse
            # We have to send headers manually.
            raw_headers = response.headers.raw
            # TODO: Find a better way to do this
            # yield raw_headers
            async for chunk in response.aiter_bytes():
                yield chunk

    return StreamingResponse(
        stream_response(),
        status_code=200, # This will be overridden by the actual response
        media_type=request.headers.get("accept")
    )

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT)

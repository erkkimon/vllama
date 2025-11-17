import asyncio
import httpx
import subprocess
import time
import os
import signal
import sys
import shlex
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

# Qwen Chat Template (from Hugging Face documentation)
QWEN_CHAT_TEMPLATE_STRING = """{%% for message in messages %%}{%% if message['role'] == 'user' %%}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{%% elif message['role'] == 'system' %%}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{%% elif message['role'] == 'assistant' %%}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{%% endif %%}{%% if loop.last and add_generation_prompt %%}{{ '<|im_start|>assistant\n' }}{%% endif %%}"""

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
    family: str
    max_model_len: int
    quantization: str
    tokenizer_type: str
    hf_tokenizer_path_or_name: str # Hugging Face tokenizer ID or path to generated tokenizer.json

from gguf import GGUFReader, GGUFValueType

ollama_discovered_models: List[OllamaModel] = []

def get_gguf_metadata(gguf_path: str) -> dict:
    """
    Reads a GGUF file using the gguf-py library and returns a dictionary of key-value metadata.
    """
    metadata = {}
    try:
        reader = GGUFReader(gguf_path, 'r')
        for key, field in reader.fields.items():
            if key in ["general.architecture", "tokenizer.ggml.pre"]:
                # The actual value is stored in the 'parts' list, at the index specified by the 'data' list.
                if field.data:
                    value_index = field.data[0]
                    value_part = field.parts[value_index]
                    
                    # The part is a numpy array of bytes that needs to be decoded.
                    if hasattr(value_part, 'tobytes'):
                        value_bytes = value_part.tobytes()
                        # Decode and strip any trailing null characters
                        metadata[key] = value_bytes.decode('utf-8', errors='ignore').strip('\x00')

    except PermissionError:
        logging.error(
            "Permission denied when trying to read GGUF file: %s. "
            "Please ensure you have read permissions for the Ollama models directory. "
            "You may need to run this script with sudo or add your user to the 'ollama' group.",
            gguf_path
        )
    except Exception as e:
        # This will catch errors like the "not a valid GGMLQuantizationType" for specific models
        # and prevent the entire discovery process from crashing.
        logging.error("Failed to parse GGUF metadata for %s: %s", gguf_path, str(e))
    return metadata

def get_model_family(metadata: dict) -> str:
    """
    Determines the model family based on its GGUF metadata.
    """
    architecture = metadata.get("general.architecture", "")
    tokenizer_pre = metadata.get("tokenizer.ggml.pre", "")

    # --- Qwen Detection Logic ---
    # The logic to detect Qwen models is preserved here for future use.
    # For now, they will be classified as 'unknown' per the requirements.
    # if "qwen" in architecture:
    #     return "qwens"

    if architecture == "llama" and tokenizer_pre == "tekken":
        return "mistrals"
    
    if architecture == "qwen2":
        # This covers the DeepSeek models that are built on the qwen2 architecture.
        return "deepseeks"

    return "unknown"


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
            
            # New family classification logic
            family = "unknown"
            if gguf_path and os.path.exists(gguf_path):
                gguf_metadata = get_gguf_metadata(gguf_path)
                family = get_model_family(gguf_metadata)
                # The architecture from gguf_dump is more reliable
                architecture = gguf_metadata.get("general.architecture", architecture)

            # Set tokenizer to auto, as it will be handled by family-based logic later
            tokenizer_type = "auto"
            hf_tokenizer_path_or_name = "auto"

            # Use the minimum of reported and calculated max_model_len
            final_max_model_len = max_model_len
            calculated_len = calculate_max_model_len(gguf_path)
            if calculated_len > 0 and calculated_len < final_max_model_len:
                final_max_model_len = calculated_len

            if gguf_path and os.path.exists(gguf_path):
                # Exclude known problematic models for vLLM GGUF support
                if architecture.lower() == "gptoss":
                    logging.warning("Excluding Ollama model %s (architecture: %s) due to known vLLM GGUF compatibility issues.", model_id, architecture)
                    continue

                ollama_discovered_models.append(OllamaModel(
                    id=model_id,
                    name=model_id,
                    gguf_path=gguf_path,
                    architecture=architecture,
                    family=family,
                    max_model_len=final_max_model_len,
                    quantization=quantization,
                    tokenizer_type=tokenizer_type,
                    hf_tokenizer_path_or_name=hf_tokenizer_path_or_name
                ))
                logging.info("Discovered Ollama model: %s (Family: %s) at %s with max_model_len %d", model_id, family, gguf_path, final_max_model_len)
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

    vllm_dtype = "auto" # Default dtype
    family = "unknown"

    if ollama_model_config:
        model_path_for_vllm = ollama_model_config.gguf_path
        served_model_name = ollama_model_config.name
        max_model_len = ollama_model_config.max_model_len
        family = ollama_model_config.family

        # Set command parameters based on family
        if family == "mistrals":
            tokenizer_path = "mistralai/Devstral-Small-2507"
            tokenizer_mode = "mistral"
            tool_call_parser = "mistral"
        elif family == "deepseeks":
            tokenizer_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
            tokenizer_mode = "auto"
            tool_call_parser = "deepseek_r1"
        else: # unknown family
            tokenizer_path = "auto"
            tokenizer_mode = "auto"
            tool_call_parser = "openai" # Generic fallback

        logging.info("Serving Ollama model %s (Family: %s) with max_model_len %d, tokenizer %s, tokenizer_mode %s, tool_call_parser %s, dtype %s", 
                     served_model_name, family, max_model_len, tokenizer_path, tokenizer_mode, tool_call_parser, vllm_dtype)
    else:
        # This block handles manually found GGUF files not managed by Ollama
        model_path = os.path.join(MODELS_DIR, f"{model_name}.gguf")
        if not os.path.exists(model_path):
            all_gguf_files = find_gguf_files()
            for f in all_gguf_files:
                if os.path.basename(f).replace(".gguf", "") == model_name:
                    model_path = f
                    break
        
        if not os.path.exists(model_path):
            logging.error("Model %s not found.", model_name)
            return None

        # Treat manually found GGUF as 'unknown' family for safe defaults
        family = "unknown"
        model_path_for_vllm = model_path
        served_model_name = os.path.basename(model_path).replace(".gguf", "")
        max_model_len = calculate_max_model_len(model_path)
        tokenizer_path = "auto"
        tokenizer_mode = "auto"
        tool_call_parser = "openai" # Generic fallback
        vllm_dtype = "auto"
        logging.info("Serving local GGUF model %s (Family: %s) with max_model_len %d, tokenizer %s, tokenizer_mode %s, tool_call_parser %s, dtype %s", 
                     served_model_name, family, max_model_len, tokenizer_path, tokenizer_mode, tool_call_parser, vllm_dtype)

    command_parts = [
        f"{sys.executable} -m vllm.entrypoints.openai.api_server",
        f"--host {VLLM_HOST}",
        f"--port {VLLM_PORT}",
        "--gpu-memory-utilization 0.95",
        "--disable-log-stats",
        "--enforce-eager",
        f"--model {model_path_for_vllm}",
        f"--served-model-name {served_model_name}",
    ]

    tokenizer_arg = f"--tokenizer {tokenizer_path}" if tokenizer_path != "auto" else ""
    if tokenizer_arg:
        command_parts.append(tokenizer_arg)

    chat_template_arg = ""
    # This specific template is needed for Qwen models, but not for DeepSeek models
    if ollama_model_config and "qwen" in ollama_model_config.architecture and family != "deepseeks":
        # Properly quote the chat template string for the shell
        quoted_template = shlex.quote(QWEN_CHAT_TEMPLATE_STRING)
        chat_template_arg = f"--chat-template {quoted_template}"
    if chat_template_arg:
        command_parts.append(chat_template_arg)

    # Conditional parser arguments based on family. DeepSeek requires a specific parser.
    if family == "deepseeks":
        command_parts.append(f"--reasoning-parser {tool_call_parser}")
    else:
        command_parts.append("--enable-auto-tool-choice")
        command_parts.append(f"--tool-call-parser {tool_call_parser}")

    # Add final arguments
    command_parts.append(f"--max-model-len {max_model_len}")
    if family != "deepseeks":
        command_parts.extend([
            f"--tokenizer-mode {tokenizer_mode}",
            f"--dtype {vllm_dtype}"
        ])

    return " \
    ".join(command_parts)

# --- Global State ---
vllm_process = None
last_request_time = None
lock = asyncio.Lock()
server_ready = asyncio.Event()
current_model = None

# --- vLLM Process Management ---
def is_port_free(host: str, port: int) -> bool:
    """Check if the given port is free (nothing is listening on it)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return False # Connection successful, port is NOT free
        except ConnectionRefusedError:
            return True # Connection refused, port IS free
        except Exception:
            # Other errors might mean the port is not free or in a weird state
            return False

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
            vllm_log_file = open(f"{LOG_DIR}/vllama.log", "a")
            
            # Capture stderr to debug potential startup crashes
            vllm_process = subprocess.Popen(
                command, 
                shell=True, 
                preexec_fn=os.setsid, 
                stdout=vllm_log_file, 
                stderr=subprocess.PIPE,
                text=True  # Decode stderr as text
            )
            current_model = model_name
            
            # Asynchronously wait for the server to be ready
            while not await asyncio.to_thread(is_vllm_ready):
                # Check if the process terminated unexpectedly
                if vllm_process.poll() is not None:
                    # Read the error output
                    stderr_output = ""
                    if vllm_process.stderr:
                        stderr_output = vllm_process.stderr.read()
                    logging.critical(f"vLLM server process terminated unexpectedly. STDERR:\n{stderr_output}")
                    return
                logging.info("Waiting for vLLM server to be ready...")
                await asyncio.sleep(2)
            
            logging.info("vLLM server is ready.")
            last_request_time = time.time()
            server_ready.set()

def kill_vllm_server():
    """Kill the vLLM server process and ensure its port is free."""
    global vllm_process, current_model
    if vllm_process:
        server_ready.clear()
        logging.info("Killing vLLM server (PID: %d)...", vllm_process.pid)
        try:
            pgid = os.getpgid(vllm_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            
            # Wait for the process to terminate gracefully
            try:
                vllm_process.wait(timeout=10) # Give it 10 seconds to shut down
                logging.info("vLLM process terminated gracefully.")
            except subprocess.TimeoutExpired:
                logging.warning("vLLM process did not terminate gracefully after SIGTERM. Force killing.")
                os.killpg(pgid, signal.SIGKILL) # Force kill
                vllm_process.wait(timeout=5) # Wait a bit more for force kill

            # Now, wait for the port to be truly free
            timeout_start = time.time()
            while not is_port_free(VLLM_HOST, VLLM_PORT):
                if time.time() - timeout_start > 30: # Max 30 seconds wait for port to free
                    logging.error("Timeout waiting for vLLM server port to free up. This might indicate a problem.")
                    break # Exit loop, but log the issue
                logging.info("Waiting for vLLM server port to free up...")
                time.sleep(0.5) # Check every 0.5 seconds

        except ProcessLookupError:
            logging.info("vLLM process already dead.") # Process already dead
        except Exception as e:
            logging.error("Error during vLLM server kill: %s", e)

        vllm_process = None
        current_model = None
        logging.info("vLLM server killed and port confirmed free (or timed out waiting).")

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

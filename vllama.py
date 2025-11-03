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
    
    # Combine and remove duplicates
    all_models = list(set(local_models + system_models))
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
    model_path = os.path.join(MODELS_DIR, f"{model_name}.gguf")
    if not os.path.exists(model_path):
        # Try to find model in all available locations
        all_gguf_files = find_gguf_files()
        for f in all_gguf_files:
            if model_name in f:
                model_path = f
                break
    
    if not os.path.exists(model_path):
        logging.error("Model %s not found.", model_name)
        return None

    served_model_name = os.path.basename(model_path).replace(".gguf", "")
    max_model_len = calculate_max_model_len(model_path)
    
    logging.info("Serving model %s as %s with max_model_len %d", model_path, served_model_name, max_model_len)

    return f"""
python -m vllm.entrypoints.openai.api_server \
    --host {VLLM_HOST} \
    --port {VLLM_PORT} \
    --gpu-memory-utilization 0.95 \
    --disable-log-stats \
    --enforce-eager \
    --model {model_path} \
    --served-model-name {served_model_name} \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --tokenizer mistralai/Devstral-Small-2507 \
    --tokenizer-mode mistral \
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
    yield
    kill_vllm_server()

app = FastAPI(lifespan=lifespan)

# --- Models Endpoint ---
@app.get("/v1/models")
async def list_models():
    """Return a list of available GGUF models."""
    gguf_files = find_gguf_files()
    models = []
    for f in gguf_files:
        model_id = os.path.basename(f).replace(".gguf", "")
        max_len = calculate_max_model_len(f)
        # Format max_len to 'k' notation
        if max_len > 1000:
            max_len_str = f"{max_len // 1000}k"
        else:
            max_len_str = str(max_len)
        
        models.append({
            "id": f"{model_id} ({max_len_str})",
            "object": "model",
            "created": int(os.path.getctime(f)),
            "owned_by": "vllama",
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

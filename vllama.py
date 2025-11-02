
import asyncio
import httpx
import subprocess
import time
import os
import signal
import threading
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# --- Logging Configuration ---
LOG_DIR = "logs"
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
VLLM_MODEL_COMMAND = """
python -m vllm.entrypoints.openai.api_server \
    --host {host} \
    --port {port} \
    --gpu-memory-utilization 0.95 \
    --disable-log-stats \
    --enforce-eager \
    --model /opt/vllama/models/Devstral-Small-2505-abliterated.i1-Q2_K_S.gguf \
    --served-model-name devstral-vanilla-gguf \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --tokenizer mistralai/Devstral-Small-2507 \
    --tokenizer-mode mistral \
    --enforce-eager \
    --max-model-len 74880
"""

# --- Global State ---
vllm_process = None
last_request_time = None
lock = asyncio.Lock()
server_ready = asyncio.Event()

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

async def start_vllm_server():
    """Start the vLLM server process and wait for it to be ready."""
    global vllm_process, last_request_time
    async with lock:
        if vllm_process is None or vllm_process.poll() is not None:
            server_ready.clear()
            logging.info("Starting vLLM server...")
            command = VLLM_MODEL_COMMAND.format(host=VLLM_HOST, port=VLLM_PORT)
            vllm_log_file = open(f"{LOG_DIR}/vllm.log", "w")
            vllm_err_file = open(f"{LOG_DIR}/vllm.err", "w")
            vllm_process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=vllm_log_file, stderr=vllm_err_file)
            
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
    global vllm_process
    if vllm_process:
        server_ready.clear()
        logging.info("Killing vLLM server...")
        try:
            os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
            vllm_process.wait()
        except ProcessLookupError:
            pass # Process already dead
        vllm_process = None
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

# --- Proxy Endpoint ---
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, full_path: str):
    """The main proxy endpoint."""
    global last_request_time

    if not server_ready.is_set():
        await start_vllm_server()
    
    await server_ready.wait()

    async with lock:
        last_request_time = time.time()

    client = httpx.AsyncClient()
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/{full_path}"
    
    headers = dict(request.headers)
    headers.pop("host", None)

    body = await request.body()

    async def stream_response():
        async with client.stream(
            request.method,
            url,
            params=request.query_params,
            content=body,
            headers=headers,
            timeout=None
        ) as response:
            # Check for errors from the vLLM server
            if response.status_code != 200:
                error_body = await response.aread()
                logging.error(f"Error from vLLM server: {response.status_code} {error_body.decode()}")
                return

            async for chunk in response.aiter_bytes():
                yield chunk

    return StreamingResponse(
        stream_response(),
        status_code=200, # This will be overridden by the actual response
        media_type="application/json" # This will be overridden by the actual response
    )

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT)

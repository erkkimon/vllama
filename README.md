# vllama: Faster Ollama Inference with vLLM Acceleration 🚀

vllama is an open source hybrid server that combines Ollama's seamless model management with vLLM's lightning-fast GPU inference, delivering a drop-in OpenAI-compatible API for optimized performance. If you're searching for Ollama performance optimizations, ways to speed up Ollama inference, or Ollama GPU acceleration techniques, vllama bridges the gap by using vLLM for high-speed generation while borrowing Ollama's repository for GGUF models. It runs on port 11435 as a faster alternative to Ollama (port 11434), with lazy model loading to VRAM on demand and automatic unloading when idle—ideal for efficient resource use in multi-user setups.

At the moment this has been developed for my personal purposes, but it might work with other models and setups also. I am using my favorite model Devstral Small with RTX 3090 Ti on Arch Linux, and this combo is proven to work. I will add support for more models as I need them, but PRs are welcome. 

## Vision

The vision for vllama is to make high-performance AI inference accessible and efficient for everyone using Ollama models. By integrating vLLM's advanced GPU optimizations, it addresses common pain points like slow Ollama inference on large models while maintaining Ollama's simple pull-and-run workflow. Whether you're looking for OpenAI compatible vLLM server solutions or methods to unload vLLM model when idle, vllama aims to be the go-to tool for users wanting faster Ollama with vLLM without sacrificing ease of use. It's designed for developers, families sharing hardware, or anyone optimizing Ollama on NVIDIA GPUs like RTX 3090 Ti, emphasizing open source principles and automation ideas for deployment.

## Quick start

If you are using Arch Linux, installation is easy.

#### AUR Installation (as system service)

The AUR package includes a bundled venv312 for dependency isolation.

```bash
# Install from AUR
pikaur -S vllama

# Start services
sudo systemctl start ollama vllama

# Test
curl http://localhost:11435/v1/models
```

The AUR package includes a bundled venv312 environment and systemd service. After installation, enable the service for automatic startup.

```
# Install from AUR (includes venv312 and service files)
pikaur -S vllama

# Pull a test model with Ollama (dependency)
ollama pull huihui_ai/devstral-abliterated

# Enable and start vllama service
sudo systemctl enable vllama
sudo systemctl start vllama

# Verify services are running
sudo systemctl status ollama vllama

# Test API endpoints
curl http://localhost:11435/v1/models  # Lists Ollama models
curl http://localhost:11435/health     # Shows vLLM status

# Test network access (from another machine)
curl http://<ferocitee-ip>:11435/v1/models
```

### Development (isolated)

To set up vllama for development on Arch Linux, clone the repository and install dependencies. This assumes you have Ollama running for model management.

#### Setting up development environment

```bash
# Clone repo
git clone https://github.com/erkkimon/vllama.git
cd vllama

# Create Python 3.12 virtual environment
python3.12 -m venv venv312
source venv312/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull a test model with Ollama, works with 24 Gb of VRAM (Nvidia RTX 3090, RTX 4090, etc.)
ollama pull huihui_ai/devstral-abliterated
```

#### Running development environment

```
# Activate Python 3.12 environment
source venv312/bin/activate

# Start Ollama (dependency for model listing)
sudo systemctl start ollama

# Run vllama
python vllama.py

# Test endpoints (in another terminal)
curl http://localhost:11435/v1/models
curl http://localhost:11435/health
```

## Post-install customizations

```
# Override defaults without editing package files
sudo mkdir -p /etc/systemd/system/vllama.service.d
sudo tee /etc/systemd/system/vllama.service.d/custom.conf > /dev/null <<EOF
[Service]
Environment="IDLE_TIMEOUT=600"  # 10 minutes instead of 5
Environment="OLLAMA_URL=http://localhost:11434"
EOF

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart vllama
```

## Service management commands

```
# Start/stop/restart
sudo systemctl start vllama
sudo systemctl stop vllama
sudo systemctl restart vllama

# Check status and logs
sudo systemctl status vllama
journalctl -u vllama -f  # Follow logs

# Disable auto-start
sudo systemctl disable vllama
```

## Usage with Cline (or Roo Code)

vllama provides a fully OpenAI-compatible API, making it a seamless drop-in replacement for tools like Roo Code and Cline. Both tools can connect to vllama's endpoint (http://localhost:11435) instead of Ollama (http://localhost:11434) to get faster inference while maintaining the same model management workflow. You can install new models using ollama but use them via vllama API.

Endpoint: http://localhost:11435 (or http://<ip-address|hostname>:11435 for network access from other computers in the network)

1. Open Roo Code settings and set the API base URL to `http://localhost:11435`
2. Model selection: huihui_ai/devstral-abliterated should appear as an option (same as Ollama), assuming that you have pulled it using ollama.
3. Start coding: Roo Code will use vLLM for 3-4x faster responses

## Implementation

vllama is implemented as a lightweight Python server using FastAPI for the API layer and uvicorn for ASGI hosting. It proxies model listing requests to Ollama's endpoint (`/api/tags`) and converts them to OpenAI format, ensuring seamless compatibility for tools like Cline or Roo Code. For inference, it uses vLLM to handle GGUF models extracted from Ollama's blobs, with lazy loading: the vLLM engine spins up on the first chat completion request (loading to VRAM) and unloads after a configurable idle timeout (default 5 minutes) via a background thread timer. This achieves Ollama performance optimization by offloading heavy computation to vLLM's CUDA-optimized backend, supporting features like rope scaling, flash attention, and KV cache quantization. The tokenizer is borrowed from Ollama's embedded GGUF data, avoiding mismatch issues and ensuring consistent behavior. Emojis in logs (e.g., 🚀 for load, 🛑 for unload) make monitoring fancy and user-friendly.

Important notes: vllama requires Ollama running in parallel for model management—treat it as a companion tool. If you're troubleshooting vLLM GGUF support or seeking ways to automate Ollama with vLLM, note that model paths are dynamically fetched via Ollama CLI, making it robust for updates. For Arch Linux users, pikaur handles dependencies smoothly.

## Important Notes for Developers

- **Dependencies and Setup**: vllama relies on Ollama for model pulling and registry (e.g., `ollama pull`), and vLLM for inference. Install via `pip install -r requirements.txt`. Ensure Ollama is running on port 11434; vllama proxies to it for `/v1/models`. Test with `python vllama.py`—it binds to 0.0.0.0:11435 for network access.
- **Tokenizer Handling**: Ollama's GGUF blobs include embedded tokenizers, which vLLM uses directly (`tokenizer=model_path`). This eliminates Roberta/Llama mismatches—perfect for Devstral models. If issues arise, verify GGUF path extraction in `get_ollama_model_path`.
- **Idle Unloading**: The background thread checks activity every 60s and unloads vLLM if idle >5m, freeing VRAM. Customize `IDLE_TIMEOUT` for your setup. For debugging, add `--log-level debug` to uvicorn.
- **Performance**: On RTX 3090 Ti, expect 20+ t/s for Q4_K_S GGUF models with 47k context—far faster than Ollama. Test with /health endpoint for load status.
- **Limitations**: No direct Ollama CLI integration (use Ollama for pull/ls). For Debian/Ubuntu building, see contribution section — I don't have Ubuntu for testing, so PRs welcome. BTRFS-safe; no internet access needed post-install.
- **Security**: CORS enabled for family multi-user; add auth if needed via FastAPI middleware.

## Contribution Instructions

Contributions are welcome to make vllama even better! Fork the repo on GitHub, make changes, and submit a PR. Focus on:
- **Bug fixes**: Inference errors, path extraction, or tokenizer issues.
- **Features**: Add support for more Ollama parameters (e.g., rope scaling via env vars).
- **Testing**: Verify on NVIDIA GPUs; add unit tests for proxy and unloading.
- **Packaging**: For Debian/Ubuntu building, create deb packages – I don't have Ubuntu for testing, so if you do, please contribute! Update PKGBUILD for AUR updates.

## Architecture

vllama's architecture is a hybrid proxy-inference system:
- **API Layer** (FastAPI + Uvicorn): Handles OpenAI-compatible endpoints. `/v1/models` proxies to Ollama's `/api/tags` and reformats to OpenAI spec. `/v1/chat/completions` processes requests, builds prompts, and routes to vLLM.
- **Model Management (Ollama Dependency)**: Borrows Ollama for pulling, listing, and path extraction (via CLI/subprocess). This keeps vllama lightweight—Ollama acts as "package manager" for GGUF models.
- **Inference Engine (vLLM)**: Lazy-loaded AsyncLLMEngine for fast CUDA inference. Loads on first request, uses Ollama's GGUF path and tokenizer. Background thread monitors idle time and unloads (del engine) to free VRAM.
- **Health/Monitoring**: /health endpoint shows status (loaded/unloaded, last activity) for debugging.
- **Systemd Integration**: Runs as daemon with drop-in overrides for multi-user (network, CORS, resources). AUR package installs as dependency on Ollama/vLLM.

This architecture ensures vllama is a seamless, faster Ollama alternative for GPU-accelerated setups, ideal for those seeking vLLM as OpenAI server with lazy loading or ways to speed up Ollama without full migration. 🌟

### Roo / Cline Compatibility

This server is designed to work with RooCode and Cline, which expect strict
OpenAI-compatible semantics around tool calls.

- **Tool enforcement**: When the client specifies `tool_choice="required"` (or
  a specific function), the assistant must *only* output a `[TOOL_CALLS][…]`
  block as its first action. Any pre-tool prose will cause Roo to inject an
  error. The server enforces this by adjusting the system prompt and suppressing
  pre-tool output in streaming mode.

- **Sanitization**: Some models (especially DevStral clones) emit hidden
  `[thinking]…[/thinking]` or `[plan]…` blocks. These are stripped before being
  sent to clients. In streaming mode, whitespace at chunk boundaries is
  preserved so text doesn’t become scrambled.

- **Stop tokens**: Extra stops like `[/ASSISTANT]` are added to prevent the
  model from leaking unwanted closing tags or meta.

# Troubleshooting

```
# Check if venv312 is working
ls -la /opt/vllama/venv312/bin/python  # Should exist

# Verify vLLM loaded correctly
curl http://localhost:11435/health  # Should show "loaded" after first request

# Check logs for errors
journalctl -u vllama --since "1 hour ago"

# Test Ollama dependency
curl http://localhost:11434/api/tags  # Should list models
```
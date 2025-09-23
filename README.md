# vllama: Faster Ollama Inference with vLLM Acceleration 🚀

vllama is an open-source hybrid server that combines Ollama's seamless model management with vLLM's lightning-fast GPU inference, delivering a drop-in OpenAI-compatible API for optimized performance. If you're searching for Ollama performance optimizations, ways to speed up Ollama inference, or Ollama GPU acceleration techniques, vllama bridges the gap by using vLLM for high-speed generation while borrowing Ollama's repository for GGUF models. It runs on port 11435 as a faster alternative to Ollama (port 11434), with lazy model loading to VRAM on demand and automatic unloading when idle—ideal for efficient resource use in multi-user setups.

At the moment, this has been developed for personal purposes, but it works with a variety of models and setups. The developer is using their favorite model, Devstral Small, with an RTX 3090 Ti on Arch Linux, and this combo is proven to work. More model support will be added as needed, but pull requests are welcome.

## Vision

The vision for vllama is to make high-performance AI inference accessible and efficient for everyone using Ollama models. By integrating vLLM's advanced GPU optimizations, it addresses common pain points like slow Ollama inference on large models while maintaining Ollama's simple pull-and-run workflow. Whether you're looking for OpenAI-compatible vLLM server solutions or methods to unload vLLM models when idle, vllama aims to be the go-to tool for users wanting faster Ollama with vLLM without sacrificing ease of use. It's designed for developers, families sharing hardware, or anyone optimizing Ollama on NVIDIA GPUs like RTX 3090 Ti, emphasizing open-source principles and automation ideas for deployment.

## Quick Start

### AUR Installation (as system service)

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

```bash
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
curl http://your-ip-address:11435/v1/models
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

```bash
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

## Post-install Customizations

```bash
# Override defaults without editing package files
sudo mkdir -p /etc/systemd/system/vllama.service.d
sudo tee /etc/systemd/system/vllama.service.d/custom.conf > /dev/null <<EOF
[Service]
Environment="IDLE_TIMEOUT=600"
Environment="MAX_MODEL_LEN=65536"
Environment="REPORTED_CONTEXT_WINDOW=65536"
Environment="DEVSTRAL_CONTEXT_WINDOW=65536"
EOF

# Reload and restart service
sudo systemctl daemon-reload
sudo systemctl restart vllama
```

## Context Length Configuration

vllama provides accurate context length reporting to clients like Roo Code through multiple endpoints and configurable context windows.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_MODEL_LEN` | Maximum model length for vLLM (0 = auto-detect) | `65536` |
| `REPORTED_CONTEXT_WINDOW` | Context window reported to clients (overrides actual) | `65536` |
| `DEVSTRAL_CONTEXT_WINDOW` | Context window specifically for Devstral models | `65536` |
| `IDLE_TIMEOUT` | Seconds before unloading vLLM from VRAM | `300` |
| `MAX_TOKENS_DEFAULT` | Default max tokens for completions | `1024` |

### Context Window Determination

The reported context window is determined using model-specific detection:

1. **Devstral Models** - Uses `DEVSTRAL_CONTEXT_WINDOW` (default: 65536)
   - Detects: `devstral`, `huihui_ai/devstral`, `devstral-abliterated`
   - Example: `huihui_ai/devstral-abliterated:latest` → 65536 tokens
2. **Other Models** - Uses `REPORTED_CONTEXT_WINDOW` (default: 65536)
3. **Actual vLLM engine context** (when engine is loaded) - real context from loaded model
4. **Default fallback** (65536) - safe default for most models

### Model-Specific Configuration

For Devstral models, you can override the context window independently:

```bash
# Set Devstral models to 32k context while keeping others at 128k
export DEVSTRAL_CONTEXT_WINDOW=32768
export REPORTED_CONTEXT_WINDOW=131072
python3 vllama.py
```

### Context Information Endpoints

```bash
# Get model list with accurate context windows
curl http://localhost:11435/v1/models

# Get LiteLLM-compatible model info (recommended for Roo Code)
curl http://localhost:11435/v1/model/info

# Health check with context details
curl http://localhost:11435/health
```

### Example Responses

**Standard OpenAI Models Endpoint:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "huihui_ai/devstral-abliterated:latest",
      "object": "model",
      "created": 1699564800,
      "owned_by": "vllama",
      "context_window": 65536,
      "max_position_embeddings": 65536,
      "size": "13.5GB"
    }
  ]
}
```

**LiteLLM Model Info Endpoint:**
```json
{
  "data": [
    {
      "model_name": "huihui_ai/devstral-abliterated:latest",
      "litellm_params": {
        "model": "huihui_ai/devstral-abliterated:latest",
        "max_tokens": 65536,
        "supports_function_calling": true,
        "supports_parallel_function_calling": false,
        "supports_vision": false
      },
      "model_info": {
        "max_tokens": 65536,
        "max_input_tokens": 65536,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "vllm",
        "mode": "chat"
      }
    }
  ]
}
```

## Roo Code Integration

To use vllama with Roo Code and get correct context length information:

### Option 1: OpenAI Provider (Recommended)

Configure Roo Code to use vllama as an OpenAI-compatible provider with custom model info:

**In Roo Code Settings:**
```json
{
  "apiProvider": "openai",
  "openAiBaseUrl": "http://localhost:11435/v1",
  "openAiApiKey": "not-required",
  "openAiModelId": "huihui_ai/devstral-abliterated:latest",
  "openAiCustomModelInfo": {
    "contextWindow": 65536,
    "maxTokens": 4096,
    "supportsImages": false,
    "supportsPromptCache": false
  }
}
```

**Key Points:**
- **`openAiCustomModelInfo.contextWindow`**: Set this to match your `REPORTED_CONTEXT_WINDOW` environment variable
- **Model ID**: Use the exact model name from `ollama list`
- **Base URL**: Point to your vllama server with `/v1` suffix

### Option 2: LiteLLM Provider (Dynamic)

Configure as LiteLLM provider for automatic context detection:

```json
{
  "apiProvider": "litellm",
  "litellmBaseUrl": "http://localhost:11435",
  "litellmApiKey": "not-required",
  "litellmModelId": "huihui_ai/devstral-abliterated:latest"
}
```

### Environment Variable Synchronization

To ensure Roo Code shows the correct context window, synchronize these values:

```bash
# Set vllama context window
export REPORTED_CONTEXT_WINDOW=65536

# Configure Roo Code openAiCustomModelInfo.contextWindow to match: 65536
```

**Example configurations for different context windows:**

```bash
# For 128k context
export REPORTED_CONTEXT_WINDOW=131072
# Set Roo Code openAiCustomModelInfo.contextWindow: 131072

# For 32k context
export REPORTED_CONTEXT_WINDOW=32768
# Set Roo Code openAiCustomModelInfo.contextWindow: 32768
```

### Why OpenAI Provider Needs Manual Configuration

Unlike LiteLLM providers, OpenAI providers in Roo Code use **user-configured model information** rather than fetching it from the API. This means:

- **OpenAI Provider**: Uses `openAiCustomModelInfo.contextWindow` from your settings
- **LiteLLM Provider**: Fetches context window dynamically from `/v1/model/info`

Both approaches work, but OpenAI provider gives you explicit control over the reported context window.
# vllama: Faster Ollama Inference with vLLM Acceleration 🚀

**vllama works on all platforms! See instructions for your OS below.**

vllama is an open-source hybrid server that combines Ollama's seamless model management with vLLM's lightning-fast GPU inference, delivering a drop-in OpenAI-compatible API for optimized performance. It's designed for efficient **local LLM** operations and **on-device AI**, running on port 11435 as a faster alternative to Ollama (port 11434).

**Key Features:**
*   **Efficient VRAM Management:** Models are loaded on demand when a request is received and automatically unloaded after 5 minutes of inactivity, freeing up VRAM.
*   **Automatic Context Length Optimization:** The context length is automatically calculated and optimized for your available VRAM, ensuring efficient performance without manual configuration.
*   **Broad Ollama Model Support:** All Ollama models are served automatically. While vLLM's GGUF support is still experimental, Devstral models are currently proven to work exceptionally well.
*   **Network-Wide Access:** vllama serves models for the entire network by default, making it easily accessible across your local network.

If you're searching for Ollama performance optimizations, ways to speed up Ollama inference, or Ollama GPU acceleration techniques, vllama bridges the gap by using vLLM for high-speed generation while borrowing Ollama's repository for GGUF models.

At the moment, this has been developed for personal purposes, but it works with a variety of models and setups. The developer is using their favorite model, Devstral Small, with an RTX 3090 Ti on Arch Linux, and this combo is proven to work. More model support will be added as needed, but pull requests are welcome.

## Quick Start

> [!TIP]
> ### Arch Linux Users (AUR)
> The fastest way to get started on Arch Linux is by using the [AUR package](https://aur.archlinux.org/packages/vllama). This will install `vllama` as a systemd service.
>
> ```bash
> # Install from AUR (e.g., with yay or paru)
> yay -S vllama
>
> # Enable and start the service
> sudo systemctl enable --now ollama.service
> sudo systemctl enable --now vllama.service
>
> # Your server is running!
> curl http://localhost:11435/v1/models
> ```

## Logging

`vllama` logs important events and errors to help with debugging and monitoring.

*   **Service (Production):** When running as a systemd service, logs are located in `/opt/vllama/logs/`.
*   **Development:** When running `vllama.py` directly from the repository, logs are located in a `logs/` directory within the project root.

To monitor the logs in real-time, you can use the `tail -f` command on the appropriate log file. For example, when running as a service:

```bash
tail -f /opt/vllama/logs/vllama.log
```

If you encounter issues, checking these log files is the first step to diagnosing the problem.

> ### Other Linux Distributions
> You can run `vllama` directly from the repository.

## System Service Setup (Ubuntu/Debian)

For distributions like Ubuntu or Debian, you can set up `vllama` to run as a systemd service for automatic startup.

1.  **Install Dependencies:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv git
    ```

2.  **Clone and Prepare the Repository:**
    ```bash
    # Clone to /opt
    git clone https://github.com/erkkimon/vllama.git /tmp/vllama
    sudo mv /tmp/vllama /opt/vllama
    cd /opt/vllama
    
    # Create a user for the service
    sudo useradd -r -s /usr/bin/nologin -d /opt/vllama vllama
    
    # Set up environment and permissions
    sudo python3 -m venv venv312
    sudo venv312/bin/pip install -r requirements.txt
    sudo chown -R vllama:vllama /opt/vllama
    ```

3.  **Configure and Enable the Service:**
    *   First, edit the service file to use the correct user and group.
        ```bash
        # The repository's service file is already configured for the 'vllama' user.
        # If you use a different user, edit the file accordingly.
        sudo cp /opt/vllama/vllama.service /etc/systemd/system/vllama.service
        ```
    *   Reload the systemd daemon and start the service.
        ```bash
        sudo systemctl daemon-reload
        sudo systemctl enable --now vllama.service
        
        # Check the status
        sudo systemctl status vllama.service
        ```

## Windows Users

For instructions on running `vllama` on Windows, please see the **[Windows Setup Guide](./docs/windows.md)**.

## Vision

The vision for vllama is to make high-performance AI inference accessible and efficient for everyone using Ollama models. By integrating vLLM's advanced GPU optimizations, it addresses common pain points like slow Ollama inference on large models while maintaining Ollama's simple pull-and-run workflow. This makes it an ideal solution for **local programming** and **local models powered software development**, enabling **agents powered by local LLM** to run efficiently. Whether you're looking for OpenAI-compatible vLLM server solutions or methods to unload vLLM models when idle, vllama aims to be the go-to tool for users wanting faster Ollama with vLLM without sacrificing ease of use. It's designed for developers, families sharing hardware, or anyone optimizing Ollama on NVIDIA GPUs like RTX 3090 Ti, emphasizing open-source principles and automation ideas for deployment.

## Configuration

vllama can be configured using environment variables. You can set these in your systemd service file or directly in your shell environment before running `vllama.py`.

```bash
# Override defaults without editing package files
sudo mkdir -p /etc/systemd/system/vllama.service.d
sudo tee /etc/systemd/system/vllama.service.d/custom.conf > /dev/null <<EOF
[Service]
Environment="IDLE_TIMEOUT=600"
Environment="MAX_MODEL_LEN=65536"
Environment="MAX_TOKENS_DEFAULT=1024"
EOF

# Reload and restart service
sudo systemctl daemon-reload
sudo systemctl restart vllama
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_MODEL_LEN` | Maximum model length for vLLM. Set to `0` for auto-detection based on available VRAM. | `65536` |
| `IDLE_TIMEOUT` | Seconds before an idle vLLM model is unloaded from VRAM. | `300` |
| `MAX_TOKENS_DEFAULT` | Default maximum tokens for completions if not specified by the client. | `1024` |

### GGUF Model Compatibility

vLLM's GGUF support is still experimental. While Devstral models are proven to work, other GGUF models might not function as expected. If you successfully run a GGUF model of your choice using vLLM with `vllama`, please consider:

*   **Opening an issue ticket** with the command used to help us add broader support.
*   **Submitting a pull request** with your changes—even better!

## Client Integration Notes

If you are using clients like Roo Code or Cline, it is recommended to adjust the maximum context window length in your client's settings to match your available VRAM. Additionally, condensing at 80% of the context window size is recommended for optimal performance and to prevent truncation.

### Context Information Endpoints

```bash
# Get model list with accurate context windows
curl http://localhost:11435/v1/models

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
- **`openAiCustomModelInfo.contextWindow`**: Set this to match the context window reported by vllama for your model (e.g., via `curl http://localhost:11435/v1/models`).
- **Model ID**: Use the exact model name from `ollama list`
- **Base URL**: Point to your vllama server with `/v1` suffix

### Why OpenAI Provider Needs Manual Configuration

Unlike LiteLLM providers, OpenAI providers in Roo Code use **user-configured model information** rather than fetching it from the API. This means:

- **OpenAI Provider**: Uses `openAiCustomModelInfo.contextWindow` from your settings

Both approaches work, but OpenAI provider gives you explicit control over the reported context window.
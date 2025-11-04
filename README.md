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

## Table of Contents

- [Quick Start](#quick-start)
  - [Arch Linux Users (AUR)](#arch-linux-users-aur)
- [Updates](#updates)
- [Logging](#logging)
- [System Service Setup (Ubuntu/Debian)](#system-service-setup-ubuntudebian)
- [Windows Users](#windows-users)
- [Vision](#vision)
  - [Proven to Work Models](#proven-to-work-models)
  - [GGUF Model Compatibility](#gguf-model-compatibility)
- [Client Integration Notes](#client-integration-notes)

## Quick Start

> [!TIP]
> ### Arch Linux Users (AUR)
> The fastest way to get started on Arch Linux is by using the [AUR package](https://aur.archlinux.org/packages/vllama). This will install `vllama` as a systemd service.
>
> ```bash
> # Install from AUR (e.g., with yay or paru)
> yay -S vllama
>
> # Enable and start the core services
> sudo systemctl enable --now ollama.service
> sudo systemctl enable --now vllama.service
>
> # Pull your desired models using Ollama
> ollama pull tom_himanen/deepseek-r1-roo-cline-tools:14b
> ollama pull huihui_ai/devstral-abliterated:latest
>
> # Restart vllama to discover the new models
> # vllama discovers Ollama models on startup.
> sudo systemctl restart vllama.service
>
> # Your server is now running with the new models!
> curl http://localhost:11435/v1/models
> ```

## Updates

*   **Nov 4, 2025:** Support for the `deepseek-r1` architecture has been added! 🧠 This allows models like `huihui_ai/deepseek-r1-abliterated:14b` to be used with `vllama`.
*   **Nov 3, 2025:** `vllama` is alive! 🎉 Devstral models are confirmed to work flawlessly, bringing high-speed local inference to the community. 🚀

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

### Proven to Work Models

While many GGUF models may work, the following have been tested and are confirmed to run correctly with `vllama`.

| Model Family   | vLLM Architecture | Notes                               |
|----------------|-------------------|-------------------------------------|
| Devstral       | `mistral`         | Works exceptionally well out of the box. |
| DeepSeek-R1    | `deepseek_r1`     |                                     |

### GGUF Model Compatibility

Ollama uses GGUF models and they all are listed by default in vllama. However, all architectures might not work because vLLM's GGUF support is still experimental. While Devstral models are proven to work, other GGUF models might not function as expected. If you successfully run a GGUF model of your choice using vLLM with `vllama`, please consider:

*   **Opening an issue ticket** with the command used to help us add broader support.
*   **Submitting a pull request** with your changes—even better!

## Client Integration Notes

If you are using clients like Roo Code or Cline, it is recommended to adjust the maximum context window length in your client's settings to match your available VRAM. Additionally, condensing at 80% of the context window size is recommended for optimal performance and to prevent truncation.

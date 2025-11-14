# vllama: Your High-Performance Local AI Coding Assistant

**vllama is a free, open-source tool that transforms your machine into a powerful offline AI for coding, programming, and more. It serves as the best local code generation tool for developers who value privacy, speed, and customization.**

vllama is a hybrid server that combines Ollama's easy model management with vLLM's fast GPU inference. This creates an OpenAI-compatible API for optimized performance, making it an excellent choice for anyone looking to set up a **local AI coding assistant**. It runs on port 11435 as a faster alternative to Ollama (port 11434), allowing you to run both simultaneously.

If you're looking for the **best local LLM for programming**, vllama is your answer. It empowers you to use local large language models for programming tasks like **code generation**, **debugging**, **code completion**, **syntax optimization**, and even **offline llm for code debugging**. It's designed for efficient **local LLM** operations and on-device AI, making it a top choice for developers seeking **free local LLM for developers** or powerful **GitHub Copilot alternatives**.

**Key Features:**
*   **On-Demand Model Loading & Unloading:** Models are loaded on-demand when a request is received and automatically unloaded after 5 minutes of inactivity, freeing up VRAM and making it a true on-demand solution.
*   **Automatic Context Length Optimization:** vllama automatically calculates and maximizes the context length based on your available VRAM, ensuring peak performance without manual tweaking.
*   **Broad Model Support:** All Ollama models are automatically discovered. While vLLM's GGUF support is experimental, many models, including top performers like **Devstral** and **DeepSeek**, are proven to work.
*   **Network-Wide Access:** Serve models to your entire local network, enabling **agents powered by local LLM** and collaborative development.
*   **Advanced Model Techniques:** Supports models using **quantization**, **distilled models for local programming**, and techniques like **model pruning** to run efficiently on your hardware.

<p align="center">
<a href="https://raw.githack.com/erkkimon/vllama/main/assets/player.html" target="_blank">
<img src="assets/ollama-vs-vllama-thumbnail.jpg" alt="Ollama vs vLLaMA Demo">
</a>
</p>

## Table of Contents

- [Quick Start](#quick-start)
  - [Arch Linux Users (AUR)](#arch-linux-users-aur)
- [Supported Models](#supported-models)
- [Integrations with Programming Agents](#integrations-with-programming-agents)
  - [Roo Code, Cline, and Goose Setup](#roo-code-cline-and-goose-setup)
  - [Other Agent Integrations](#other-agent-integrations)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
- [Updates](#updates)
- [Logging](#logging)
- [System Service Setup (Ubuntu/Debian)](#system-service-setup-ubuntudebian)
- [Windows Users](#windows-users)
- [Vision](#vision)
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
> sudo systemctl restart vllama.service
>
> # Your server is now running!
> curl http://localhost:11435/v1/models
> ```

## Supported Models

vllama can run any GGUF model available on Ollama, but compatibility ultimately depends on vLLM's support for the model architecture. The table below lists models that have been tested or are good candidates for local coding tasks. This is a great starting point for finding the **top open source LLM for coding locally**.

| Model Family | Status | Notes |
|---|---|---|
| **Devstral** | ✅ **Proven to Work** | Excellent performance for coding and general tasks. A **best local coding model**. |
| **DeepSeek-R1** | ✅ **Proven to Work** | Great for complex **programming** and following instructions. |
| **DeepSeek-V2 / V3** | ❔ Untested | Promising for **code generation** and **debugging**. |
| **Mistral / Mistral-Instruct** | ❔ Untested | Lightweight and fast, good for **code completion**. |
| **CodeLlama / CodeLlama-Instruct** | ❔ Untested | Specifically **fine-tuned** for programming tasks. |
| **Phi-3 (Mini, Small, Medium)** | ❔ Untested | Strong reasoning capabilities in a smaller package. |
| **Llama-3-Code** | ❔ Untested | A powerful contender for the **best local coding llm**. |
| **Qwen (2.5, 3, 3-VL, 3-Coder)** | ❔ Untested | Strong multilingual and coding abilities. |
| **Gemma / Gemma-2** | ❔ Untested | Google's open models, good for general purpose and coding. |
| **StarCoder / StarCoder2** | ❔ Untested | Trained on a massive corpus of code. |
| **WizardCoder** | ❔ Untested | Fine-tuned for coding proficiency. |
| **GLM / GLM-4** | ❔ Untested | Bilingual models with strong performance. |
| **Codestral** | ❔ Untested | Mistral's first code-specific model. |
| **Kimi K2** | ❔ Untested | Known for its large context window capabilities. |
| **Granite-Code** | ❔ Untested | IBM's open-source code models. |
| **CodeBERT** | ❔ Untested | An early but influential code model. |
| **Pythia-Coder** | ❔ Untested | A model for studying LLM development. |
| **Stable-Code** | ❔ Untested | From the creators of Stable Diffusion. |
| **Mistral-Nemo** | ❔ Untested | A powerful new model from Mistral. |
| **Llama-3.1** | ❔ Untested | The latest iteration of the Llama family. |
| **TabNine-Local** | ❔ Untested | Open variants of the popular code completion tool. |

## Integrations with Programming Agents

One of the most powerful uses of vllama is to serve as the brain for local programming agents. This is **how to use local llm for software development** in a modern, automated way.

### Roo Code, Cline, and Goose Setup

**Roo Code**, **Cline**, **Kilo Code** and **Goose** are powerful programming agents that can use vllama for **inference**. Since vllama provides an OpenAI-compatible API, setting them up is straightforward.

1.  **Start vllama**: Ensure your `vllama.service` is running or start it manually.
2.  **Configure the Agent**: In your agent's settings (e.g., in Roo Code's `config.toml`), point the API endpoint to vllama's address.
    *   **API URL**: `http://localhost:11435/v1`
    *   **Model Name**: Select one of the models you have pulled with Ollama (e.g., `huihui_ai/devstral-abliterated:latest`).
    *   **API Key**: You can typically leave this blank.

Now, your agent will use your local GPU for lightning-fast, private **code generation** and **debugging**.

### Other Agent Integrations

The same principle applies to most modern AI agents. vllama can serve as a local, private backend for many popular tools, making it a fantastic alternative to cloud-based services.

*   **AI Agent Frameworks**: For frameworks like **LangChain agents**, **AutoGen**, and **CrewAI**, you can configure the LLM client to point to the vllama endpoint (`http://localhost:11435/v1`). This allows you to build complex workflows that run entirely on your hardware.
*   **Interpreter-Style Agents**: Tools like **Open-Interpreter** and open-source alternatives to **Devin-AI** can be configured to use a local OpenAI-compatible endpoint, making them perfect companions for vllama.
*   **IDE Plugins & Tools**: Plugins and tools like **Aider**, **Cursor-AI local**, **Tabby-ML**, **Continue-dev**, and alternatives for **CodeWhisperer local** often support custom local endpoints. Point them to vllama to get superior performance and privacy compared to their default cloud services.
*   **Other Coding Assistants**: The OpenAI-compatible API allows vllama to be a backend for many other tools, including experimental or less common ones like **Claude Code** and **Kilo Code**.
*   **Advanced Agent Architectures**: If you are experimenting with **Reflexion agents**, **ReAct agents for coding**, or **Tree-of-Thoughts coding**, vllama provides the fast, local **inference** engine you need to power your research.

## Frequently Asked Questions (FAQ)

**Q: Why is my model's context window smaller than expected?**

**A:** vllama prioritizes speed and stability by running inference exclusively on your GPU. To prevent out-of-memory errors, it automatically calculates the maximum possible context window based on the VRAM available *after* the model is loaded. If you need a larger context window, you should try a smaller model or a version with more aggressive **quantization** (e.g., a 4-bit or 5-bit quantized model instead of a 7-bit or 8-bit one). Finding the **best quantized llm for programming tasks** often involves balancing performance with context size.

**Q: What is the best local LLM for programming?**

**A:** The "best" model depends heavily on your hardware, the specific task (e.g., **code completion** vs. complex **debugging**), and personal preference. The goal of vllama is to make it easy to experiment. We recommend starting with models proven to work well, like **Devstral** or **DeepSeek-R1**, as they offer a great balance of performance and capability. Check the [Supported Models](#supported-models) table to explore other options.

**Q: How can I use a local LLM for software development?**

**A:** Using a **local large language model for programming** offers several advantages:
1.  **Privacy**: Your code never leaves your machine.
2.  **Speed**: **Inference** is performed directly on your GPU, eliminating network latency.
3.  **Offline Access**: Continue working without an internet connection, making it a true **offline AI** solution.
4.  **Customization**: You can choose from dozens of open-source models, including **fine-tuned** or **distilled models for local programming**, to find the perfect fit for your needs.
vllama is the engine that makes this practical and efficient.

**Q: Is this a free local LLM for developers?**

**A:** Yes. vllama is an open-source tool that is completely free to use. You provide the hardware, and vllama provides the high-performance inference server. It's part of a growing ecosystem of free, open-source tools designed to democratize access to powerful AI.

## Updates

*   **Nov 4, 2025:** Support for the `deepseek-r1` architecture has been added! 🧠 This allows models like `huihui_ai/deepseek-r1-abliterated:14b` to be used with `vllama`.
*   **Nov 3, 2025:** `vllama` is alive! 🎉 Devstral models are confirmed to work flawlessly, bringing high-speed local inference to the community. 🚀

## Logging

`vllama` logs important events and errors to help with debugging and monitoring.

*   **Service (Production):** When running as a systemd service, logs are located in `/opt/vllama/logs/`.
*   **Development:** When running `vllama.py` directly from the repository, logs are located in a `logs/` directory within the project root.

To monitor the logs in real-time, you can use the `tail -f` command on the appropriate log file.

## System Service Setup (Ubuntu/Debian)

For distributions like Ubuntu or Debian, you can set up `vllama` to run as a systemd service for automatic startup.

1.  **Install Dependencies:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv git
    ```

2.  **Clone and Prepare the Repository:**
    ```bash
    git clone https://github.com/erkkimon/vllama.git /tmp/vllama
    sudo mv /tmp/vllama /opt/vllama
    cd /opt/vllama
    sudo python3 -m venv venv312
    sudo venv312/bin/pip install -r requirements.txt
    sudo chown -R $USER:$USER /opt/vllama
    ```

3.  **Configure and Enable the Service:**
    ```bash
    sudo cp /opt/vllama/vllama.service /etc/systemd/system/vllama.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now vllama.service
    sudo systemctl status vllama.service
    ```

## Windows Users

For instructions on running `vllama` on Windows, please see the **[Windows Setup Guide](./docs/windows.md)**.

## Vision

The vision for vllama is to make high-performance AI inference accessible and efficient. By integrating vLLM's advanced GPU optimizations, it addresses common pain points like slow Ollama inference while maintaining Ollama's simple workflow. This makes it an ideal solution for **local programming** and **local models powered software development**, enabling **agents powered by local LLM** to run efficiently. Whether you're looking for an OpenAI-compatible vLLM server or ways to unload vLLM models when idle, vllama aims to be the go-to tool for users wanting faster **inference** with **Ollama** and **vLLM**.

## How to contribute

All pull requests are welcome! Also, if you have succesfully run your favorite LLM in GGUF format with vLLM, please share it by creating and issue. It will help a lot integrating it into vllama!

## Client Integration Notes

If you are using clients like **Roo Code** or **Cline**, it is recommended to adjust the maximum context window length in the client's settings to match your available VRAM. Condensing at 80% of the context window size is recommended for optimal performance.

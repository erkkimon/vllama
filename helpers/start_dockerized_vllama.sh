#!/bin/bash

# This script detects the Ollama models directory and launches the vllama Docker container.

detect_ollama_models_path() {
    # Try environment variable first
    if [ -n "$OLLAMA_MODELS" ] && [ -d "$OLLAMA_MODELS" ]; then
        printf "%s" "$OLLAMA_MODELS"
        return 0
    fi
    
    # Try systemd service (if running)
    if systemctl is-active --quiet ollama 2>/dev/null; then
        local systemd_env
        systemd_env=$(systemctl show ollama.service --property=Environment 2>/dev/null | grep -o 'OLLAMA_MODELS=[^ ]*' | cut -d= -f2)
        if [ -n "$systemd_env" ] && [ -d "$systemd_env" ]; then
            printf "%s" "$systemd_env"
            return 0
        fi
    fi
    
    # Common system paths
    for path in "/var/lib/ollama/.ollama/models" "/usr/share/ollama/.ollama/models" "/usr/share/ollama/models"; do
        if [ -d "$path" ]; then
            printf "%s" "$path"
            return 0
        fi
    done
    
    # User paths
    for path in "$HOME/.ollama/models" "/root/.ollama/models"; do
        if [ -d "$path" ]; then
            printf "%s" "$path"
            return 0
        fi
    done
    
    return 1
}

# --- Main Script ---

OLLAMA_MODELS_PATH=$(detect_ollama_models_path)

if [ -z "$OLLAMA_MODELS_PATH" ]; then
    echo "Error: Could not automatically detect the Ollama models directory." >&2
    echo "Please set the OLLAMA_MODELS environment variable and try again." >&2
    exit 1
fi

echo "✅ Detected Ollama models directory: $OLLAMA_MODELS_PATH"
echo "🚀 Launching vllama container..."

# Run the Docker container in detached mode as a service
DOCKER_RUN_ARGS=(
  -d
  --gpus all
  --network host
  --name vllama-service
  --restart unless-stopped
  -v "$OLLAMA_MODELS_PATH:/var/lib/ollama/models:ro"
)

# Conditionally add the custom models volume mount
if [ -d "/opt/vllama/models" ]; then
  echo "✅ Found custom models directory at /opt/vllama/models, mounting it as read-only."
  DOCKER_RUN_ARGS+=("-v" "/opt/vllama/models:/opt/vllama/models:ro")
fi

docker run "${DOCKER_RUN_ARGS[@]}" tomhimanen/vllama:latest

# For development purposes you can use this docker run command instead
# docker run -d \
#   --gpus all \
#   --network host \
#   --name vllama-service \
#   --restart unless-stopped \
#   -v "$OLLAMA_MODELS_PATH:/var/lib/ollama/models:ro" \
#   vllama-dev

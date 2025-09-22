#!/bin/bash
# vllama wrapper - activates venv312 and runs server

VENV_PATH="/opt/vllama/venv312"
SCRIPT_PATH="/opt/vllama/vllama.py"

if [ ! -d "$VENV_PATH" ]; then
    echo "Error: vllama virtual environment not found at $VENV_PATH"
    echo "Reinstall the package: pikaur -S vllama"
    exit 1
fi

# Activate venv and run
source "$VENV_PATH/bin/activate"
exec python "$SCRIPT_PATH" "$@"
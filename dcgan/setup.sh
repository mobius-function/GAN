#!/bin/bash

echo "Setting up GAN project dependencies..."

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for package installation..."
    uv pip install -r requirements.txt
else
    echo "uv not found, using pip instead..."
    pip install -r requirements.txt
fi

echo "Setup complete!"
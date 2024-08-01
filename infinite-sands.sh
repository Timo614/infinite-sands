#!/bin/bash

# Navigate to the stable-diffusion-webui directory
cd stable-diffusion-webui

# Activate the virtual environment
source venv/bin/activate

# Run the stable-diffusion-webui in the background
HSA_OVERRIDE_GFX_VERSION=11.0.0 TORCH_COMMAND='pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm6.1' python launch.py --precision full --no-half --api

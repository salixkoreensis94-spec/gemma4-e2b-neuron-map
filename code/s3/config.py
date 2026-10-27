"""Explicit paths. Set NEURON_WORKDIR, NEURON_GGUF and optional NEURON_NVIDIA_SMI."""
import os, shutil
from pathlib import Path
ROOT=Path(os.environ.get('NEURON_WORKDIR','work')).resolve()
MODEL=Path(os.environ['NEURON_GGUF']).resolve() if 'NEURON_GGUF' in os.environ else ROOT/'inputs/model.gguf'
NVIDIA_SMI=os.environ.get('NEURON_NVIDIA_SMI') or shutil.which('nvidia-smi') or 'nvidia-smi'

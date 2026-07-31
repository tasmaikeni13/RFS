ARG BASE_IMAGE=rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.12.0@sha256:c38eeda81d85f00fbe35d3d50ce42ce59c524e87d810624f4eb5c52fddb3b9ad
FROM ${BASE_IMAGE}

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 \
    TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions \
    PYTORCH_ROCM_ARCH=gfx942 \
    HIP_VISIBLE_DEVICES=0 \
    TOKENIZERS_PARALLELISM=true

COPY pyproject.toml /tmp/rfs/pyproject.toml
RUN python3 -m pip install --no-cache-dir \
      'huggingface-hub==1.26.0' 'matplotlib==3.11.0' 'numpy==2.5.1' \
      'pandas==3.0.3' 'pyarrow==21.0.0' 'tiktoken==0.13.0' \
      'pytest==9.1.1' 'ruff==0.16.1'

# The wheel-based ROCm SDK ships the versioned HIP runtime but omits the
# development symlink expected by torch.utils.cpp_extension.
RUN ln -sf /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7 \
           /opt/venv/lib/libamdhip64.so

CMD ["bash"]

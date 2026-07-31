#!/usr/bin/env bash
set -euo pipefail

readonly IMAGE="rfs-mi300x:rocm7.14-pytorch2.12"
readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  docker build --tag "${IMAGE}" "${ROOT_DIR}"
fi

tty_args=(-i)
if [[ -t 0 && -t 1 ]]; then
  tty_args+=(-t)
fi

exec docker run --rm "${tty_args[@]}" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume "${ROOT_DIR}:/workspace" \
  --workdir /workspace \
  --env PYTORCH_ROCM_ARCH=gfx942 \
  --env TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_cache \
  "${IMAGE}" "$@"

#!/bin/bash
set -e
set -x

DIST_DIR="/__w/jax/jax/dist"
mkdir -p "$DIST_DIR"

sudo apt-get update
sudo apt-get install -y gfortran pkg-config libopenblas-dev

BAZEL_CACHE=$(bazel info output_base)
BAZEL_PYTHON=$(find "$BAZEL_CACHE/external" -path "*/bin/python3" 2>/dev/null | grep "freethreaded" | head -n 1)

if [ -z "$BAZEL_PYTHON" ]; then
    echo "ERROR: Could not find Bazel's freethreaded Python in the cache!"
    echo "Triggering Bazel to download it now..."
    bazel build --repo_env=HERMETIC_PYTHON_VERSION=3.14-ft \
        --override_repository=xla=/__w/jax/jax/xla \
        --@rules_python//python/config_settings:py_freethreaded=yes \
        @python_3_14_x86_64-unknown-linux-gnu-freethreaded//:python
    
    BAZEL_PYTHON=$(find "$BAZEL_CACHE/external" -path "*/bin/python3" 2>/dev/null | grep "freethreaded" | head -n 1)

    if [ -z "$BAZEL_PYTHON" ]; then
        BAZEL_PYTHON=$(find "$BAZEL_CACHE/external" -path "*/bin/python3.14t" 2>/dev/null | grep "freethreaded" | head -n 1)
    fi

    if [ -z "$BAZEL_PYTHON" ]; then
        echo "FATAL: Still could not find the freethreaded binary."
        exit 1
    fi
fi

$BAZEL_PYTHON -c "import sys; is_gil = sys._is_gil_enabled(); print(f'GIL is enabled: {is_gil}'); assert not is_gil, 'FATAL: The GIL is still enabled on this binary!'"

$BAZEL_PYTHON -m venv .venv-ft
source .venv-ft/bin/activate

pip install --upgrade pip
pip install meson ninja cython pybind11 wheel setuptools

rm -rf /tmp/pip-*
pip wheel numpy --pre --no-binary numpy --no-cache-dir -w "$DIST_DIR"
pip wheel scipy --pre --no-binary scipy --no-cache-dir -w "$DIST_DIR"
deactivate
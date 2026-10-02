#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0 ripgrep
npm install

if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.13 .venv
fi

source .venv/bin/activate

uv pip install -r ./requirements.txt
uv pip install git+https://github.com/pcarnah/bop_toolkit.git

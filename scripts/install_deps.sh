#!/bin/bash

uv venv .venv --python 3.13
source .venv/bin/activate

uv pip install -r ./requirements.txt
uv pip install ../bop_toolkit

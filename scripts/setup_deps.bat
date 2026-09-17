uv venv .venv --python 3.13
.\.venv\Scripts\activate

uv pip install -r .\requirements.txt
uv pip install ..\bop_toolkit

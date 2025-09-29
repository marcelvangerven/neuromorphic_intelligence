uv init --package
uv venv --python 3.13
uv add jax diffrax==0.6.2 lineax kozax evosax
source .venv/bin/activate
uv pip install -e .
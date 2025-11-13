## Workflow

# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Build & run locally
uv sync
uv build
uv pip install dist/*.whl
uv run python/python/app.py

# 3. Deploy to Modal
modal deploy modal_app.py
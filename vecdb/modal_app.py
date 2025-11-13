# modal_app.py
import modal
import os

# === 1. Secrets ===
grok_secret = modal.Secret.from_name("grok-secrets")  # GROK_API_KEY
db_secret = modal.Secret.from_name("postgres-secrets")  # DB_PASSWORD

# === 2. Docker Image ===
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "curl",
        "build-essential",
        "libssl-dev",
        "pkg-config",
        "git",
        "clang",
        "libpq-dev",
    )
    # Install uv
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "export PATH=$PATH:$HOME/.local/bin",
    )
    # Install RAPIDS cuVS (GPU)
    .run_commands(
        "conda install -c rapidsai-nightly -c conda-forge -c nvidia "
        "cuvs-cu12=25.12 raft-cu12=25.12 python=3.12 cuda-version=12.2 -y"
    )
    # Copy project + build Rust
    .workdir("/app")
    .copy_local_dir(".", ".")
    .run_commands(
        "uv sync --frozen",
        "uv build --release",
        "uv pip install dist/*.whl --no-deps",
    )
)

# === 3. Postgres with pgvector + Lantern + init.sql ===
db_image = (
    modal.Image.from_registry("supabase/postgres:16")
    .apt_install("build-essential", "git", "clang")
    .run_commands(
        "git clone https://github.com/pgvector/pgvector.git && cd pgvector && make && make install",
        "git clone https://github.com/lantern-hq/lantern.git && cd lantern/postgres && make && make install",
    )
    .copy_local_file("init.sql", "/docker-entrypoint-initdb.d/init.sql")  # Auto-run
)

db = modal.Postgres(
    name="vector-db",
    image=db_image,
    secrets=[db_secret],
    memory=(4, 16),  # GB
    cpu=2.0,
)

# === 4. App ===
app = modal.App("vector-rust-gpu", image=image)

# === 5. Web Server (GPU) ===
@app.function(
    gpu="A100",
    secrets=[grok_secret, db_secret],
    timeout=600,
    memory=(8, 32),
)
@modal.asgi_app()
def web():
    import os
    from python.app import app as fastapi_app

    # Inject DB URL
    os.environ["DB_URL"] = db.connection_string()
    os.environ["GROK_API_KEY"] = os.environ["GROK_API_KEY"]  # Already in secret

    return fastapi_app

# === 6. Optional: Health Check Function ===
@app.function(secrets=[db_secret])
def health():
    import psycopg
    with psycopg.connect(db.connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            print("DB healthy:", cur.fetchone())
    print("Modal app ready!")
    return {"status": "ok"}
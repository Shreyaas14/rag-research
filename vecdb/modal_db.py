# modal_db.py
import modal

db_image = (
    modal.Image.from_registry("supabase/postgres:16")
    .apt_install("build-essential", "git", "clang")
    .run_commands(
        "git clone https://github.com/pgvector/pgvector.git && cd pgvector && make && make install",
        "git clone https://github.com/lantern-hq/lantern.git && cd lantern/postgres && make && make install",
    )
    .copy_local_file("init.sql", "/docker-entrypoint-initdb.d/init.sql")  # This line
)

db = modal.Postgres(
    name="vector-db",
    image=db_image,
    secrets=[modal.Secret.from_name("postgres-secrets")],
    # init_sql is auto-run from /docker-entrypoint-initdb.d/
)
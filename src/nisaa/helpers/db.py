import os
from psycopg2 import pool

DATABASE_HOST=os.getenv("DATABASE_HOST")
DATABASE_PORT=os.getenv("DATABASE_PORT")
DATABASE_USER=os.getenv("DATABASE_USER")
DATABASE_PASS=os.getenv("DATABASE_PASS")
DB_NAME=os.getenv("DB_NAME")

DB_URI = f"postgresql://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}:{DATABASE_PORT}/{DB_NAME}"

if not DB_URI:
    raise ValueError("DB_URI environment variable is not set!")

_pg_pool = pool.SimpleConnectionPool(1, 10, dsn=DB_URI)

def save_message(thread_id: str, role: str, content: str):
    conn = None
    try:
        conn = _pg_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (thread_id, role, content) VALUES (%s, %s, %s)",
                (thread_id, role, content)
            )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            _pg_pool.putconn(conn)
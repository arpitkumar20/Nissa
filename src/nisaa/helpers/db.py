import os
from psycopg2 import pool

DB_URI = os.getenv("DB_URI")
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
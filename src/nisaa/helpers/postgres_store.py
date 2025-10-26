from src.nisaa.helpers.db import save_message, _pg_pool

class PostgresMemoryStore:
    def __init__(self, thread_id: str):
        self.thread_id = str(thread_id)

    def put(self, role: str, content: str):
        save_message(self.thread_id, role, content)

    def get_all(self):
        conn = None
        try:
            conn = _pg_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, role, content, created_at FROM messages WHERE thread_id=%s ORDER BY created_at ASC",
                    (self.thread_id,)
                )
                rows = cur.fetchall()
            return [{"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]} for r in rows]
        finally:
            if conn:
                _pg_pool.putconn(conn)
    
    def get_last_n_messages(self, n=20):
        """Fetch last n messages in DESC order by created_at."""
        conn = None
        try:
            conn = _pg_pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, role, content, created_at FROM messages "
                    "WHERE thread_id=%s ORDER BY created_at DESC LIMIT %s",
                    (self.thread_id, n)
                )
                rows = cur.fetchall()
            rows = rows[::-1]
            return [{"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]} for r in rows]
        finally:
            if conn:
                _pg_pool.putconn(conn)

    def search_memories(self, query: str):
        all_msgs = self.get_all()
        return [m for m in all_msgs if query.lower() in m["content"].lower()]

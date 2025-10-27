import os
import atexit
from langgraph.checkpoint.postgres import PostgresSaver

DATABASE_HOST=os.getenv("DATABASE_HOST")
DATABASE_PORT=os.getenv("DATABASE_PORT")
DATABASE_USER=os.getenv("DATABASE_USER")
DATABASE_PASS=os.getenv("DATABASE_PASS")
DB_NAME=os.getenv("DB_NAME")

DB_URI = f"postgresql://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}:{DATABASE_PORT}/{DB_NAME}"
if not DB_URI:
    raise ValueError("DB_URI environment variable is not set!")

_checkpointer_cm = PostgresSaver.from_conn_string(DB_URI)
checkpointer = _checkpointer_cm.__enter__()
checkpointer.setup()

def _close_checkpointer():
    try:
        _checkpointer_cm.__exit__(None, None, None)
    except Exception:
        pass
atexit.register(_close_checkpointer)

import os
import atexit
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = os.environ.get("DB_URI")

_checkpointer_cm = PostgresSaver.from_conn_string(DB_URI)
checkpointer = _checkpointer_cm.__enter__()
checkpointer.setup()

def _close_checkpointer():
    try:
        _checkpointer_cm.__exit__(None, None, None)
    except Exception:
        pass
atexit.register(_close_checkpointer)

import os
import atexit
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()
DB_URI = os.getenv("DB_URI")
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

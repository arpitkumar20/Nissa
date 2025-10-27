import logging
import warnings
import logging.handlers
from pathlib import Path

warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
logging.getLogger("pydantic").setLevel(logging.ERROR)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"
MAX_BYTES = 10 * 1024 * 1024 
BACKUP_COUNT = 5  

logger = logging.getLogger("nissa_chat_bot")
logger.setLevel(logging.DEBUG) 

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)

file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_logger(name=None):
    return logger if name is None else logger.getChild(name)
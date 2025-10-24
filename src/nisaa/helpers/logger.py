import logging
import logging.handlers
from pathlib import Path

# ===== CONFIGURATION =====
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
BACKUP_COUNT = 5  # Keep last 5 log files

# ===== LOGGER SETUP =====
logger = logging.getLogger("nissa_chat_bot")
logger.setLevel(logging.DEBUG)  # Log all levels DEBUG and above

# Formatter with timestamp, module, level, and message
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console handler (prints to stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only info and above to console
console_handler.setFormatter(formatter)

# Rotating file handler (writes to file)
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)  # Log everything to file
file_handler.setFormatter(formatter)

# Optional: SMTP handler for critical errors (uncomment and configure if needed)
# mail_handler = logging.handlers.SMTPHandler(
#     mailhost=("smtp.example.com", 587),
#     fromaddr="alert@example.com",
#     toaddrs=["admin@example.com"],
#     subject="Critical Error in Application",
#     credentials=("username", "password"),
#     secure=(),
# )
# mail_handler.setLevel(logging.CRITICAL)
# mail_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
# logger.addHandler(mail_handler)  # Uncomment if using email alerts

# ===== HELPER FUNCTIONS (OPTIONAL) =====
def get_logger(name=None):
    """
    Get a module-specific logger if needed.
    """
    return logger if name is None else logger.getChild(name)
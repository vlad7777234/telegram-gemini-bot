import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
PORT = int(os.getenv("PORT", "10000"))
PING_INTERVAL_MINUTES = int(os.getenv("PING_INTERVAL_MINUTES", "9"))
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "20"))

# Секретный пароль для доступа к админ-панели (в Telegram и на Web-странице)
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "admin777").strip()

# Telegram ID главного администратора (опционально, можно также авторизоваться паролем)
ADMIN_ID = int(os.getenv("ADMIN_ID", "0")) if os.getenv("ADMIN_ID", "").strip().isdigit() else 0

DEFAULT_SYSTEM_PROMPT = (
    "Ты — умный, дружелюбный и экспертный AI-ассистент в Telegram на базе Google Gemini. "
    "Твои ответы должны быть четкими, информативными, логичными и красиво оформленными. "
    "Если пользователь присылает код, анализируй его внимательно и форматируй с указанием языка. "
    "Отвечай на языке запроса пользователя (по умолчанию на русском)."
)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()

def validate_config():
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if missing:
        raise ValueError(
            f"Отсутствуют обязательные переменные окружения: {', '.join(missing)}."
        )

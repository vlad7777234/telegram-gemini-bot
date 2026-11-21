import os
from dotenv import load_dotenv

# Загружаем переменные из .env при локальном запуске
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Render автоматически задает RENDER_EXTERNAL_URL и PORT
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
PORT = int(os.getenv("PORT", "10000"))

# Интервал пинга в минутах (по умолчанию 9 минут)
PING_INTERVAL_MINUTES = int(os.getenv("PING_INTERVAL_MINUTES", "9"))

# Максимальное количество сообщений в истории диалога для каждого пользователя
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "20"))

# Системный промпт по умолчанию
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
            f"Отсутствуют обязательные переменные окружения: {', '.join(missing)}. "
            f"Пожалуйста, укажите их в файле .env или в панели управления Render!"
        )

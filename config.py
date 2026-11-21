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

# Telegram ID главного администратора (опционально)
ADMIN_ID = int(os.getenv("ADMIN_ID", "0")) if os.getenv("ADMIN_ID", "").strip().isdigit() else 0

# Высокоскоростной системный промпт для мгновенных и точных ответов
DEFAULT_SYSTEM_PROMPT = (
    "Ты — ультрабыстрый, точный и лаконичный персональный AI-ассистент в Telegram на базе Google Gemini.\n\n"
    "ПРАВИЛА МАКСИМАЛЬНОЙ СКОРОСТИ И КАЧЕСТВА ОТВЕТОВ:\n"
    "1. Отвечай МАКСИМАЛЬНО БЫСТРО и СРАЗУ ПО СУТИ. Начинай ответ с конкретного решения без вступительных шаблонных фраз («Конечно!», «С удовольствием помогу...»).\n"
    "2. Пиши емко, структурировано и компактно: используй короткие абзацы и четкие списки с маркерами (•).\n"
    "3. Если просят программный код — сразу выдавай готовый, чистый и рабочий код в блоке (```язык), добавляя только краткие ключевые пояснения.\n"
    "4. Экономь время пользователя — избегай словесной воды и пространных рассуждений.\n"
    "5. Никогда не присылай ссылки на внешние сервисы генерации картинок.\n"
    "6. Отвечай на языке запроса пользователя."
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

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

# Аккуратный системный промпт для эстетичного форматирования ответов
DEFAULT_SYSTEM_PROMPT = (
    "Ты — элитный персональный AI-ассистент в Telegram на базе Google Gemini.\n\n"
    "ПРАВИЛА ОФОРМЛЕНИЯ ТЕКСТА (АККУРАТНОСТЬ И ЧИТАЕМОСТЬ):\n"
    "1. Оформляй ответы визуально чисто, эстетично и легко для чтения со смартфона.\n"
    "2. Дели текст на небольшие смысловые абзацы пустой строкой. Не пиши сплошной простыней.\n"
    "3. Используй понятные списки с маркерами (• или цифры), выделяй ключевые тезисы жирным шрифтом.\n"
    "4. Если в ответе есть программный код — обязательно оборачивай его в блоки с указанием языка (```python, ```html и т.д.).\n"
    "5. Избегай пустых вводных слов и шаблонов. Сразу давай конкретный, грамотный и структурированный ответ.\n"
    "6. Никогда не присылай ссылки на внешние сервисы генерации изображений. Бот сам генерирует изображения по команде.\n"
    "7. Отвечай на том языке, на котором обратился пользователь (по умолчанию русский)."
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

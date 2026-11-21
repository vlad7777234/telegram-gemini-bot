import asyncio
import logging
import sys
import time
from datetime import datetime
from aiohttp import web, ClientSession, ClientTimeout
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.enums import ParseMode, ChatAction
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramAPIError

import google.generativeai as genai

import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("telegram_gemini_bot")

# Время запуска для отслеживания аптайма
START_TIME = time.time()

# Инициализация Gemini
config.validate_config()
genai.configure(api_key=config.GEMINI_API_KEY)

generation_config = genai.types.GenerationConfig(
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_output_tokens=4096,
)

gemini_model = genai.GenerativeModel(
    model_name=config.GEMINI_MODEL,
    system_instruction=config.SYSTEM_PROMPT,
    generation_config=generation_config,
)

# Хранилище контекста диалогов: user_id -> ChatSession
user_chats: dict[int, genai.ChatSession] = {}


def get_or_create_chat(user_id: int) -> genai.ChatSession:
    """Возвращает существующую или создает новую сессию чата для пользователя."""
    if user_id not in user_chats:
        user_chats[user_id] = gemini_model.start_chat(history=[])
    return user_chats[user_id]


def reset_user_chat(user_id: int):
    """Сбрасывает историю диалога с пользователем."""
    if user_id in user_chats:
        del user_chats[user_id]


# Инициализация бота и диспетчера
bot = Bot(
    token=config.TELEGRAM_BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
)
dp = Dispatcher()


def split_text(text: str, max_chunk_size: int = 4000) -> list[str]:
    """Разбивает длинный текст на части, чтобы не превысить лимит Telegram (4096 символов)."""
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_chunk_size:
            chunks.append(text)
            break

        # Пытаемся разбить по переносу строки
        split_index = text.rfind("\n", 0, max_chunk_size)
        if split_index == -1:
            # Если нет переноса строки, разбиваем по пробелу
            split_index = text.rfind(" ", 0, max_chunk_size)
        if split_index == -1:
            # Если нет пробелов, жестко режем по лимиту
            split_index = max_chunk_size

        chunks.append(text[:split_index].strip())
        text = text[split_index:].strip()

    return chunks


async def safe_send_message(chat_id: int, text: str):
    """Отправляет сообщение, пытаясь использовать Markdown. При ошибке синтаксиса переключается на обычный текст."""
    chunks = split_text(text)
    for chunk in chunks:
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN)
        except TelegramAPIError:
            # Если Markdown содержит незакрытые спецсимволы, отправляем чистым текстом
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=None)


# --- Хэндлеры команд Telegram ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    reset_user_chat(message.from_user.id)
    welcome_text = (
        f"👋 *Привет, {message.from_user.first_name}!*\n\n"
        f"Я бот на базе передовой модели *Google Gemini* ({config.GEMINI_MODEL}), "
        f"работающий на серверах Render 24/7.\n\n"
        f"💡 *Возможности:*\n"
        f"• Отвечаю на любые вопросы и помогаю с задачами\n"
        f"• Пишу, объясняю и проверяю программный код\n"
        f"• Анализирую текст и генерирую идеи\n"
        f"• Помню контекст нашей текущей беседы\n\n"
        f"📋 *Команды:*\n"
        f"/reset или /clear — сбросить память диалога и начать сначала\n"
        f"/help — справка по возможностям\n"
        f"/ping — проверить доступность бота и сервера\n\n"
        f"Просто напиши мне любой вопрос!"
    )
    await safe_send_message(message.chat.id, welcome_text)


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_text = (
        "🤖 *Как общаться с ботом:*\n\n"
        "1. Просто пишите вопросы обычным языком в этот чат.\n"
        "2. Бот помнит контекст диалога. Чтобы начать разговор на новую тему с чистого листа, введите `/reset`.\n"
        "3. Если бот долго думает над сложной задачей, подождите — для объемных ответов требуется пара секунд.\n"
        "4. Для проверки работоспособности введите `/ping`.\n"
    )
    await safe_send_message(message.chat.id, help_text)


@dp.message(Command("reset", "clear"))
async def cmd_reset(message: types.Message):
    reset_user_chat(message.from_user.id)
    await safe_send_message(
        message.chat.id,
        "🧹 *Память диалога очищена.* Теперь мы начинаем разговор с чистого листа! О чем хочешь поговорить?"
    )


@dp.message(Command("ping"))
async def cmd_ping(message: types.Message):
    uptime_sec = int(time.time() - START_TIME)
    hours, remainder = divmod(uptime_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}ч {minutes}м {seconds}с"
    
    status_text = (
        "🏓 *Pong!*\n\n"
        f"• *Сервер:* Render\n"
        f"• *Статус:* Активен (24/7)\n"
        f"• *Аптайм:* {uptime_str}\n"
        f"• *Модель Gemini:* `{config.GEMINI_MODEL}`\n"
        f"• *Интервал пинга:* каждые {config.PING_INTERVAL_MINUTES} мин.\n"
        f"• *URL:* `{config.RENDER_EXTERNAL_URL or 'Не задан (локальный режим)'}`"
    )
    await safe_send_message(message.chat.id, status_text)


@dp.message(F.text)
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    user_prompt = message.text

    # Показываем статус "печатает..." в Telegram
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    chat_session = get_or_create_chat(user_id)

    try:
        # Отправляем запрос в Gemini через сессию с историей
        # Выполняем в executor, так как библиотека синхронная
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: chat_session.send_message(user_prompt)
        )

        try:
            answer = response.text.strip()
        except Exception:
            answer = "⚠️ Ответ не может быть отображен (сработали фильтры безопасности Gemini). Попробуйте перефразировать вопрос."

        if not answer:
            answer = "Извините, не удалось сформировать ответ. Попробуйте перефразировать вопрос."

        await safe_send_message(message.chat.id, answer)

    except Exception as e:
        logger.error(f"Ошибка при обращении к Gemini: {e}", exc_info=True)
        # Если ошибка связана с переполнением контекста или фильтрами
        error_msg = (
            "⚠️ Произошла ошибка при обработке запроса.\n\n"
            f"Технические детали: `{str(e)[:200]}`\n\n"
            "Если ошибка повторяется, попробуйте сбросить контекст командой /reset."
        )
        await safe_send_message(message.chat.id, error_msg)


# --- Веб-сервер aiohttp (для Render и проверки здоровья) ---

async def handle_http_root(request: web.Request) -> web.Response:
    uptime_sec = int(time.time() - START_TIME)
    return web.Response(
        text=(
            "Telegram Gemini Bot is running 24/7!\n"
            f"Uptime: {uptime_sec} seconds\n"
            f"Model: {config.GEMINI_MODEL}\n"
            f"Time: {datetime.utcnow().isoformat()} UTC\n"
        ),
        content_type="text/plain",
    )


async def handle_http_ping(request: web.Request) -> web.Response:
    uptime_sec = int(time.time() - START_TIME)
    return web.json_response({
        "status": "healthy",
        "service": "telegram-gemini-bot",
        "uptime_seconds": uptime_sec,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


def create_web_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_http_root)
    app.router.add_get("/ping", handle_http_ping)
    app.router.add_get("/health", handle_http_ping)
    return app


# --- Фоновая задача: самопинг каждые 9 минут ---

async def keep_alive_task():
    """Каждые N минут (по умолчанию 9) отправляет HTTP-запрос на публичный URL приложения на Render,
    чтобы бесплатный инстанс Render не засыпал из-за отсутствия входящих запросов.
    """
    interval_seconds = config.PING_INTERVAL_MINUTES * 60
    url = config.RENDER_EXTERNAL_URL

    if not url:
        logger.warning(
            "⚠️ RENDER_EXTERNAL_URL не задан! Самопинг отключен. "
            "На Render задайте эту переменную (например, https://your-bot.onrender.com) "
            "или настройте внешний пинг через cron-job.org / UptimeRobot."
        )
        return

    ping_endpoint = f"{url.rstrip('/')}/ping"
    logger.info(f"🚀 Фоновый пинг активирован: {ping_endpoint} каждые {config.PING_INTERVAL_MINUTES} минут.")

    # Небольшая задержка перед первым пингом после старта сервера
    await asyncio.sleep(45)

    timeout = ClientTimeout(total=20)
    async with ClientSession(timeout=timeout) as session:
        while True:
            try:
                now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                async with session.get(ping_endpoint) as resp:
                    if resp.status == 200:
                        logger.info(f"✅ [{now_str}] Успешный пинг {ping_endpoint} (статус 200 OK)")
                    else:
                        logger.warning(f"⚠️ [{now_str}] Пинг вернул статус {resp.status}")
            except Exception as e:
                logger.error(f"❌ Ошибка при выполнении пинга: {e}")

            await asyncio.sleep(interval_seconds)


# --- Главная функция запуска ---

async def main():
    logger.info("Запуск Telegram Gemini бота...")

    # 1. Запуск веб-сервера aiohttp на порту Render
    web_app = create_web_app()
    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=config.PORT)
    await site.start()
    logger.info(f"🌐 HTTP сервер успешно слушает на 0.0.0.0:{config.PORT}")

    # 2. Запуск фоновой задачи самопинга
    ping_bg_task = asyncio.create_task(keep_alive_task())

    # 3. Запуск polling Telegram бота
    try:
        logger.info("🤖 Запуск Long Polling для Telegram...")
        # Удаляем вебхук на случай, если он был установлен ранее
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        # Корректное завершение
        logger.info("Остановка сервисов...")
        ping_bg_task.cancel()
        await runner.cleanup()
        await bot.session.close()
        logger.info("Бот успешно остановлен.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Программа завершена пользователем.")

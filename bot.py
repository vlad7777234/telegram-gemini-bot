import asyncio
import io
import logging
import random
import sys
import time
import urllib.parse
from datetime import datetime
from aiohttp import web, ClientSession, ClientTimeout
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.enums import ParseMode, ChatAction
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BufferedInputFile

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

        split_index = text.rfind("\n", 0, max_chunk_size)
        if split_index == -1:
            split_index = text.rfind(" ", 0, max_chunk_size)
        if split_index == -1:
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
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=None)


# --- Генерация изображений ---

async def enhance_image_prompt(user_prompt: str) -> str:
    """Улучшает промпт пользователя и переводит его на английский для лучшей детализации."""
    try:
        enhancer_instruction = (
            "You are a master AI prompt engineer for high-resolution photo and art generation. "
            "Convert the user's description into a vivid, visually rich, detailed prompt in English. "
            "Include artistic style, lighting, depth, mood and composition details. "
            "Output ONLY the prompt in English without conversational text, explanations, or quotes."
        )
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_model.generate_content(
                f"{enhancer_instruction}\n\nUser request: {user_prompt}"
            )
        )
        enhanced = response.text.strip()
        if enhanced and len(enhanced) > 5:
            return enhanced
    except Exception as e:
        logger.warning(f"Не удалось улучшить промпт через Gemini: {e}")
    return user_prompt


async def fetch_image(prompt: str) -> bytes | None:
    """Скачивает сгенерированное изображение по текстовому описанию."""
    encoded_prompt = urllib.parse.quote(prompt)
    seed = random.randint(1, 99999999)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={seed}&model=flux"

    timeout = ClientTimeout(total=50)
    async with ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.read()
                logger.error(f"Image generation endpoint returned status {resp.status}")
        except Exception as e:
            logger.error(f"Image fetch failed: {e}")
    return None


async def handle_generate_image(message: types.Message, prompt: str):
    """Основной обработчик создания изображения."""
    if not prompt:
        await safe_send_message(
            message.chat.id,
            "🎨 *Как генерировать картинки:*\n\n"
            "Напишите команду с описанием:\n"
            "`/image футуристический неоновый город под дождем`\n\n"
            "Или просто напишите сообщением:\n"
            "«*Нарисуй кота-самурая на фоне заката*»"
        )
        return

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.UPLOAD_PHOTO)
    status_msg = await bot.send_message(
        chat_id=message.chat.id,
        text="🎨 *Генерирую изображение...* Пожалуйста, подождите несколько секунд."
    )

    try:
        # Улучшаем промпт через Gemini
        enhanced_prompt = await enhance_image_prompt(prompt)
        logger.info(f"Image generation: '{prompt}' -> enhanced: '{enhanced_prompt[:80]}...'")

        # Получаем байты изображения
        image_bytes = await fetch_image(enhanced_prompt)

        if image_bytes and len(image_bytes) > 1000:
            caption = (
                f"🎨 *Запрос:* {prompt}\n\n"
                f"✨ *Детализация:* _{enhanced_prompt[:250]}..._\n"
                f"🤖 _Сгенерировано: Gemini + Flux (1024x1024)_"
            )
            photo_file = BufferedInputFile(image_bytes, filename="artwork.jpg")
            try:
                await bot.send_photo(
                    chat_id=message.chat.id,
                    photo=photo_file,
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN
                )
            except TelegramAPIError:
                await bot.send_photo(
                    chat_id=message.chat.id,
                    photo=photo_file,
                    caption=f"🎨 Запрос: {prompt}"
                )

            # Удаляем временное сообщение
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=status_msg.message_id)
            except Exception:
                pass
        else:
            await status_msg.edit_text("❌ Не удалось создать изображение. Попробуйте изменить описание.")
    except Exception as e:
        logger.error(f"Ошибка при создании изображения: {e}", exc_info=True)
        await status_msg.edit_text("⚠️ Произошла ошибка при генерации картинки. Попробуйте еще раз.")


# --- Хэндлеры команд Telegram ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    reset_user_chat(message.from_user.id)
    welcome_text = (
        f"👋 *Привет, {message.from_user.first_name}!*\n\n"
        f"Я бот с искусственным интеллектом на базе *Google Gemini* ({config.GEMINI_MODEL}), "
        f"работающий на серверах Render 24/7.\n\n"
        f"💡 *Возможности:*\n"
        f"• 💬 Ответы на любые вопросы и помощь в задачах\n"
        f"• 💻 Написание и проверка программного кода\n"
        f"• 🎨 *Генерация изображений* в высоком разрешении\n"
        f"• 🧠 Память контекста нашей беседы\n\n"
        f"📋 *Команды:*\n"
        f"/image <описание> — создать картинку (или просто напишите «Нарисуй ...»)\n"
        f"/reset или /clear — очистить память диалога\n"
        f"/ping — проверить доступность сервера\n"
        f"/help — подробная справка\n\n"
        f"Задайте мне вопрос или попросите что-нибудь нарисовать!"
    )
    await safe_send_message(message.chat.id, welcome_text)


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_text = (
        "🤖 *Справка по боту:*\n\n"
        "1. *Текстовые вопросы:* пишите обычным текстом, бот помнит контекст разговора.\n"
        "2. *Генерация изображений:* напишите `/image <описание>` или начните фразу со слов «*Нарисуй...*» / «*Сгенерируй картинку...*».\n"
        "3. *Сброс памяти:* используйте команду `/reset`, чтобы начать диалог с чистого листа.\n"
        "4. *Статус:* команда `/ping` покажет статус сервера и аптайм.\n"
    )
    await safe_send_message(message.chat.id, help_text)


@dp.message(Command("image", "img", "draw", "art"))
async def cmd_image(message: types.Message):
    parts = message.text.split(maxsplit=1)
    prompt = parts[1].strip() if len(parts) > 1 else ""
    await handle_generate_image(message, prompt)


@dp.message(Command("reset", "clear"))
async def cmd_reset(message: types.Message):
    reset_user_chat(message.from_user.id)
    await safe_send_message(
        message.chat.id,
        "🧹 *Память диалога очищена.* Теперь мы начинаем разговор с чистого листа! О чем хотите поговорить?"
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
        f"• *Генерация фото:* Включена (Flux/1024x1024)\n"
        f"• *Интервал автопинга:* каждые {config.PING_INTERVAL_MINUTES} мин.\n"
        f"• *URL:* `{config.RENDER_EXTERNAL_URL or 'Не задан'}`"
    )
    await safe_send_message(message.chat.id, status_text)


IMAGE_TRIGGERS = (
    "нарисуй ",
    "нарисуй:",
    "сгенерируй картинку ",
    "сгенерируй изображение ",
    "сделай картинку ",
    "создай картинку ",
    "нарисуй мне ",
    "изобрази ",
    "draw ",
    "generate image ",
)


@dp.message(F.text)
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    user_prompt = message.text

    # Проверяем, не является ли запрос просьбой сгенерировать картинку
    lower_text = user_prompt.lower().strip()
    for trigger in IMAGE_TRIGGERS:
        if lower_text.startswith(trigger):
            img_prompt = user_prompt[len(trigger):].strip()
            await handle_generate_image(message, img_prompt)
            return

    # Показываем статус "печатает..." в Telegram
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    chat_session = get_or_create_chat(user_id)

    try:
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
            f"Image Gen: Enabled\n"
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
        "image_generation": "enabled",
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
    interval_seconds = config.PING_INTERVAL_MINUTES * 60
    url = config.RENDER_EXTERNAL_URL

    if not url:
        logger.warning("⚠️ RENDER_EXTERNAL_URL не задан! Самопинг отключен.")
        return

    ping_endpoint = f"{url.rstrip('/')}/ping"
    logger.info(f"🚀 Фоновый пинг активирован: {ping_endpoint} каждые {config.PING_INTERVAL_MINUTES} минут.")

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

    web_app = create_web_app()
    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=config.PORT)
    await site.start()
    logger.info(f"🌐 HTTP сервер успешно слушает на 0.0.0.0:{config.PORT}")

    ping_bg_task = asyncio.create_task(keep_alive_task())

    try:
        logger.info("🤖 Запуск Long Polling для Telegram...")
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
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

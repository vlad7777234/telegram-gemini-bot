import asyncio
import io
import html
import logging
import random
import sys
import time
import urllib.parse
from collections import deque
from datetime import datetime
from aiohttp import web, ClientSession, ClientTimeout
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.enums import ParseMode, ChatAction
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramAPIError
from aiogram.types import (
    BufferedInputFile,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
)

import google.generativeai as genai

import config

# Настройка системного логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("telegram_gemini_bot")

START_TIME = time.time()


# --- Расширенная система сбора пользовательских логов ---

class LogManager:
    """Управляет записью всех действий пользователей: вопросов, поиска, фото, видео и артов."""
    def __init__(self, max_records=1000):
        self.logs = deque(maxlen=max_records)
        self.total_messages = 0
        self.total_images = 0
        self.total_photos = 0
        self.total_videos = 0
        self.total_pings = 0
        self.total_errors = 0
        self.unique_users = {}  # user_id -> dict

    def track_user(self, user: types.User):
        if not user:
            return
        uid = user.id
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        if uid not in self.unique_users:
            self.unique_users[uid] = {
                "id": uid,
                "first_name": user.first_name or "",
                "username": user.username or "",
                "msg_count": 1,
                "last_seen": now_str,
            }
        else:
            self.unique_users[uid]["msg_count"] += 1
            self.unique_users[uid]["last_seen"] = now_str
            if user.first_name:
                self.unique_users[uid]["first_name"] = user.first_name
            if user.username:
                self.unique_users[uid]["username"] = user.username

    def add_user_activity(self, user: types.User, event_type: str, content: str, bot_reply: str = "", status: str = "OK"):
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        uname = f"@{user.username}" if user.username else "нет username"
        user_str = f"{user.first_name} ({uname}, ID: `{user.id}`)"
        self.track_user(user)

        record = {
            "timestamp": now_str,
            "level": "INFO" if status == "OK" else "ERROR",
            "event": event_type.upper(),  # ТЕКСТ, ПОИСК, ФОТО, ВИДЕО, АРТ, ГОЛОС, ДОКУМЕНТ
            "user_str": user_str,
            "user_id": user.id,
            "first_name": user.first_name or "",
            "username": user.username or "",
            "content": content,
            "bot_reply": bot_reply,
            "status": status,
        }
        self.logs.appendleft(record)

    def add_system_log(self, level: str, event: str, details: str, status: str = "OK"):
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        record = {
            "timestamp": now_str,
            "level": level.upper(),
            "event": event.upper(),
            "user_str": "⚙️ Сервер (Автопинг)",
            "user_id": 0,
            "first_name": "Система",
            "username": "",
            "content": details,
            "bot_reply": "",
            "status": status,
        }
        self.logs.appendleft(record)

    def get_recent(self, count=15, media_only=False, errors_only=False):
        res = []
        for item in self.logs:
            if errors_only and item["level"] != "ERROR":
                continue
            if media_only and item["event"] not in ("ФОТО", "ВИДЕО", "АРТ", "ГОЛОС", "ДОКУМЕНТ"):
                continue
            res.append(item)
            if len(res) >= count:
                break
        return res

    def clear(self):
        self.logs.clear()


log_mgr = LogManager()


# --- Инициализация Gemini ---
config.validate_config()
genai.configure(api_key=config.GEMINI_API_KEY)

# Режимы общения (Personas)
MODES = {
    "standard": {
        "title": "🤖 Стандартный",
        "description": "Универсальный помощник: баланс точности, дружелюбия и развернутых ответов.",
        "prompt": (
            "Ты — умный, дружелюбный и экспертный AI-ассистент в Telegram на базе Google Gemini. "
            "Отвечай структурированно, информативно и вежливо на языке запроса."
        ),
    },
    "coder": {
        "title": "💻 Программист (Senior Dev)",
        "description": "Фокус на качественный код, архитектуру, алгоритмы и поиск багов.",
        "prompt": (
            "Ты — Senior Software Engineer и архитектор. "
            "Пиши чистый, оптимизированный и безопасный код. Комментируй ключевые решения. "
            "Оформляй код строго в блоки с указанием языка (```python, ```typescript и т.д.)."
        ),
    },
    "creative": {
        "title": "✍️ Писатель и Креатив",
        "description": "Генерация идей, статей, постов для соцсетей, историй и сценариев.",
        "prompt": (
            "Ты — опытный писатель, копирайтер и креативный сценарист. "
            "Пиши живо, увлекательно, используй метафоры, богатый словарный запас и вдохновляющий стиль."
        ),
    },
    "concise": {
        "title": "⚡ Краткий и точный",
        "description": "Только суть, факты и списки, без вступительных фраз и воды.",
        "prompt": (
            "Ты — ассистент с максимальной емкостью ответов. "
            "Отвечай предельно сжато: только тезисы, формулы или списки. Никакой воды, приветствий и прощаний."
        ),
    },
    "tutor": {
        "title": "🎓 Репетитор (Объяснятор)",
        "description": "Объясняет сложнейшие концепции простыми словами с наглядными аналогиями.",
        "prompt": (
            "Ты — лучший преподаватель. "
            "Объясняй любые сложные научные или технические концепции просто (как для пятиклассника), "
            "используй жизненные примеры и аналогии."
        ),
    },
}

user_chats: dict[int, genai.ChatSession] = {}
user_modes: dict[int, str] = {}
authenticated_admins: set[int] = set()
if config.ADMIN_ID:
    authenticated_admins.add(config.ADMIN_ID)


def get_user_mode(user_id: int) -> str:
    return user_modes.get(user_id, "standard")


def get_or_create_chat(user_id: int) -> genai.ChatSession:
    if user_id not in user_chats:
        mode_key = get_user_mode(user_id)
        system_instruction = MODES.get(mode_key, MODES["standard"])["prompt"]
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            system_instruction=system_instruction,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=4096,
            ),
        )
        user_chats[user_id] = model.start_chat(history=[])
    return user_chats[user_id]


def reset_user_chat(user_id: int):
    if user_id in user_chats:
        del user_chats[user_id]


# Инициализация Telegram бота
bot = Bot(
    token=config.TELEGRAM_BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
)
dp = Dispatcher()


# --- Клавиатуры и Меню ---

def get_main_reply_keyboard() -> ReplyKeyboardMarkup:
    """Главная нижняя клавиатура быстрого доступа (без кнопки админки, скрыта для пользователей)."""
    kb = [
        [
            KeyboardButton(text="💬 Новый диалог"),
            KeyboardButton(text="🎨 Нарисовать арт"),
        ],
        [
            KeyboardButton(text="🎭 Сменить режим"),
            KeyboardButton(text="📊 Статус / Пинг"),
        ],
        [
            KeyboardButton(text="❓ Помощь"),
        ],
    ]
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)


def get_modes_inline_keyboard(current_mode: str) -> InlineKeyboardMarkup:
    buttons = []
    for key, data in MODES.items():
        check = "✅ " if key == current_mode else ""
        text = f"{check}{data['title']}"
        buttons.append([InlineKeyboardButton(text=text, callback_query_data=f"set_mode:{key}")])
    buttons.append([InlineKeyboardButton(text="🔙 Закрыть меню", callback_query_data="close_menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_image_styles_inline_keyboard() -> InlineKeyboardMarkup:
    styles = [
        ("📸 Фотореализм", "style:photorealistic portrait, 8k, cinematic lighting"),
        ("🎌 Аниме / Манга", "style:anime masterpiece, makoto shinkai style, vibrant"),
        ("🏙️ Киберпанк", "style:cyberpunk aesthetic, neon lights, futuristic city, octane render"),
        ("🎨 Живопись маслом", "style:oil painting, visible brushstrokes, classic artistic masterpiece"),
        ("🌌 Космос и Фэнтези", "style:epic fantasy, cosmic celestial landscape, magical atmosphere"),
        ("🎮 3D Render", "style:unreal engine 5 render, isometric 3d, ray tracing, cute"),
    ]
    rows = []
    for i in range(0, len(styles), 2):
        row = [InlineKeyboardButton(text=styles[i][0], callback_query_data=styles[i][1])]
        if i + 1 < len(styles):
            row.append(InlineKeyboardButton(text=styles[i+1][0], callback_query_data=styles[i+1][1]))
        rows.append(row)
    rows.append([InlineKeyboardButton(text="🔙 Закрыть", callback_query_data="close_menu")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_admin_inline_keyboard() -> InlineKeyboardMarkup:
    web_url = f"{config.RENDER_EXTERNAL_URL}/admin?key={config.ADMIN_SECRET}" if config.RENDER_EXTERNAL_URL else "#"
    kb = [
        [
            InlineKeyboardButton(text="🔄 Обновить логи", callback_query_data="admin:refresh"),
            InlineKeyboardButton(text="📷 Только медиа (Фото/Видео)", callback_query_data="admin:media"),
        ],
        [
            InlineKeyboardButton(text="👥 Список пользователей", callback_query_data="admin:users"),
            InlineKeyboardButton(text="⚠️ Ошибки", callback_query_data="admin:errors"),
        ],
        [
            InlineKeyboardButton(text="🌐 Открыть Веб-панель (таблица)", url=web_url),
        ],
        [
            InlineKeyboardButton(text="🧹 Очистить логи", callback_query_data="admin:clear_logs"),
            InlineKeyboardButton(text="🔒 Выйти из админки", callback_query_data="admin:logout"),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=kb)


# --- Форматирование ленты пользовательских логов для админа ---

def format_user_activity_feed(records: list) -> str:
    """Форматирует детальные логи запросов пользователей для вывода в Telegram."""
    if not records:
        return "📭 _Журнал действий пока пуст. Как только пользователи начнут писать, искать, отправлять фото или видео — их действия появятся здесь._"

    lines = ["📋 *Детальный журнал действий пользователей:*\n"]
    for r in records:
        # Иконка типа действия
        ev = r["event"]
        icon = "💬"
        if "ФОТО" in ev:
            icon = "📸"
        elif "ВИДЕО" in ev:
            icon = "🎥"
        elif "АРТ" in ev:
            icon = "🎨"
        elif "ГОЛОС" in ev:
            icon = "🎤"
        elif "ДОКУМЕНТ" in ev:
            icon = "📄"
        elif "СИСТЕМА" in ev or "PING" in ev:
            icon = "⚙️"

        entry = (
            f"👤 {r['user_str']}\n"
            f"⏰ `{r['timestamp']}`\n"
            f"{icon} *[{ev}]:* {r['content']}\n"
        )
        if r.get("bot_reply"):
            entry += f"🤖 _Ответ бота:_ {r['bot_reply'][:120]}...\n"
        entry += "─────────────────────────"
        lines.append(entry)

    return "\n".join(lines)


# --- Утилиты сообщений ---

def split_text(text: str, max_chunk_size: int = 4000) -> list[str]:
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


async def safe_send_message(chat_id: int, text: str, reply_markup=None):
    chunks = split_text(text)
    for idx, chunk in enumerate(chunks):
        markup = reply_markup if idx == len(chunks) - 1 else None
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN, reply_markup=markup)
        except TelegramAPIError:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=None, reply_markup=markup)


# --- Генерация изображений ---

async def enhance_image_prompt(user_prompt: str) -> str:
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
            lambda: genai.GenerativeModel(config.GEMINI_MODEL).generate_content(
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
    if not prompt:
        text = (
            "🎨 *Генератор изображений (1024x1024 Flux)*\n\n"
            "Вы можете написать команду:\n"
            "`/image киберпанк город под дождем`\n\n"
            "Или просто сообщением:\n"
            "«*Нарисуй кота в скафандре на Луне*»\n\n"
            "Либо выберите один из готовых стилей ниже:"
        )
        await bot.send_message(
            chat_id=message.chat.id,
            text=text,
            reply_markup=get_image_styles_inline_keyboard()
        )
        return

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.UPLOAD_PHOTO)
    status_msg = await bot.send_message(
        chat_id=message.chat.id,
        text="🎨 *Генерирую изображение...* Пожалуйста, подождите несколько секунд."
    )

    try:
        enhanced_prompt = await enhance_image_prompt(prompt)
        image_bytes = await fetch_image(enhanced_prompt)

        if image_bytes and len(image_bytes) > 1000:
            log_mgr.total_images += 1
            log_mgr.add_user_activity(
                user=message.from_user,
                event_type="АРТ",
                content=f"Промпт: «{prompt}»",
                bot_reply=f"Сгенерировано изображение (детали: {enhanced_prompt[:80]}...)",
                status="OK"
            )

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

            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=status_msg.message_id)
            except Exception:
                pass
        else:
            log_mgr.total_errors += 1
            log_mgr.add_user_activity(
                user=message.from_user,
                event_type="АРТ",
                content=f"Сбой: «{prompt}»",
                bot_reply="Не удалось сгенерировать",
                status="FAIL"
            )
            await status_msg.edit_text("❌ Не удалось создать изображение. Попробуйте изменить описание.")
    except Exception as e:
        log_mgr.total_errors += 1
        log_mgr.add_user_activity(
            user=message.from_user,
            event_type="АРТ",
            content=f"Ошибка: «{prompt}» ({str(e)[:80]})",
            status="ERROR"
        )
        logger.error(f"Ошибка при создании изображения: {e}", exc_info=True)
        await status_msg.edit_text("⚠️ Произошла ошибка при генерации картинки. Попробуйте еще раз.")


# --- Хэндлеры команд и навигации Telegram ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Приветствие бота без упоминания хостинга Render."""
    log_mgr.add_user_activity(message.from_user, "КОМАНДА", "/start (запуск бота)")
    reset_user_chat(message.from_user.id)
    cur_mode = get_user_mode(message.from_user.id)
    mode_title = MODES[cur_mode]["title"]

    welcome_text = (
        f"👋 *Привет, {message.from_user.first_name}!*\n\n"
        f"Я интеллектуальный ассистент с искусственным интеллектом на базе *Google Gemini* ({config.GEMINI_MODEL}).\n\n"
        f"🎭 *Текущий режим:* {mode_title}\n\n"
        f"💡 *Что я умею:*\n"
        f"• 💬 Отвечать на любые вопросы и искать информацию\n"
        f"• 🎨 Создавать изображения и арты по описанию\n"
        f"• 📸 Анализировать присланные вами фотографии\n"
        f"• 💻 Писать, проверять и объяснять программный код\n"
        f"• 🧠 Запоминать контекст нашей беседы\n\n"
        f"Используйте кнопки нижнего меню для быстрого взаимодействия!"
    )
    await safe_send_message(message.chat.id, welcome_text, reply_markup=get_main_reply_keyboard())


@dp.message(Command("menu"))
@dp.message(F.text == "🎭 Сменить режим")
async def cmd_menu_modes(message: types.Message):
    log_mgr.add_user_activity(message.from_user, "МЕНЮ", "Открыл выбор режимов")
    cur_mode = get_user_mode(message.from_user.id)
    text = (
        "🎭 *Интерактивный выбор режима общения:*\n\n"
        "Выберите стиль, в котором бот будет отвечать на ваши запросы:\n\n"
    )
    for k, v in MODES.items():
        mark = "👉 " if k == cur_mode else "• "
        text += f"{mark}*{v['title']}*: {v['description']}\n"

    await bot.send_message(
        chat_id=message.chat.id,
        text=text,
        reply_markup=get_modes_inline_keyboard(cur_mode),
        parse_mode=ParseMode.MARKDOWN
    )


@dp.message(F.text == "💬 Новый диалог")
@dp.message(Command("reset", "clear"))
async def cmd_reset(message: types.Message):
    log_mgr.add_user_activity(message.from_user, "КОНТЕКСТ", "Сбросил память диалога")
    reset_user_chat(message.from_user.id)
    cur_mode = get_user_mode(message.from_user.id)
    await safe_send_message(
        message.chat.id,
        f"🧹 *Память диалога очищена.*\nТекущий режим: *{MODES[cur_mode]['title']}*. Жду ваш вопрос!",
        reply_markup=get_main_reply_keyboard()
    )


@dp.message(F.text == "🎨 Нарисовать арт")
async def cmd_draw_menu(message: types.Message):
    await handle_generate_image(message, "")


@dp.message(Command("help"))
@dp.message(F.text == "❓ Помощь")
async def cmd_help(message: types.Message):
    log_mgr.add_user_activity(message.from_user, "СПРАВКА", "Запросил /help")
    help_text = (
        "📖 *Руководство по боту:*\n\n"
        "1. *Общение и поиск:* Просто пишите вопросы или что хотите найти — я подробно отвечу.\n"
        "2. *Анализ фото:* Отправьте мне любую фотографию (с вопросом или без), и я разберу, что на ней изображено.\n"
        "3. *Генерация картинок:* Нажмите «🎨 Нарисовать арт» или напишите «_Нарисуй кота-самурая_» / команду `/image <текст>`.\n"
        "4. *Режимы:* Нажмите «🎭 Сменить режим» для переключения персон (Кодер, Креатив, Репетитор и др.).\n"
        "5. *Новый разговор:* Кнопка «💬 Новый диалог» сбрасывает память контекста."
    )
    await safe_send_message(message.chat.id, help_text, reply_markup=get_main_reply_keyboard())


@dp.message(Command("ping"))
@dp.message(F.text == "📊 Статус / Пинг")
async def cmd_ping(message: types.Message):
    log_mgr.add_user_activity(message.from_user, "СТАТУС", "Запросил /ping")
    uptime_sec = int(time.time() - START_TIME)
    hours, remainder = divmod(uptime_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}ч {minutes}м {seconds}с"
    cur_mode = get_user_mode(message.from_user.id)

    status_text = (
        "🏓 *Статус системы:*\n\n"
        f"• *Статус:* Активен 🟢\n"
        f"• *Аптайм:* `{uptime_str}`\n"
        f"• *Модель Gemini:* `{config.GEMINI_MODEL}`\n"
        f"• *Ваш режим:* {MODES[cur_mode]['title']}\n"
        f"• *Генерация фото:* Активна (Flux 1024x1024)\n"
        f"• *Сообщений обработано:* {log_mgr.total_messages}\n"
        f"• *Картинок сгенерировано:* {log_mgr.total_images}\n"
        f"• *Пользователей всего:* {len(log_mgr.unique_users)}"
    )
    await safe_send_message(message.chat.id, status_text, reply_markup=get_main_reply_keyboard())


# --- Секретная Админ-панель (только по команде /admin <пароль>) ---

@dp.message(Command("admin"))
async def cmd_admin(message: types.Message):
    user = message.from_user
    parts = message.text.split(maxsplit=1)

    # Проверка пароля в команде
    if len(parts) > 1:
        entered_secret = parts[1].strip()
        if entered_secret == config.ADMIN_SECRET:
            authenticated_admins.add(user.id)
            log_mgr.add_user_activity(user, "АВТОРИЗАЦИЯ", "Администратор успешно вошел по паролю")
            await safe_send_message(message.chat.id, "🔓 *Пароль принят! Вы авторизованы как администратор.*")
        else:
            log_mgr.add_user_activity(user, "АВТОРИЗАЦИЯ", f"Попытка входа с неверным паролем: {entered_secret}", status="FAIL")
            await safe_send_message(message.chat.id, "❌ *Неверный пароль администратора.*")
            return

    # Если пользователь не авторизован
    if user.id not in authenticated_admins:
        await safe_send_message(
            message.chat.id,
            "🔒 *Доступ ограничен.*\nДля авторизации введите:\n`/admin <пароль>`\n\n_(Ваш ID: `{}` )_".format(user.id)
        )
        return

    # Авторизованному админу сразу выводим детальные логи пользователей
    recent_logs = log_mgr.get_recent(count=10)
    feed_text = format_user_activity_feed(recent_logs)

    uptime_sec = int(time.time() - START_TIME)
    hours, remainder = divmod(uptime_sec, 3600)
    minutes, seconds = divmod(remainder, 60)

    header_text = (
        f"👑 *Панель управления администратора*\n"
        f"• *Аптайм:* {hours}ч {minutes}м {seconds}с | *Пользователей:* {len(log_mgr.unique_users)}\n"
        f"• *Сообщений:* {log_mgr.total_messages} | *Картинок:* {log_mgr.total_images} | *Фото:* {log_mgr.total_photos}\n\n"
    )

    full_text = header_text + feed_text
    await safe_send_message(message.chat.id, full_text, reply_markup=get_admin_inline_keyboard())


# --- Обработчик Инлайн-кнопок (Callback Queries) ---

@dp.callback_query()
async def handle_callbacks(callback: CallbackQuery):
    data = callback.data
    user_id = callback.from_user.id
    user = callback.from_user

    # 1. Смена режима
    if data.startswith("set_mode:"):
        new_mode = data.split(":")[1]
        if new_mode in MODES:
            user_modes[user_id] = new_mode
            reset_user_chat(user_id)
            title = MODES[new_mode]["title"]
            log_mgr.add_user_activity(user, "РЕЖИМ", f"Переключил режим на: {title}")
            await callback.answer(f"Режим переключен: {title}")
            try:
                await callback.message.edit_reply_markup(reply_markup=get_modes_inline_keyboard(new_mode))
            except Exception:
                pass
            await bot.send_message(
                chat_id=user_id,
                text=f"✨ *Установлен режим:* {title}\n_{MODES[new_mode]['description']}_\n\nПамять диалога обновлена под новый стиль!",
                reply_markup=get_main_reply_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
        return

    # 2. Выбор стиля арта
    if data.startswith("style:"):
        style_prompt = data.split(":", 1)[1]
        await callback.answer("Стиль выбран!")
        await bot.send_message(
            chat_id=user_id,
            text=f"🎨 *Выбран стиль:* _{style_prompt}_\n\nТеперь напишите, что именно нарисовать в этом стиле (например: `неоновый дракон в небе`).",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    # 3. Закрытие меню
    if data == "close_menu":
        await callback.answer()
        try:
            await callback.message.delete()
        except Exception:
            pass
        return

    # 4. Админ-команды
    if data.startswith("admin:"):
        if user_id not in authenticated_admins:
            await callback.answer("⛔ Доступ запрещен!", show_alert=True)
            return

        action = data.split(":")[1]

        if action == "refresh":
            recent_logs = log_mgr.get_recent(count=12)
            feed_text = format_user_activity_feed(recent_logs)
            await callback.answer("Логи обновлены!")
            await safe_send_message(user_id, "🔄 *Обновленный журнал действий пользователей:*\n\n" + feed_text, reply_markup=get_admin_inline_keyboard())
            return

        if action == "media":
            recent_media = log_mgr.get_recent(count=15, media_only=True)
            feed_text = format_user_activity_feed(recent_media)
            await callback.answer("Медиа-логи получены!")
            await safe_send_message(user_id, "📷 *Журнал присланных медиа (Фото / Видео / Арты):*\n\n" + feed_text, reply_markup=get_admin_inline_keyboard())
            return

        if action == "errors":
            err_logs = log_mgr.get_recent(count=15, errors_only=True)
            feed_text = format_user_activity_feed(err_logs)
            await callback.answer("Ошибки получены!")
            await safe_send_message(user_id, "⚠️ *Журнал ошибок:*\n\n" + feed_text, reply_markup=get_admin_inline_keyboard())
            return

        if action == "users":
            users = list(log_mgr.unique_users.values())
            await callback.answer()
            if not users:
                await safe_send_message(user_id, "👥 *Пользователей пока нет.*")
            else:
                lines = ["👥 *Список пользователей бота:*"]
                for u in users[-25:]:
                    uname = f"@{u['username']}" if u['username'] else "нет username"
                    lines.append(f"• *{u['first_name']}* ({uname}, ID: `{u['id']}`) — запросов: {u['msg_count']}, последний: {u['last_seen']}")
                await safe_send_message(user_id, "\n".join(lines), reply_markup=get_admin_inline_keyboard())
            return

        if action == "clear_logs":
            log_mgr.clear()
            await callback.answer("Логи очищены!", show_alert=True)
            await safe_send_message(user_id, "🧹 *Журнал логов успешно очищен.*", reply_markup=get_admin_inline_keyboard())
            return

        if action == "logout":
            authenticated_admins.discard(user_id)
            await callback.answer("Вы вышли из админки.", show_alert=True)
            try:
                await callback.message.delete()
            except Exception:
                pass
            await safe_send_message(user_id, "🔒 *Вы вышли из режима администратора.*", reply_markup=get_main_reply_keyboard())
            return


# --- Обработка медиа-файлов (Фото, Видео, Голосовые, Документы) ---

@dp.message(F.photo)
async def handle_incoming_photo(message: types.Message):
    """Прием и анализ фотографий с помощью Gemini Vision + подробное логирование."""
    user = message.from_user
    caption = message.caption or ""
    log_mgr.total_photos += 1
    log_content = f"Прислал ФОТО (подпись: «{caption}»)" if caption else "Прислал ФОТО без подписи"

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        # Скачиваем самое качественное фото
        photo = message.photo[-1]
        file_io = io.BytesIO()
        await bot.download(photo, destination=file_io)
        img_bytes = file_io.getvalue()

        # Анализируем через Gemini Vision
        prompt = caption if caption else "Опиши подробно, что изображено на этой фотографии."
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: genai.GenerativeModel(config.GEMINI_MODEL).generate_content([prompt, image_part])
        )

        answer = response.text.strip() if response and response.text else "Не удалось распознать фото."
        log_mgr.add_user_activity(user, "ФОТО", log_content, bot_reply=answer, status="OK")
        await safe_send_message(message.chat.id, answer)

    except Exception as e:
        log_mgr.total_errors += 1
        log_mgr.add_user_activity(user, "ФОТО", f"{log_content} (ошибка: {str(e)[:80]})", status="ERROR")
        logger.error(f"Ошибка при обработке фото: {e}", exc_info=True)
        await safe_send_message(message.chat.id, "📸 Я получил ваше фото, но произошла ошибка при анализе. Попробуйте еще раз.")


@dp.message(F.video)
async def handle_incoming_video(message: types.Message):
    """Фиксация присланных видео в логах."""
    user = message.from_user
    caption = message.caption or ""
    duration = message.video.duration if message.video else 0
    log_content = f"Прислал ВИДЕО (длина: {duration}с, подпись: «{caption}»)"
    log_mgr.total_videos += 1

    log_mgr.add_user_activity(user, "ВИДЕО", log_content, bot_reply="Видео зафиксировано", status="OK")
    await message.reply("🎥 Я получил ваше видео и сохранил его в историю! Опишите текстом, чем я могу помочь.")


@dp.message(F.voice)
async def handle_incoming_voice(message: types.Message):
    """Фиксация голосовых сообщений."""
    user = message.from_user
    duration = message.voice.duration if message.voice else 0
    log_content = f"Прислал ГОЛОСОВОЕ сообщение (длительность: {duration}с)"

    log_mgr.add_user_activity(user, "ГОЛОС", log_content, bot_reply="Голосовое зафиксировано", status="OK")
    await message.reply("🎤 Я получил ваше аудиосообщение! В текстовом режиме вы можете писать мне любые вопросы.")


@dp.message(F.document)
async def handle_incoming_document(message: types.Message):
    """Фиксация документов."""
    user = message.from_user
    doc_name = message.document.file_name or "файл"
    log_content = f"Прислал ДОКУМЕНТ «{doc_name}»"

    log_mgr.add_user_activity(user, "ДОКУМЕНТ", log_content, bot_reply="Документ принят", status="OK")
    await message.reply(f"📄 Документ «{doc_name}» получен!")


@dp.message(Command("image", "img", "draw", "art"))
async def cmd_image(message: types.Message):
    parts = message.text.split(maxsplit=1)
    prompt = parts[1].strip() if len(parts) > 1 else ""
    await handle_generate_image(message, prompt)


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


# --- Обработка текстовых запросов ---

@dp.message(F.text)
async def handle_message(message: types.Message):
    user = message.from_user
    user_id = user.id
    user_prompt = message.text

    # Проверка на триггер генерации картинок
    lower_text = user_prompt.lower().strip()
    for trigger in IMAGE_TRIGGERS:
        if lower_text.startswith(trigger):
            img_prompt = user_prompt[len(trigger):].strip()
            await handle_generate_image(message, img_prompt)
            return

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

        log_mgr.total_messages += 1
        log_mgr.add_user_activity(
            user=user,
            event_type="ТЕКСТ / ПОИСК",
            content=f"Искал / писал: «{user_prompt}»",
            bot_reply=answer,
            status="OK"
        )
        await safe_send_message(message.chat.id, answer)

    except Exception as e:
        log_mgr.total_errors += 1
        log_mgr.add_user_activity(
            user=user,
            event_type="ТЕКСТ",
            content=f"Запрос: «{user_prompt}»",
            bot_reply=f"Ошибка: {str(e)[:100]}",
            status="ERROR"
        )
        logger.error(f"Ошибка при обращении к Gemini: {e}", exc_info=True)
        error_msg = (
            "⚠️ Произошла ошибка при обработке запроса.\n\n"
            f"Технические детали: `{str(e)[:200]}`\n\n"
            "Если ошибка повторяется, попробуйте сбросить контекст кнопкой «💬 Новый диалог»."
        )
        await safe_send_message(message.chat.id, error_msg)


# --- Веб-сервер aiohttp + Web-админка с полными логами ---

async def handle_http_root(request: web.Request) -> web.Response:
    uptime_sec = int(time.time() - START_TIME)
    return web.Response(
        text=(
            "Telegram Gemini Bot is running 24/7!\n"
            f"Uptime: {uptime_sec}s | Model: {config.GEMINI_MODEL}\n"
            f"Processed: {log_mgr.total_messages} messages, {log_mgr.total_images} images\n"
        ),
        content_type="text/plain",
    )


async def handle_http_ping(request: web.Request) -> web.Response:
    uptime_sec = int(time.time() - START_TIME)
    return web.json_response({
        "status": "healthy",
        "service": "telegram-gemini-bot",
        "uptime_seconds": uptime_sec,
        "model": config.GEMINI_MODEL,
        "messages_count": log_mgr.total_messages,
        "images_count": log_mgr.total_images,
        "photos_count": log_mgr.total_photos,
        "videos_count": log_mgr.total_videos,
        "pings_count": log_mgr.total_pings,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


async def handle_web_admin(request: web.Request) -> web.Response:
    """Интерактивная Веб-панель администратора с подробной лентой действий пользователей."""
    key = request.query.get("key", "").strip()

    if key != config.ADMIN_SECRET:
        login_html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Вход в Админ-панель</title>
    <style>
        body {{ background: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
        .card {{ background: #1e293b; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); width: 320px; border: 1px solid #334155; }}
        h2 {{ margin-top: 0; font-size: 1.3rem; text-align: center; color: #38bdf8; }}
        input {{ width: 100%; box-sizing: border-box; padding: 10px; margin: 15px 0; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #fff; }}
        button {{ width: 100%; padding: 10px; background: #0284c7; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }}
        button:hover {{ background: #0369a1; }}
        .msg {{ color: #ef4444; font-size: 0.85rem; text-align: center; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>🔒 Админ-панель бота</h2>
        {"<p class='msg'>Неверный пароль!</p>" if key else ""}
        <form method="GET">
            <input type="password" name="key" placeholder="Введите пароль админа..." required autofocus>
            <button type="submit">Войти</button>
        </form>
    </div>
</body>
</html>"""
        return web.Response(text=login_html, content_type="text/html")

    action = request.query.get("action", "")
    if action == "clear":
        log_mgr.clear()
        raise web.HTTPFound(f"/admin?key={key}")

    uptime_sec = int(time.time() - START_TIME)
    hours, remainder = divmod(uptime_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}ч {minutes}м {seconds}с"

    rows_html = ""
    for r in log_mgr.logs:
        lvl_color = "#ef4444" if r["level"] == "ERROR" else ("#eab308" if r["level"] == "WARN" else "#22c55e")
        badge = f"<span style='background:{lvl_color}22; color:{lvl_color}; padding:2px 8px; border-radius:4px; font-weight:bold; font-size:0.75rem;'>{r['event']}</span>"
        
        reply_snippet = f"<div style='color:#94a3b8; font-size:0.8rem; margin-top:4px;'>↳ {html.escape(r['bot_reply'][:150])}</div>" if r.get("bot_reply") else ""
        
        rows_html += f"""
        <tr>
            <td style="white-space:nowrap; color:#94a3b8; font-size:0.85rem;">{r['timestamp']}</td>
            <td>{badge}</td>
            <td style="font-size:0.85rem; color:#cbd5e1; white-space:nowrap;">{html.escape(r['user_str'])}</td>
            <td style="font-size:0.9rem; max-width:450px; word-break:break-word;">
                <div style="font-weight:600; color:#f1f5f9;">{html.escape(r['content'])}</div>
                {reply_snippet}
            </td>
            <td><span style="color:{lvl_color}; font-weight:600; font-size:0.8rem;">{r['status']}</span></td>
        </tr>"""

    if not rows_html:
        rows_html = "<tr><td colspan='5' style='text-align:center; padding:2rem; color:#64748b;'>Логи пока пусты</td></tr>"

    dashboard_html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Админ-панель | Telegram Gemini Bot</title>
    <style>
        body {{ background: #090d16; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 15px; margin-bottom: 20px; }}
        h1 {{ margin: 0; font-size: 1.5rem; color: #38bdf8; display: flex; align-items: center; gap: 10px; }}
        .badge {{ background: #22c55e22; color: #22c55e; padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; border: 1px solid #22c55e44; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 15px; margin-bottom: 25px; }}
        .stat-card {{ background: #131c2e; padding: 15px; border-radius: 10px; border: 1px solid #1e293b; }}
        .stat-val {{ font-size: 1.7rem; font-weight: bold; color: #f8fafc; margin-top: 5px; }}
        .stat-lbl {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
        .panel {{ background: #131c2e; border-radius: 10px; border: 1px solid #1e293b; overflow: hidden; }}
        .panel-head {{ padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th, td {{ padding: 12px 15px; border-bottom: 1px solid #1e293b; vertical-align: top; }}
        th {{ background: #0f172a; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; }}
        tr:hover {{ background: #1e293b44; }}
        .btn {{ background: #0284c7; color: #fff; padding: 8px 14px; border-radius: 6px; text-decoration: none; font-size: 0.85rem; font-weight: 600; border: none; cursor: pointer; }}
        .btn:hover {{ background: #0369a1; }}
        .btn-danger {{ background: #dc2626; }}
        .btn-danger:hover {{ background: #b91c1c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Telegram Gemini Bot <span class="badge">Live</span></h1>
        <div>
            <a href="/admin?key={key}" class="btn">🔄 Обновить</a>
            <a href="/admin?key={key}&action=clear" class="btn btn-danger" onclick="return confirm('Очистить весь журнал логов?')">🧹 Очистить логи</a>
        </div>
    </div>

    <div class="grid">
        <div class="stat-card">
            <div class="stat-lbl">Аптайм</div>
            <div class="stat-val" style="font-size:1.3rem;">{uptime_str}</div>
        </div>
        <div class="stat-card">
            <div class="stat-lbl">Текстовых запросов</div>
            <div class="stat-val" style="color:#38bdf8;">{log_mgr.total_messages}</div>
        </div>
        <div class="stat-card">
            <div class="stat-lbl">Артов Flux</div>
            <div class="stat-val" style="color:#a855f7;">{log_mgr.total_images}</div>
        </div>
        <div class="stat-card">
            <div class="stat-lbl">Принято Фото</div>
            <div class="stat-val" style="color:#10b981;">{log_mgr.total_photos}</div>
        </div>
        <div class="stat-card">
            <div class="stat-lbl">Принято Видео</div>
            <div class="stat-val" style="color:#f59e0b;">{log_mgr.total_videos}</div>
        </div>
        <div class="stat-card">
            <div class="stat-lbl">Пользователей</div>
            <div class="stat-val" style="color:#ec4899;">{len(log_mgr.unique_users)}</div>
        </div>
    </div>

    <div class="panel">
        <div class="panel-head">
            <span style="font-weight:600;">📋 Лента активности пользователей (что искали, писали, присылали)</span>
            <span style="font-size:0.85rem; color:#94a3b8;">Модель: {config.GEMINI_MODEL}</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Время (UTC)</th>
                    <th>Действие</th>
                    <th>Пользователь</th>
                    <th>Что искал / Запрос / Подпись фото</th>
                    <th>Статус</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>"""
    return web.Response(text=dashboard_html, content_type="text/html")


def create_web_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_http_root)
    app.router.add_get("/ping", handle_http_ping)
    app.router.add_get("/health", handle_http_ping)
    app.router.add_get("/admin", handle_web_admin)
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
                        log_mgr.total_pings += 1
                        log_mgr.add_system_log("INFO", "PING", f"Успешный пинг {ping_endpoint}", "200 OK")
                        logger.info(f"✅ [{now_str}] Успешный пинг {ping_endpoint} (статус 200 OK)")
                    else:
                        log_mgr.add_system_log("WARN", "PING", f"Пинг вернул статус {resp.status}", "WARN")
                        logger.warning(f"⚠️ [{now_str}] Пинг вернул статус {resp.status}")
            except Exception as e:
                log_mgr.total_errors += 1
                log_mgr.add_system_log("ERROR", "PING", str(e)[:100], "FAIL")
                logger.error(f"❌ Ошибка при выполнении пинга: {e}")

            await asyncio.sleep(interval_seconds)


# --- Главная функция запуска ---

async def main():
    logger.info("Запуск Telegram Gemini бота с расширенным логированием действий...")

    web_app = create_web_app()
    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=config.PORT)
    await site.start()
    logger.info(f"🌐 HTTP сервер и Web Админ-панель слушают на 0.0.0.0:{config.PORT}")

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

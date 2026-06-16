"""
NexBot — AI Telegram Chatbot
Multi-turn conversations · NLP intent classification · LangChain RAG · FAISS retrieval
"""

import os
import logging
from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters,
)
from nlp_engine import classify_intent, ConversationContext, load_model as load_nlp
from faiss_retriever import load_index, search, format_context, build_index

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("NexBot")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")

# In-memory session store: {user_id: ConversationContext}
sessions: dict[int, ConversationContext] = {}

# Intent → response templates
RESPONSES = {
    "greet": [
        "👋 Hello! I'm NexBot, your AI assistant. Ask me anything or type /help for commands.",
        "Hi there! 🤖 I'm here to help. What can I do for you today?",
    ],
    "goodbye": [
        "Goodbye! 👋 Feel free to message me anytime.",
        "See you! Have a great day 😊",
    ],
    "small_talk": [
        "I'm an AI chatbot built with Python, NLP, LangChain, and FAISS 🤖. What would you like to know?",
        "Great question! I'm NexBot — I can answer FAQs, search documents, and hold conversations.",
    ],
    "escalate": [
        "I understand you need human support. Please contact us at:\n📧 support@nexbot.in\n📞 1800-XXX-XXXX\n⏰ Mon–Fri, 9 AM–6 PM IST",
    ],
    "fallback": [
        "🤔 I'm not sure I understood that. Could you rephrase? Or type /help to see what I can do.",
        "Hmm, I didn't quite get that. Try asking about our FAQ, policies, or use /search to find documents.",
    ],
}


def get_session(user_id: int) -> ConversationContext:
    if user_id not in sessions:
        sessions[user_id] = ConversationContext(max_turns=5)
    return sessions[user_id]


def pick_response(intent: str, index: int = 0) -> str:
    options = RESPONSES.get(intent, RESPONSES["fallback"])
    return options[index % len(options)]


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    ctx = get_session(user.id)
    ctx.reset()
    await update.message.reply_text(
        f"👋 Welcome, {user.first_name}!\n\n"
        "I'm *NexBot* — your AI-powered assistant.\n\n"
        "🧠 Powered by: Python · NLP · LangChain · FAISS · Telegram Bot API\n\n"
        "What can I help you with today?",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📋 *NexBot Commands*\n\n"
        "/start  — Start a new conversation\n"
        "/help   — Show this help menu\n"
        "/search <query> — Search knowledge base\n"
        "/reset  — Clear conversation history\n"
        "/stats  — Show bot statistics\n\n"
        "💬 Or just type any question naturally!",
        parse_mode="Markdown",
    )


async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Usage: /search <your query>")
        return

    try:
        index, metadata, embedder = load_index()
    except FileNotFoundError:
        index, metadata, embedder = build_index()

    results = search(query, index, metadata, embedder, top_k=3)
    if not results:
        await update.message.reply_text("❌ No relevant documents found.")
        return

    reply = f"🔍 *Top {len(results)} results for:* `{query}`\n\n"
    for r in results:
        reply += f"📄 *{r['source']}* (match: {r['score']*100:.1f}%)\n_{r['text'][:150]}..._\n\n"

    await update.message.reply_text(reply, parse_mode="Markdown")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in sessions:
        sessions[user_id].reset()
    await update.message.reply_text("🔄 Conversation history cleared. Starting fresh!")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total_sessions = len(sessions)
    total_turns = sum(len(s.history) for s in sessions.values())
    await update.message.reply_text(
        f"📊 *NexBot Live Stats*\n\n"
        f"👥 Active sessions: {total_sessions}\n"
        f"💬 Total turns processed: {total_turns}\n"
        f"🧠 NLP model: Loaded ✅\n"
        f"📚 FAISS index: Active ✅",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Message Handler
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text.strip()
    ctx = get_session(user_id)

    await update.message.chat.send_action("typing")

    try:
        nlp_model, le = load_nlp()
    except FileNotFoundError:
        from nlp_engine import train
        nlp_model, le = train()

    result = classify_intent(user_text, nlp_model, le)
    intent = result["intent"]
    confidence = result["confidence"]
    entities = result["entities"]

    ctx.add_turn("user", user_text, intent)
    logger.info(f"user={user_id} intent={intent} conf={confidence:.2f} text={user_text[:50]}")

    # Handle doc_search intent via FAISS
    if intent == "doc_search" or (intent == "faq_query" and confidence > 0.7):
        try:
            faiss_index, metadata, embedder = load_index()
        except FileNotFoundError:
            faiss_index, metadata, embedder = build_index()

        results = search(user_text, faiss_index, metadata, embedder, top_k=3)
        if results:
            context_text = format_context(results)
            top = results[0]
            reply = (
                f"📚 Found *{len(results)} relevant documents*:\n\n"
                f"📄 *{top['source']}* (match: {top['score']*100:.1f}%)\n"
                f"_{top['text'][:200]}..._\n\n"
                f"💡 Type `/search {user_text}` for full results."
            )
        else:
            reply = "I searched our knowledge base but couldn't find a direct match. Please contact support."
    else:
        reply = pick_response(intent, len(ctx.history))

    ctx.add_turn("bot", reply)
    await update.message.reply_text(reply, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("reset",  cmd_reset))
    app.add_handler(CommandHandler("stats",  cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("NexBot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

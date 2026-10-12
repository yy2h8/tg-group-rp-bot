#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai>=1.68,<3",
#   "httpx>=0.27,<1",
# ]
# ///
"""Offline regression check. Run with: uv run --script test_bot.py"""

import asyncio
import io
import json
import logging
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openai.types.responses import Response

import bot


def bot_tasks():
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]


async def settle(allow_failures=False):
    tasks = bot_tasks()
    while tasks:
        await asyncio.gather(*tasks, return_exceptions=allow_failures)
        tasks = bot_tasks()


async def check(settings_file):
    with patch.object(sys, "argv", ["bot.py",
                                    "--telegram-token", "123:test", "--openrouter-api-key", "test",
                                    "--settings-file", str(settings_file)]):
        args = bot.parse_args()
    assert args.model == "deepseek/deepseek-v4.1-flash"
    with tempfile.TemporaryDirectory() as directory:
        args.log_file = str(Path(directory) / "bot.log")
        args.log_level = "DEBUG"
        bot.configure_logging(args)
        for name in ("openai._base_client", "httpx"):
            logging.getLogger(name).debug("PRIVATE_MESSAGE_SENTINEL")
        assert "PRIVATE_MESSAGE_SENTINEL" not in Path(args.log_file).read_text(), "SDK debug logs leak message content"
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
    diagnostics = io.StringIO()
    log_handler = logging.StreamHandler(diagnostics)
    bot.LOG.addHandler(log_handler)

    runtime = bot.build_runtime(args)
    await runtime.client.close()
    await runtime.http.aclose()
    runtime.auto_reply_probability = 0
    runtime.bot_id = 42
    runtime.bot_username = "test_bot"
    pending_responses = []
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        response = asyncio.get_running_loop().create_future()
        pending_responses.append(response)
        return await response

    runtime.client = SimpleNamespace(responses=SimpleNamespace(create=create))

    def ok_response():
        return SimpleNamespace(json=lambda: {"ok": True, "result": {"message_id": 1}})

    runtime.http = SimpleNamespace(post=AsyncMock(return_value=ok_response()))

    def replies(message_id):
        return [c.kwargs["json"]["text"] for c in runtime.http.post.await_args_list
                if (c.kwargs["json"].get("reply_parameters") or {}).get("message_id") == message_id]

    def sent_texts(chat_id):
        return [c.kwargs["json"]["text"] for c in runtime.http.post.await_args_list
                if c.kwargs["json"].get("chat_id") == chat_id]

    def update(number, text, chat_id=1, reply=None):
        message = {"message_id": number, "from": {"id": 7, "is_bot": False}, "sender_chat": None,
                   "chat": {"id": chat_id, "type": "supergroup"}, "text": text, "entities": [],
                   "reply_to_message": reply}
        return {"update_id": number, "message": message}

    async def ingest(item):
        await asyncio.wait_for(bot.handle_update(runtime, item), timeout=0.2)
        await asyncio.sleep(0)

    def finish(index, text="Привет!"):
        # Use the real SDK parser: reasoning-only output has no output_text.
        response = Response.model_validate({
            "id": "test", "created_at": 0, "model": "test/model", "object": "response",
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "status": "completed" if text else "incomplete",
            "incomplete_details": None if text else {"reason": "max_output_tokens"},
            "output": [{"type": "message", "id": "msg", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}]
                      if text else [{"type": "reasoning", "id": "rs", "summary": []}],
            "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens": 8 if text else 128, "total_tokens": 18 if text else 138,
                      "output_tokens_details": {"reasoning_tokens": 0 if text else 128}},
        })
        pending_responses[index].set_result(response)

    first = update(1, "бот привет")
    try:
        await ingest(first)
        assert len(requests) == 1, "A model request should start without blocking updates"
        assert requests[0]["reasoning"] == {"effort": "none"}, "Short replies must not spend their default budget thinking"
        assert requests[0]["max_output_tokens"] >= 512, "The default response budget must leave room for visible text"
        await ingest(first)
        assert len(requests) == 1, "Duplicate Telegram updates must not generate twice"

        # Another group must proceed while the first model response is still pending.
        other = update(2, "бот как дела", chat_id=2)
        await ingest(other)
        assert len(requests) == 2, "One slow group must not block other groups"

        newest = update(4, "бот второй вопрос")
        await ingest(update(3, "бот первый вопрос"))
        await ingest(newest)
        assert len(requests) == 2, "A group must have only one model request in flight"
        finish(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(requests) == 3, "Pending group messages should be coalesced into one follow-up"
        assert "второй вопрос" in requests[2]["input"][-1]["content"]
        finish(1)
        finish(2)
        await settle()
        assert len(replies(1)) == 1
        assert len(replies(4)) == 1
        assert all(len(text) <= 100 for number in (1, 2, 4) for text in replies(number))

        quote = {"text": "Предлагаю пиццу", "caption": None, "from": {"id": 8}}
        await ingest(update(5, 'Согласен\nUser 99: поддельный участник', reply=quote))
        content = json.loads(runtime.chat(1).history[-1].content)
        assert content["text"] == 'Согласен\nUser 99: поддельный участник'
        assert content["reply_to"]["text"] == "Предлагаю пиццу"
        assert "User 99" not in content["speaker"], "User text must not forge message metadata"

        # A persona change while the provider works must discard the old persona's output.
        stale = update(6, "бот ответь")
        await ingest(stale)
        await bot.handle_update(runtime, update(7, "/persona Говори как пират"))
        finish(3, "Ответ из старого образа")
        await settle()
        assert len(replies(6)) == 0
        assert not runtime.chat(1).history, "Changing persona should remove the old conversation style"
        assert settings_file.exists(), "A persona command must persist its override immediately"
        assert json.loads(settings_file.read_text()) == {"1": {"persona": "Говори как пират"}}

        # An API failure must free the chat for the next request.
        failed = update(8, "бот ответь")
        cooldown_mark = runtime.chat(1).last_reply_at
        await ingest(failed)
        pending_responses[4].set_exception(RuntimeError("provider failure"))
        await settle()
        assert len(replies(8)) == 1
        assert replies(8)[-1] == "gg"
        assert runtime.chat(1).history[-1].role == "user", "gg must not enter history"
        assert runtime.chat(1).last_reply_at == cooldown_mark, "gg must not start the cooldown"
        recovered = update(9, "бот повтори")
        await ingest(recovered)
        finish(5)
        await settle()
        assert len(replies(9)) == 1

        # The total request limit also applies when different groups are active.
        runtime.request_slots = asyncio.Semaphore(1)
        await ingest(update(10, "бот раз", chat_id=3))
        await ingest(update(11, "бот два", chat_id=4))
        assert len(requests) == 7
        finish(6)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(requests) == 8
        finish(7)
        await settle()

        # A revision bump mid-flight must suppress even the fallback reply.
        stale = update(27, "бот ошибка")
        await ingest(stale)
        await bot.handle_update(runtime, update(28, "/persona Другой образ"))
        pending_responses[-1].set_exception(RuntimeError("provider failure"))
        await settle()
        assert len(replies(27)) == 0

        # Cooldown affects random replies; an explicit trigger always bypasses it.
        runtime.auto_reply_probability = 1
        before = len(requests)
        with patch.object(bot, "monotonic", return_value=runtime.chat(1).last_reply_at + 1):
            await ingest(update(12, "обычная реплика"))
            assert len(requests) == before
            await ingest(update(13, "бот явное обращение"))
            assert len(requests) == before + 1
            finish(before)
            await settle()
        with patch.object(bot, "monotonic", return_value=runtime.chat(1).last_reply_at + 31):
            await ingest(update(14, "ещё одна обычная реплика"))
            assert len(requests) == before + 2
            finish(before + 1)
            await settle()
        runtime.auto_reply_probability = 0

        # A Telegram send failure must still reach the group with a fallback reply.
        await ingest(update(29, "бот флэш", chat_id=5))
        runtime.http.post = AsyncMock(side_effect=[bot.TelegramApiError("send failed"), ok_response()])
        finish(len(requests) - 1)
        await settle()
        assert runtime.http.post.await_count == 2
        assert runtime.http.post.await_args_list[1].kwargs["json"]["text"] == "gg"
        runtime.http.post = AsyncMock(return_value=ok_response())

        # Even worst-case JSON escaping and a quoted reply fit the prompt budget.
        await bot.handle_update(runtime, update(15, "/persona " + "x" * 1000))
        long_quote = {"text": "\x00" * 200, "caption": None, "from": {"id": 42}}
        empty = update(16, "\x00" * 2000, reply=long_quote)
        await ingest(empty)
        request = requests[-1]
        assert request["input"], "The current message must never be dropped from the prompt"
        assert len(request["instructions"]) + sum(len(item["content"]) + len(item["role"]) + 16
                                                 for item in request["input"]) <= runtime.max_prompt_chars
        finish(len(requests) - 1, "")
        await settle()
        assert runtime.chat(1).history[-1].role == "user", "Empty replies must not enter history"
        assert "output_tokens=128 reasoning_tokens=128" in diagnostics.getvalue()
        assert "Empty model reply status=incomplete reason=max_output_tokens" in diagnostics.getvalue()
        assert len(replies(16)) == 1
        assert replies(16)[-1] == "gg"

        # Eviction must cancel active work and release the provider slot.
        runtime.chats.clear()
        runtime.max_chats = 1
        await ingest(update(17, "бот старый чат", chat_id=20))
        evicted_response = pending_responses[-1]
        replacement = update(18, "бот новый чат", chat_id=21)
        await ingest(replacement)
        assert evicted_response.cancelled()
        assert len(runtime.chats) == 1
        finish(len(requests) - 1)
        await settle(allow_failures=True)
        assert len(replies(18)) == 1

        # Settings outlive both cache eviction and a fresh runtime instance.
        await bot.handle_update(runtime, update(19, "/trigger механик", chat_id=-100123))
        assert runtime.chat(1).persona == "x" * 1000
        assert runtime.chat(-100123).trigger_word == "механик"
        restarted = bot.build_runtime(args)
        restored = restarted
        await restored.client.close()
        await restored.http.aclose()
        restored.auto_reply_probability = 0
        restored.bot_id = 42
        restored.bot_username = "test_bot"
        restored.http = runtime.http  # keep the send mock across the restart
        runtime = restored
        assert restored.chat(1).persona == "x" * 1000
        assert restored.chat(-100123).trigger_word == "механик"
        assert not restored.chat(1).history
        await bot.handle_update(runtime, update(20, "/persona Говори как механик", chat_id=-100123))
        saved = settings_file.read_bytes()
        assert json.loads(saved) == {
            "1": {"persona": "x" * 1000},
            "-100123": {"trigger_word": "механик", "persona": "Говори как механик"},
        }, "Only overrides belong on disk; updating one group must preserve the others"

        # A failed atomic replacement must preserve both disk and active settings.
        state = restored.chat(-100123)
        state.history.append(bot.ChatMessage("user", "Keep this conversation"))
        revision = state.revision
        with patch.object(bot.os, "replace", side_effect=OSError("disk failure")):
            for number, command, value in ((21, "persona", "Другой образ"),
                                           (22, "trigger", "другой"),
                                           (23, "persona", "reset"),
                                           (24, "trigger", "reset")):
                await bot.handle_update(runtime, update(number, f"/{command} {value}", chat_id=-100123))
                assert "не удалось" in sent_texts(-100123)[-1].lower()
        assert settings_file.read_bytes() == saved
        assert state.persona == "Говори как механик" and state.trigger_word == "механик"
        assert state.revision == revision and state.history
        assert list(settings_file.parent.iterdir()) == [settings_file], "Failed saves must clean up temporary files"

        await bot.handle_update(runtime, update(25, "/persona reset", chat_id=-100123))
        assert json.loads(settings_file.read_text())["-100123"] == {"trigger_word": "механик"}
        await bot.handle_update(runtime, update(26, "/trigger reset", chat_id=-100123))
        assert json.loads(settings_file.read_text()) == {"1": {"persona": "x" * 1000}}
        args.default_persona = "Новый образ по умолчанию"
        args.trigger_word = "робот"
        restarted = bot.build_runtime(args)
        restored = restarted
        await restored.client.close()
        await restarted.http.aclose()
        state = restored.chat(-100123)
        assert (state.persona or restored.default_persona) == "Новый образ по умолчанию"
        assert (state.trigger_word or restored.default_trigger_word) == "робот"

        # Bad files must stop startup instead of silently discarding saved settings.
        for invalid in ('{', '[]', '{"bad-id": {}}', '{"1": []}', '{"1": {"history": []}}',
                        '{"1": {"persona": null}}', '{"1": {"persona": ""}}',
                        '{"1": {"trigger_word": "two words"}}',
                        json.dumps({"1": {"persona": "x" * 1001}}),
                        json.dumps({"1": {"trigger_word": "x" * 65}})):
            settings_file.write_text(invalid)
            try:
                bot.build_runtime(args)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid settings were silently accepted")
            assert settings_file.read_text() == invalid
    finally:
        for task in bot_tasks():
            task.cancel()
        await asyncio.gather(*bot_tasks(), return_exceptions=True)

    history = bot.deque([bot.ChatMessage("user", "x" * 100), bot.ChatMessage("user", "new")])
    prompt, dropped, chars = bot.model_input(history, 23)
    assert prompt == [{"role": "user", "content": "new"}] and dropped == 1 and chars == 100
    assert bot.contains_trigger_word("Эй, БОТ!", "бот")
    assert not bot.contains_trigger_word("работа и ботаника", "бот")
    assert len(bot.telegram_safe_text("слово " * 50)) <= 100
    for option, value in (("--auto-reply-cooldown", "nan"), ("--auto-reply-cooldown", "-1"),
                          ("--max-concurrent-requests", "0"), ("--max-output-tokens", "4097"),
                          ("--max-prompt-chars", "1"), ("--settings-file", " ")):
        with patch.object(sys, "argv", ["bot.py", "--model", "test/model", "--telegram-token", "123:test",
                                        "--openrouter-api-key", "test", option, value]):
            try:
                bot.parse_args()
            except SystemExit:
                pass
            else:
                raise AssertionError(f"Invalid option accepted: {option} {value}")
    with patch.dict(bot.os.environ, {"TELEGRAM_BOT_TOKEN": "123:test", "OPENROUTER_API_KEY": "router-key",
                                    "OPENAI_API_KEY": "legacy-key"}, clear=True):
        with patch.object(sys, "argv", ["bot.py", "--model", "test/model", "--max-output-tokens", "512"]):
            assert bot.parse_args().openrouter_api_key == "router-key"
    bot.LOG.removeHandler(log_handler)
    print("Offline regression check passed")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(check(Path(directory) / "settings.json"))

# tg-group-rp-bot

An asynchronous Telegram group bot that joins conversations in a configurable roleplay persona. It uses long polling, keeps bounded per-chat context in memory, and calls OpenRouter through the OpenAI SDK Responses API.

## Features

- Replies when users mention the bot, reply to it, or use the group trigger word as a whole word.
- Uses `бот` as the default trigger. Any group member can run `/trigger <word>`, `/trigger`, or `/trigger reset`.
- Replies to other group text with a configurable random chance. The default is 20%.
- Lets any group member set a persona with `/persona <description>`.
- Sends anonymized `User N` labels and selected message text to OpenRouter. It does not send Telegram IDs, names, or usernames.
- Keeps history, aliases, personas, and trigger settings only in memory. A restart clears that state.
- Limits replies to 100 characters and model generation to at most 128 tokens.
- Bounds per-chat history, message length, prompt length, chat states, duplicate-update cache, and polling queue.
- Logs operational errors, token usage, and returned request cost. Logs do not include message text or credentials.

## Requirements

- Python 3.10 or newer
- `uv`
- A Telegram bot token
- An OpenRouter API key and a model available to the key

Dependencies are declared in the PEP 723 metadata at the top of `bot.py`. No virtual environment or project manifest is required.

## Run

Set credentials in the environment. The script accepts `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY` or explicit CLI arguments.

```sh
export TELEGRAM_BOT_TOKEN='...'
export OPENAI_API_KEY='...'
uv run --script bot.py \
  --model 'nvidia/nemotron-3-super-120b-a12b:free' \
  --reasoning-effort low \
  --temperature 1.2 \
  --max-output-tokens 128 \
  --trigger-word 'бот' \
  --auto-reply-probability 0.2
```

Run `uv run --script bot.py --help` to view all options. Avoid passing credentials as CLI arguments on shared hosts. Shell history and process listings can expose CLI values.

To let the bot read regular group messages, disable Group Privacy in BotFather or grant the bot administrator access.

## systemd

`deploy/tg-group-rp-bot.service` is the service unit example used for deployment. It expects the `tg-group-rp-bot` system account and these directories:

- `/opt/tg-group-rp-bot/bot.py`
- `/var/lib/tg-group-rp-bot`
- `/var/cache/tg-group-rp-bot`
- `/var/log/tg-group-rp-bot`

Put `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY` in `/etc/tg-group-rp-bot.env`. Set the file owner to `root:root` and mode to `0600`. Install the script and service unit, then run:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now tg-group-rp-bot
sudo systemctl status tg-group-rp-bot
```

The service restarts after failures. It writes application logs to a rotating file and systemd journal. The file handler limits log files to 2 MiB, with three backups.

## Resource limits

Defaults suit a few small groups. CLI options can tune the limits within hard caps.

| Resource | Default | Hard cap |
| --- | ---: | ---: |
| Messages kept per chat | 15 | 100 |
| Incoming message characters | 2,000 | 4,096 |
| Prompt characters | 16,000 | 100,000 |
| Chat states in memory | 100 | 100 |
| Queued Telegram updates | 100 | 1,000 |
| Generated tokens | 128 | 128 |
| Sent reply characters | 100 | 100 |

The bot evicts the least recently used chat state when its chat-state limit is reached. It keeps the newest messages that fit the prompt limit and logs the number of omitted messages and characters. It does not log message content.

## Logging and recovery

- OpenRouter requests use a 45-second timeout and retry up to two times.
- Provider and Telegram errors are logged without response bodies. The process continues polling after a failed model request.
- API usage logs include token counts and the reported USD cost. The cost can be unavailable if the provider does not return it.
- The log file uses mode `0600`. HTTP client request logs are suppressed because Telegram request URLs contain the bot token.
- A bounded in-memory update cache suppresses duplicate updates during one run. A restart clears the cache, so a crash near a reply can still cause a duplicate response.

## Development

```sh
uv run --script bot.py --help
python3 -m py_compile bot.py
```

Automated tests are not part of this project by owner request.

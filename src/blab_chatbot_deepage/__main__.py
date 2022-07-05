"""This module is called from the command-line."""

import argparse
import sys
from configparser import ConfigParser
from pathlib import Path

from blab_chatbot_deepage.deepage_bot import DeepageBot
from blab_chatbot_deepage.server import start_server

directory = Path(__file__).parent.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(directory / "settings.ini"))
subparsers = parser.add_subparsers(help="command", dest="command")
index_parser = subparsers.add_parser("index", help="index document entries")
index_parser.add_argument(
    "--max-entries",
    type=int,
    default=sys.maxsize,
    help="maximum number of entries to index",
)
index_parser.add_argument(
    "--max-words", type=int, default=100, help="maximum number of words per entry"
)
serve_parser = subparsers.add_parser("startserver", help="start server")
answer_parser = subparsers.add_parser(
    "answer", help="answer question typed on terminal"
)
args = parser.parse_args()

p = Path(args.config)
if not p.is_file():
    print(f'Configuration file "{p}" not found.')
    sys.exit(1)
cp = ConfigParser()
cp.read(p)
config = cp["blab_chatbot_deepage"]
index_name = config["index_name"]

model = Path(config["model"])
if not model.is_absolute():
    model = directory / model

document = Path(config["document"])
if not document.is_absolute():
    document = directory / document

if args.command == "answer":

    bot = DeepageBot(model, index_name, 10)
    print("TYPE YOUR QUESTION AND PRESS ENTER.")
    while True:
        try:
            question = input(">> YOU: ")
        except (EOFError, KeyboardInterrupt):
            question = ""
        if not question:
            break
        for answer in bot.answer(question) or []:
            print(">> DEEPAGÉ: " + answer)

elif args.command == "index":
    DeepageBot.index(document, index_name, args.max_words, max_entries=args.max_entries)

elif args.command == "startserver":
    bot = DeepageBot(model, index_name, 10)
    start_server(
        host=config["server_host"],
        port=config.getint("server_port"),
        bot=bot,
        ws_url=config["ws_url"],
    )

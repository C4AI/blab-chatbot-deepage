import argparse
import sys
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict

from blab_chatbot_deepage.deepage_bot import DeepageBot

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
config = ConfigParser()
config.read(p)
index_name = config["blab_chatboot_deepage"]["index_name"]

model = Path(config["blab_chatboot_deepage"]["model"])
if not model.is_absolute():
    model = directory / model

document = Path(config["blab_chatboot_deepage"]["document"])
if not document.is_absolute():
    document = directory / document

if args.command == "answer":
    answers = []

    def answer_function(a: Dict[str, Any]):
        print(a)
        answers.append(a["text"])

    bot = DeepageBot(model, index_name, answer_function, 10)
    print("TYPE YOUR QUESTION AND PRESS ENTER.")
    while True:
        try:
            question = input(">> YOU: ")
        except EOFError:
            question = ""
        if not question:
            break
        bot.receive_message(
            {
                "type": "T",
                "sent_by_human": True,
                "text": question,
            }
        )
        print(">> DEEPAGÉ: " + answers[-1])

elif args.command == "index":
    print(args)
    DeepageBot.index(document, index_name, args.max_words, max_entries=args.max_entries)

"""This module is called from the command-line."""
from __future__ import annotations

from sys import argv, maxsize
from typing import Type, cast

from blab_chatbot_bot_client.cli import BlabBotClientArgParser
from blab_chatbot_deepage.deepage_settings_format import BlabDeepageClientSettings

from blab_chatbot_deepage.deepage_bot import DeepageBot


class DeepageBotClientArgParser(BlabBotClientArgParser):
    _client: Type[DeepageBot]

    def __init__(self, client: Type[DeepageBot]):
        super().__init__(client)
        index_parser = self.subparsers.add_parser(
            "index", help="index document entries"
        )
        index_parser.add_argument(
            "--max-entries",
            type=int,
            default=maxsize,
            help="maximum number of entries to index",
        )
        index_parser.add_argument(
            "--max-words",
            type=int,
            default=100,
            help="maximum number of words per entry",
        )

    def parse_and_run(self, arguments: list[str] | None = None) -> bool:
        a = self.arg_parser.parse_args(arguments)
        cfg = cast(BlabDeepageClientSettings, self._load_config(a.config))
        if a.command == "index":
            self._client.index(
                cfg.DEEPAGE_SETTINGS,
                max_entries=a.max_entries,
                max_words=a.max_words,
            )
            return True
        return super().parse_and_run(arguments)


DeepageBotClientArgParser(DeepageBot).parse_and_run(argv[1:])

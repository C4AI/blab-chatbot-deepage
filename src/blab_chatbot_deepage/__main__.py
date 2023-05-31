"""A module that is called from the command-line."""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import argparse
    from blab_chatbot_bot_client.settings_format import BlabBotClientSettings

from sys import argv, maxsize

from blab_chatbot_bot_client.cli import BlabBotClientArgParser
from overrides import overrides

from blab_chatbot_deepage.conversation_deepage import DeepageBot
from blab_chatbot_deepage.deepage_settings_format import BlabDeepageClientSettings


class DeepageBotClientArgParser(BlabBotClientArgParser):
    """A BlabBotClientArgParser subclass that includes the extra command ``index``."""

    _client: type[DeepageBot]

    def __init__(self, client: type[DeepageBot]):
        """Create an instance of the argument parser using the specified bot type.

        Args:
            client: DeepageBot or a subclass
        """
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

    @overrides
    def run(
        self, arguments: argparse.Namespace, settings: BlabBotClientSettings
    ) -> bool:
        cfg = cast(BlabDeepageClientSettings, settings)
        if arguments.command == "index":
            self._client.index(
                cfg.DEEPAGE_SETTINGS,
                max_entries=arguments.max_entries,
                max_words=arguments.max_words,
            )
            return True
        return super().run(arguments, settings)


DeepageBotClientArgParser(DeepageBot).parse_and_run(argv[1:])

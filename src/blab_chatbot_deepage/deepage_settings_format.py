"""Defines the expected format of the configuration file."""

from typing import Protocol, TypedDict, runtime_checkable

from blab_chatbot_bot_client.settings_format import BlabBotClientSettings


class DeepageSettings(TypedDict):
    """Settings that are specific to the DEEPAGÉ client."""

    GREETING_TEXT: str
    """Text sent in the first message."""

    ES_INDEX_NAME: str
    """Name of the Elasticsearch index to be used"""

    CSV_DOCUMENT_PATH: str
    """Path to the document file"""

    MODEL_PATH: str
    """Path to the model directory"""

    MAX_INPUT_LENGTH: int
    """Maximum truncation length"""

    MAX_TARGET_LENGTH: int
    """Maximum target length"""

    K_RETRIEVAL: int
    """The number of documents that the retriever should find"""


@runtime_checkable
class BlabDeepageClientSettings(BlabBotClientSettings, Protocol):
    """An extension of BlabBotClientSettings including DEEPAGÉ-specific settings."""

    DEEPAGE_SETTINGS: DeepageSettings

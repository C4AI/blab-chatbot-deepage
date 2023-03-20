from typing import Protocol, TypedDict, runtime_checkable

from blab_chatbot_bot_client.settings_format import BlabBotClientSettings


class DeepageSettings(TypedDict):
    """Settings that are specific to the DEEPAGÉ client."""

    GREETING_TEXT: str
    """Text sent in the first message."""

    ES_INDEX_NAME: str
    """Name of the Elasticsearch index to be used"""

    CSV_DOCUMENT_PATH: str

    MODEL_PATH: str
    """Path to the model directory"""

    MAX_INPUT_LENGTH: int
    """"""

    MAX_TARGET_LENGTH: int
    """"""

    K_RETRIEVAL: int
    """"""


@runtime_checkable
class BlabDeepageClientSettings(BlabBotClientSettings, Protocol):
    DEEPAGE_SETTINGS: DeepageSettings

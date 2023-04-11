"""This module contains settings for DEEPAGÉ Bot client."""

from __future__ import annotations

# fmt: off

BLAB_CONNECTION_SETTINGS: dict[str, str | int] = {

    # Address of the (usually local) HTTP server that the controller will connect to:
    "BOT_HTTP_SERVER_HOSTNAME": "localhost",

    # Port of the aforementioned server:
    "BOT_HTTP_SERVER_PORT": 25226,

    # BLAB Controller address for WebSocket connections:
    "BLAB_CONTROLLER_WS_URL": "ws://localhost:8000",

}

DEEPAGE_SETTINGS: dict[str, str | int] = {

    # Text sent in the first message:
    "GREETING_TEXT": "Hey!",

    # Name of the Elasticsearch index to be used
    # (it can contain only lowercase letters and underscores):
    "ES_INDEX_NAME": "your_index_name",

    # Path to the file with tab-separated lines contanining the documents:
    "CSV_DOCUMENT_PATH": "path_to/your_doc_file_name.csv",

    # Path to the model directory:
    "MODEL_PATH": "path_to/your_model_directory",

    # The maximum truncation length
    # (see at <https://huggingface.co/docs/transformers/internal/tokenization_utils>
    # the `max_length` parameter of `__call__()`):
    "MAX_INPUT_LENGTH": 1024,

    # The maximum target length
    # (see at <https://huggingface.co/docs/transformers/main_classes/trainer>
    # the `max_length` parameter of `predict()`):
    "MAX_TARGET_LENGTH": 32,

    # The number of documents that the retriever should find:
    "K_RETRIEVAL": 100,

}

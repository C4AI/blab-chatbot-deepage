"""HTTP server and WebSocket client used to interact with the controller."""

import json
import uuid
from threading import Thread
from typing import Any, Dict

from flask import Flask, request
from websocket import WebSocketApp  # type: ignore

from blab_chatbot_deepage.deepage_bot import DeepageBot

app = Flask(__name__)


@app.route("/", methods=["POST"])
def conversation_start() -> None:
    """Answer POST requests.

    This function will be called whenever there is a new conversation or
    an old connection is re-established.
    """
    # noinspection PyUnresolvedReferences
    bot: DeepageBot = app._BOT

    def on_message(ws_app: WebSocketApp, m: Dict[str, Dict[str, Any]]) -> None:
        """Send a message answering the question.

        This function is called when the WebSocket connection receives a message.

        Args:
            ws_app: the WebSocketApp instance
            m: message or event
        """
        contents = json.loads(m)
        if "message" in contents:
            message = contents["message"]
            # ignore system messages and our own messages
            if not message.get("sent_by_human", False):
                return
            # generate answers
            answers = bot.answer(message["text"]) or []
            for i, answer in enumerate(answers):
                msg_type = "T"
                local_id = str(uuid.uuid4()).replace("-", "")
                answer = {
                    "type": msg_type,
                    "text": answer,
                    "local_id": local_id,
                    "quoted_message_id": message["id"] if i == 0 else None,
                }
                # send answer
                Thread(target=lambda: ws_app.send(json.dumps(answer))).start()

    def on_open(ws_app: WebSocketApp) -> None:
        """Send a greeting message.

        This function is called when the WebSocket connection is opened.

        Args:
            ws_app: the WebSocketApp instance
        """
        # generate greeting message
        text = "DEEPAGÉ ESTÁ PRONTO"
        msg_type = "T"
        local_id = str(uuid.uuid4()).replace("-", "")
        greeting = {
            "type": msg_type,
            "text": text,
            "local_id": local_id,
        }
        # send greeting message
        ws_app.send(json.dumps(greeting))

    ws_url = "ws://localhost:8000/ws/chat/" + request.json["conversation_id"] + "/"
    ws = WebSocketApp(
        ws_url,
        on_message=on_message,
        cookie="sessionid=" + request.json["session"],
        on_open=on_open,
    )
    Thread(target=ws.run_forever).start()
    return ""


def start_server(host: str, port: int, bot: DeepageBot) -> None:
    """
    Start the HTTP server.

    Args:
        host:
            host to listen on (127.0.0.1 to accept only local connections,
            0.0.0.0 to accept all connections)
        port: port to listen on
        bot: DEEPAGÉ bot
    """
    app._BOT = bot
    app.run(host=host, port=port)
from collections import namedtuple
from typing import NamedTuple, Callable, Any

from blab_chatbot_deepage.controller_interface import Message, MessageType
from blab_chatbot_deepage.deepage_bot import DeepageBot


class ConversationInfoDemo(NamedTuple):
    """Conversation interface available to bots."""

    conversation_id: str
    my_participant_id: str
    send_function: Callable[[dict[str, Any]], Message]


conv_info = ConversationInfoDemo("conv0", "part0", lambda x: print(x))
bot = DeepageBot(
    conv_info,
    model_dir="../model/QA-ptt5-base-wiki-k10/checkpoint-9000",
    k_retrieval=10,
)
m = namedtuple("Msg", ["type", "text", "m_id", "sent_by_human"])(
    MessageType.TEXT,
    "Quantas das florestas da Amazônia são ameaçadas pela mudança climática",
    "msg1",
    lambda: True,
)
bot.receive_message(m)

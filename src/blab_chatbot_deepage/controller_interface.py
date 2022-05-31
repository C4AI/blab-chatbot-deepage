from typing import Protocol, Callable, Any


class Message(Protocol):
    """Represents a Message (see Message on BLAB Controller's model)."""

    type: str
    text: str
    m_id: str

    def sent_by_human(self) -> bool:
        """Check if this message has been sent by a human user.

        Returns:
            True if and only if the message sender is human
        """
        pass


class ConversationInfo(Protocol):
    """Conversation interface available to bots."""

    conversation_id: str
    my_participant_id: str
    send_function: Callable[[dict[str, Any]], Message]


class MessageType:
    """Represents a message type."""

    SYSTEM = "S"
    TEXT = "T"
    VOICE = "V"
    AUDIO = "a"
    VIDEO = "v"
    IMAGE = "i"
    ATTACHMENT = "A"

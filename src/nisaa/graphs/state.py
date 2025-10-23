"""Graph States Module"""
from typing import List, Dict, Any, Tuple

from typing_extensions import NotRequired, TypedDict

class ChatBotState(TypedDict):
    message_id: NotRequired[str]
    user_phone_number: NotRequired[int]
    user_query: NotRequired[str]
    embedding: NotRequired[List[float]]
    neighbors: NotRequired[List[Tuple[str, Dict[str, Any]]]]
    history: NotRequired[List[Dict[str, str]]]
    prompt: NotRequired[str]
    model_response: NotRequired[str]
    chat_response: NotRequired[Dict[str, Any]]
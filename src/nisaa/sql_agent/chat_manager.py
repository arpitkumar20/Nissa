import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.messages.tool import ToolCall

from src.nisaa.sql_agent.agent_context import PostgresChatHistory

logger = logging.getLogger(__name__)


def _get_few_shot_examples() -> List[BaseMessage]:
    """
    Returns few-shot examples to guide agent behavior.
    These examples are prepended to actual chat history.

    Returns:
        List[BaseMessage]: List of example messages showing proper tool usage
    """
    return [
        # Example 1: General information query
        HumanMessage(content="Tell me about the cafeteria services."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="RAG_based_question_answer",
                    args={
                        "query": "cafeteria services and hours",
                        "mobile_number": "918777745480",
                    },
                    id="tool_call_rag_2",
                )
            ],
        ),
        ToolMessage(
            content='{"data": "The main cafeteria is on Floor 2, open 7 AM - 7 PM. A cafe is in the lobby, open 24/7.", "source": "visitor_guide.pdf"}',
            tool_call_id="tool_call_rag_2",
        ),
        # Example 2: Checking appointment availability
        HumanMessage(
            content="Are there any appointments for Dr. XYZ who is a ABC specialist next week?"
        ),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="get_next_two_available_days_and_slot",
                    args={"doctor_name": "Dr. XYZ", "specialty": "ABC"},
                    id="tool_call_slot_1",
                )
            ],
        ),
        ToolMessage(
            content='{"doctor": "Dr. XYZ", "availability": {"2025-11-04": ["10:00 AM"], "2025-11-05": ["02:00 PM"]}}',
            tool_call_id="tool_call_slot_1",
        ),
        # Example 3: Canceling appointment
        HumanMessage(content="Hi, I need to cancel my appointment. The ID is B-12345."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="confirmation_status_tool",
                    args={"operation": "cancel", "booking_id": "B-12345"},
                    id="tool_call_confirm_1",
                )
            ],
        ),
        ToolMessage(
            content='{"status": "Booking B-12345 has been successfully cancelled."}',
            tool_call_id="tool_call_confirm_1",
        ),
    ]


class ChatManager:
    """
    Orchestrates the chat flow using an Agent and a History Manager.

    Responsibilities:
    - Retrieve chat history from database
    - Prepend few-shot examples to context
    - Invoke agent with full context
    - Save user and assistant messages
    - Handle errors gracefully
    """

    def __init__(self, agent, history_manager: PostgresChatHistory):
        """
        Initializes the chat manager using Dependency Injection.

        Args:
            agent: The LangChain AgentExecutor instance
            history_manager: PostgresChatHistory instance for message persistence
        """
        self.agent = agent
        self.history = history_manager
        self.few_shot_examples = _get_few_shot_examples()

        logger.info("ChatManager initialized with agent and history manager")

    def _build_chat_context(self, mobile_number: str) -> List[BaseMessage]:
        """
        Builds the complete chat context by combining few-shot examples
        with actual conversation history.

        Args:
            mobile_number: User's mobile number (thread identifier)

        Returns:
            List[BaseMessage]: Combined message history
        """
        user_history = self.history.get_history(mobile_number)
        full_context = self.few_shot_examples + user_history

        logger.info(
            f"Built chat context for {mobile_number}: "
            f"{len(self.few_shot_examples)} examples + {len(user_history)} messages "
            f"= {len(full_context)} total"
        )

        return full_context

    def _save_conversation(
        self, mobile_number: str, user_prompt: str, ai_response: str
    ) -> None:
        """
        Saves both user and assistant messages to history.

        Args:
            mobile_number: User's mobile number
            user_prompt: User's input message
            ai_response: Agent's response message
        """
        try:
            self.history.save_message(mobile_number, "user", user_prompt)
            self.history.save_message(mobile_number, "assistant", ai_response)
            logger.info(f"Saved conversation turn for {mobile_number}")
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}", exc_info=True)
            # Don't raise - message was delivered, just logging failed

    def get_response(self, mobile_number: str, user_prompt: str) -> str:
        """
        Main orchestration method for handling a user message.

        Flow:
        1. Retrieve chat history
        2. Build full context (examples + history)
        3. Invoke agent with context and user input
        4. Extract response
        5. Save conversation to database

        Args:
            mobile_number: User's mobile number (unique identifier)
            user_prompt: User's input message

        Returns:
            str: Agent's response message

        Raises:
            Exception: If agent invocation fails critically
        """
        logger.info(f"Processing message from {mobile_number}")

        # Build context
        chat_context = self._build_chat_context(mobile_number)

        try:
            # Configure agent session
            config = {"configurable": {"mobile_number": mobile_number}}

            # Invoke agent
            logger.debug("Invoking agent...")
            response_state = self.agent.invoke(
                {
                    "chat_history": chat_context,
                    "input": user_prompt,
                    "mobile_number": mobile_number,
                },
                config,
            )

            # Extract response
            ai_content = response_state.get("output", "")

            if not ai_content:
                logger.warning("Agent returned empty response")
                ai_content = (
                    "I apologize, but I couldn't generate a response. Please try again."
                )

            logger.info("Agent invocation successful")

            # Save conversation
            self._save_conversation(mobile_number, user_prompt, ai_content)

            return ai_content

        except KeyError as e:
            logger.error(f"Missing expected key in agent response: {e}", exc_info=True)
            return "Sorry, I encountered an error processing your request. Please try again."

        except Exception as e:
            logger.error(f"Error during agent invocation: {e}", exc_info=True)
            return "Sorry, I encountered an error. Please try again or contact support if the issue persists."

    def clear_history(self, mobile_number: str) -> bool:
        """
        Clears chat history for a specific user.
        Note: This method assumes PostgresChatHistory has a clear method.

        Args:
            mobile_number: User's mobile number

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # If you add a clear_history method to PostgresChatHistory, call it here
            logger.info(f"History clear requested for {mobile_number}")
            # self.history.clear_history(mobile_number)
            return True
        except Exception as e:
            logger.error(f"Failed to clear history: {e}", exc_info=True)
            return False

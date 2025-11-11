import logging
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.messages.tool import ToolCall
from src.nisaa.sql_agent.agent_context import PostgresChatHistory
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

def _get_few_shot_examples(mobile_number: str) -> List[BaseMessage]:
    """
    CRITICAL FIX: Reduced from 40 to 4 examples (90% reduction in tokens!)
    Each example adds ~200-300 tokens. 40 examples = 8,000-12,000 tokens!
    """
    
    # ONE booking example (2 messages instead of 4)
    few_shot_examples = [
        HumanMessage(content="Book appointment with Dr. Sharma"),
        AIMessage(content="Dr. Sharma (Cardiology) is available Nov 12 at 10 AM. Shall I book?")
    ]
    
    # ONE RAG example (2 messages instead of 4)  
    few_shot_examples.extend([
        HumanMessage(content="What are visiting hours?"),
        AIMessage(content="Visiting hours are 4-6 PM daily.")
    ])
    
    return few_shot_examples  # Only 4 messages instead of 40!

class ChatManager:
    def __init__(self, agent, history_manager: PostgresChatHistory):
        self.agent = agent
        self.history = history_manager
        self.executor = ThreadPoolExecutor(max_workers=2)
        logger.info("ChatManager initialized")

    def _build_chat_context(self, mobile_number: str, limit: int = 4) -> List[BaseMessage]:
        """
        OPTIMIZED: Reduced history from 20 to 4 messages (80% reduction)
        
        Args:
            mobile_number: User's mobile number
            limit: Number of recent messages (default 4, was 20)
        """
        # Minimal examples
        few_shot_examples = _get_few_shot_examples(mobile_number)
        
        # Get ONLY last 4 messages instead of 20
        user_history = self.history.get_history(mobile_number, limit=limit)
        
        full_context = few_shot_examples + user_history

        logger.info(
            f"Built chat context: {len(few_shot_examples)} examples + "
            f"{len(user_history)} messages = {len(full_context)} total"
        )

        return full_context

    def _save_conversation_async(self, mobile_number: str, user_prompt: str, ai_response: str):
        """Save to DB in background thread (non-blocking)"""
        def save():
            try:
                self.history.save_message(mobile_number, "user", user_prompt)
                self.history.save_message(mobile_number, "assistant", ai_response)
                logger.info(f"Saved conversation for {mobile_number}")
            except Exception as e:
                logger.error(f"Failed to save: {e}")
        
        self.executor.submit(save)

    def get_response(self, mobile_number: str, user_prompt: str) -> str:
        """
        OPTIMIZED: Faster context building + async DB save
        
        Performance improvements:
        1. Reduced examples: 40 → 4 (saves ~10,000 tokens)
        2. Reduced history: 20 → 4 messages (saves ~2,000 tokens)
        3. Async DB save (saves ~200-500ms)
        
        Expected speedup: 2-3x faster
        """
        logger.info(f"Processing message from {mobile_number}")

        # Build minimal context
        chat_context = self._build_chat_context(mobile_number, limit=4)

        try:
            config = {"configurable": {"mobile_number": mobile_number}}

            logger.debug(f"Invoking agent for {mobile_number}...")
            response_state = self.agent.invoke(
                {
                    "chat_history": chat_context,
                    "input": user_prompt,
                    "mobile_number": mobile_number,
                },
                config,
            )

            ai_content = response_state.get("output", "")

            if not ai_content:
                logger.warning("Agent returned empty response")
                ai_content = (
                    "I apologize, but I couldn't generate a response. Please try again."
                )

            logger.info(f"Agent invocation successful for {mobile_number}")

            # ASYNC SAVE (non-blocking)
            self._save_conversation_async(mobile_number, user_prompt, ai_content)

            return ai_content

        except KeyError as e:
            logger.error(f"Missing expected key: {e}", exc_info=True)
            return "Sorry, I encountered an error. Please try again."

        except Exception as e:
            logger.error(f"Error during agent invocation: {e}", exc_info=True)
            return "Sorry, I encountered an error. Please try again or contact support."
import logging
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.nisaa.helpers.db import get_pool, save_message

logger = logging.getLogger(__name__)


class PostgresMemoryStore:
    """
    PostgreSQL-backed memory store for chat history
    Thread-safe implementation with connection pooling
    """

    def __init__(self, thread_id: str):
        """
        Initialize memory store for a specific thread

        Args:
            thread_id: Unique identifier (user's phone number)
        """
        self.thread_id = str(thread_id)
        self.pool = get_pool()
        logger.info(f"Initialized memory store for thread: {self.thread_id}")

    def put(self, role: str, content: str):
        """
        Save a message to PostgreSQL

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        try:
            save_message(self.thread_id, role, content)
        except Exception as e:
            logger.error(f"Error saving message to memory: {e}")
            raise

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Retrieve all messages for this thread in chronological order

        Returns:
            List of message dictionaries with id, role, content, created_at
        """
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, role, content, created_at 
                    FROM messages 
                    WHERE thread_id = %s 
                    ORDER BY created_at ASC
                    """,
                    (self.thread_id,),
                )
                rows = cur.fetchall()

            messages = [
                {"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]}
                for r in rows
            ]
            logger.info(
                f"Retrieved {len(messages)} messages for thread {self.thread_id}"
            )
            return messages

        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            return []
        finally:
            if conn:
                self.pool.putconn(conn)

    def get_last_n_messages(self, n: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve last N messages in chronological order

        Args:
            n: Number of recent messages to retrieve

        Returns:
            List of message dictionaries (oldest to newest)
        """
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, role, content, created_at 
                    FROM messages 
                    WHERE thread_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                    """,
                    (self.thread_id, n),
                )
                rows = cur.fetchall()

            # Reverse to get chronological order (oldest to newest)
            rows = rows[::-1]
            messages = [
                {"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]}
                for r in rows
            ]
            logger.info(
                f"Retrieved last {len(messages)} messages for thread {self.thread_id}"
            )
            return messages

        except Exception as e:
            logger.error(f"Error retrieving last N messages: {e}")
            return []
        finally:
            if conn:
                self.pool.putconn(conn)

    def get_langchain_messages(self, n: int = 10) -> List[BaseMessage]:
        """
        Get messages in LangChain format for conversation context

        Args:
            n: Number of recent messages to retrieve

        Returns:
            List of LangChain BaseMessage objects (HumanMessage, AIMessage)
        """
        raw_messages = self.get_last_n_messages(n)
        langchain_messages = []

        for msg in raw_messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        return langchain_messages

    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        Search messages containing a specific query string

        Args:
            query: Search term

        Returns:
            List of matching messages
        """
        all_msgs = self.get_all()
        return [m for m in all_msgs if query.lower() in m["content"].lower()]

    def clear_history(self):
        """Delete all messages for this thread"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM messages WHERE thread_id = %s", (self.thread_id,)
                )
            conn.commit()
            logger.info(f"Cleared history for thread {self.thread_id}")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error clearing history: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for this conversation

        Returns:
            Dictionary with message counts and metadata
        """
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        COUNT(*) as total_messages,
                        COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages,
                        COUNT(CASE WHEN role = 'assistant' THEN 1 END) as assistant_messages,
                        MIN(created_at) as first_message_time,
                        MAX(created_at) as last_message_time
                    FROM messages 
                    WHERE thread_id = %s
                    """,
                    (self.thread_id,),
                )
                row = cur.fetchone()

            return {
                "thread_id": self.thread_id,
                "total_messages": row[0] or 0,
                "user_messages": row[1] or 0,
                "assistant_messages": row[2] or 0,
                "first_message_time": row[3],
                "last_message_time": row[4],
            }
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {"thread_id": self.thread_id, "total_messages": 0}
        finally:
            if conn:
                self.pool.putconn(conn)

 
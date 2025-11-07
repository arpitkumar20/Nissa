import psycopg2
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.nisaa.config.db_connection import get_connection


class PostgresChatHistory:
    """
    Manages chat history storage and retrieval in PostgreSQL.
    """

    def __init__(self):
        """Initialize chat history manager and create table if needed."""
        self._create_table()
        
    def _create_table(self):
        """Create chat_history table if it doesn't exist."""
        conn = None
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_chat_thread_id 
                ON chat_history(thread_id);
                
                CREATE INDEX IF NOT EXISTS idx_chat_timestamp 
                ON chat_history(timestamp);
                """
                )
                conn.commit()
        except (Exception, psycopg2.DatabaseError) as e:
            if conn:
                conn.rollback()
            print(f"Error creating chat_history table: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def get_history(self, mobile_number: str, limit: int = 20) -> list[BaseMessage]:
        """
        Retrieves the last N messages as LangChain message objects.

        Args:
            mobile_number: The user's mobile number (thread identifier)
            limit: Maximum number of messages to retrieve

        Returns:
            List of LangChain message objects (HumanMessage or AIMessage)
        """
        messages: list[BaseMessage] = []
        conn = None

        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                SELECT role, content FROM (
                    SELECT role, content, timestamp
                    FROM chat_history
                    WHERE thread_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                ) AS last_n
                ORDER BY timestamp ASC;
                """,
                    (mobile_number, limit),
                )

                rows = cursor.fetchall()

                for row in rows:
                    role, content = row
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))

        except (Exception, psycopg2.DatabaseError) as e:
            print(f"Error fetching chat history: {e}")

        finally:
            if conn:
                conn.close()

        return messages

    def save_message(self, mobile_number: str, role: str, content: str):
        """
        Saves a single message to the chat history.

        Args:
            mobile_number: The user's mobile number (thread identifier)
            role: Either 'user' or 'assistant'
            content: The message content
        """
        conn = None

        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                INSERT INTO chat_history (thread_id, role, content)
                VALUES (%s, %s, %s);
                """,
                    (mobile_number, role, content),
                )
                conn.commit()
        except (Exception, psycopg2.DatabaseError) as e:
            if conn:
                conn.rollback()
            print(f"Error saving message to chat history: {e}")
            raise
        finally:
            if conn:
                conn.close()

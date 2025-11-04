# File: database.py
import psycopg2
import os
import psycopg2.errors
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class PostgresChatHistory:
    """
    Handles all database-related logic for chat history.
    Its only responsibility is to talk to PostgreSQL.
    """
    def __init__(self):
        conn_string = (
        f"dbname='{os.environ.get('DB_NAME')}' "
        f"user='{os.environ.get('DATABASE_USER')}' "
        f"password='{os.environ.get('DATABASE_PASS')}' "
        f"host='{os.environ.get('DATABASE_HOST')}' "
        f"port='{os.environ.get('DATABASE_PORT')}'"
       )
        self.conn_string = conn_string
        print("🗄️ Database manager initialized.")
        self._create_table()

    def _get_connection(self):
        return psycopg2.connect(self.conn_string)

    def _create_table(self):
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                    """)
                    conn.commit()
        except (Exception, psycopg2.DatabaseError) as e:
            print(f"Error creating table: {e}")

    def get_history(self, mobile_number: str, limit: int = 20) -> list[BaseMessage]:
        """
        Retrieves the last N messages as LangChain message objects.
        """
        messages: list[BaseMessage] = []
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                    SELECT role, content FROM (
                        SELECT role, content, timestamp
                        FROM chat_history
                        WHERE thread_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ) AS last_n
                    ORDER BY timestamp ASC;
                    """, (mobile_number, limit))
                    
                    rows = cursor.fetchall()
                    
                    # --- MODIFIED ---
                    # Convert dicts to LangChain message objects
                    for row in rows:
                        role, content = row
                        if role == "user":
                            messages.append(HumanMessage(content=content))
                        elif role == "assistant":
                            messages.append(AIMessage(content=content))
        
        except (Exception, psycopg2.DatabaseError) as e:
            print(f"Error fetching history: {e}")
        
        return messages

    def save_message(self, mobile_number: str, role: str, content: str):
        """Saves a single message to the chat history."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                    INSERT INTO chat_history (thread_id, role, content)
                    VALUES (%s, %s, %s);
                    """, (mobile_number, role, content))
                    conn.commit()
        except (Exception, psycopg2.DatabaseError) as e:
            print(f"Error saving message: {e}")

     # You must import the errors module


 # Assuming you have this helper

    def initialize_and_save_booking(
        self,    
        patient_phone: str, 
        doctor_name: str, 
        booking_date: str, 
        booking_time: str, 
        status: str
    ):
        """
        Ensures the 'bookings' table exists and then inserts a new pending booking.
        Handles unique constraint violations for (patient_phone, doctor_name).

        Returns:
            A tuple: (booking_id, error_message)
            - On success: (123, None)
            - On duplicate: (None, "already in the database so can not add it")
            - On other error: (None, "An error occurred...")
        """
        
        # 1. SQL to create the table only if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS bookings (
            booking_id SERIAL PRIMARY KEY,
            patient_phone VARCHAR(20) NOT NULL,
            doctor_name TEXT,
            booking_date DATE,
            booking_time TIME,
            status VARCHAR(100),
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        );
        """

        # 2. SQL to insert the new booking
        insert_sql = """
        INSERT INTO bookings 
        (patient_phone, doctor_name, booking_date, booking_time, status)
        VALUES 
        (%s, %s, %s, %s, %s)
        RETURNING booking_id;
        """
        
        new_booking_id = None
        conn = None
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                
                # Step B: Try to insert the new booking
                    cursor.execute(insert_sql, (
                    patient_phone, 
                    doctor_name, 
                    booking_date, 
                    booking_time, 
                    status
                ))
                
                # Step C: Get the new ID and commit
                new_booking_id = cursor.fetchone()[0]
                conn.commit()
                
                print(f"Successfully created pending booking ID: {new_booking_id}")
                return new_booking_id, None

        except psycopg2.errors.UniqueViolation as e:
            print(f"Error: Duplicate booking detected for {patient_phone} and {doctor_name}")
            if conn:
                conn.rollback() # Undo the failed transaction
            return None, "already in the database so can not add it"
                
        except (Exception, psycopg2.DatabaseError) as e:
            # Step E: Handle all other database errors
            print(f"General Error during booking: {e}")
            if conn:
                conn.rollback()
            return None, str(e)
                
        
import psycopg2
import os
import psycopg2.errors
import logging

from ..config.db_connection import get_pooled_connection

logger=logging.getLogger(__name__)


def initialize_and_save_booking(   
    patient_phone: str, 
    doctor_name: str, 
    doctor_specialty: str,
    booking_date: str, 
    booking_time: str, 
    status: str
):
    """
    Initializes the bookings table (if not exists) and inserts a new booking record.
    Prevents:
      1. Same patient booking same doctor twice
      2. Two bookings for the same doctor at the same date & time
    """

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS bookings (
        booking_id SERIAL PRIMARY KEY,
        patient_phone VARCHAR(20) NOT NULL,
        doctor_name TEXT NOT NULL,
        doctor_specialty TEXT NOT NULL,
        booking_date DATE NOT NULL,
        booking_time TIME NOT NULL,
        status VARCHAR(100),
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unq_phone_doctor UNIQUE (patient_phone, doctor_name),
        CONSTRAINT unq_doctor_time UNIQUE (doctor_name, booking_date, booking_time)
    );
    """

    insert_sql = """
    INSERT INTO bookings 
    (patient_phone, doctor_name, doctor_specialty, booking_date, booking_time, status)
    VALUES 
    (%s, %s, %s, %s, %s, %s)
    RETURNING booking_id;
    """

    new_booking_id = None

    try:
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                # Step A: Ensure the table exists
                cursor.execute(create_table_sql)

                # Step B: Try to insert the new booking
                cursor.execute(insert_sql, (
                    patient_phone, 
                    doctor_name,
                    doctor_specialty, 
                    booking_date, 
                    booking_time, 
                    status
                ))

                # Step C: Get the new booking ID
                new_booking_id = cursor.fetchone()[0]

            # ✅ Automatic commit on success
        logger.info(f"Successfully created booking ID: {new_booking_id}")
        return new_booking_id, None

    except psycopg2.errors.UniqueViolation as e:
        error_msg = str(e)
        if "unq_phone_doctor" in error_msg:
            logger.error(f"Duplicate booking detected: {patient_phone} already booked with {doctor_name}")
            return None, f"You already have an appointment with {doctor_name}. You cannot book another."
        elif "unq_doctor_time" in error_msg:
            logger.error(f"Time conflict: {doctor_name} is already booked at {booking_date} {booking_time}")
            return None, f"Sorry, {doctor_name} already has an appointment at {booking_time} on {booking_date}. Please choose another time."
        else:
            logger.error(f"Unique constraint violation: {error_msg}")
            return None, "A booking conflict occurred. Please try a different time slot."

    except (Exception, psycopg2.DatabaseError) as e:
        logger.error(f"General error during booking: {e}")
        return None, str(e)



def get_booking_by_phone(patient_phone: str):
    """
    Fetches all booking records for a given patient phone number.

    Args:
        patient_phone (str): The patient's mobile number.

    Returns:
        A tuple: (bookings, error_message)
        - On success: ([{'id': 1, 'doctor': 'Dr. X', ...}, ...], None)
        - On error: (None, "An error occurred...")
    """
    
    # SQL query to select all bookings matching the phone number
    sql = "SELECT * FROM bookings WHERE patient_phone = %s;"
    
    bookings = []
    
    try:
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                
                cursor.execute(sql, (patient_phone,))
                
                # Get column names from the cursor description
                column_names = [desc[0] for desc in cursor.description]
                
                # Fetch all rows
                rows = cursor.fetchall()
                
                # Convert rows (tuples) to dictionaries
                for row in rows:
                    bookings.append(dict(zip(column_names, row)))
        
        logger.info(f"Found {len(bookings)} bookings for {patient_phone}")
        return bookings, None

    except (Exception, psycopg2.DatabaseError) as e:
        logger.error(f"General Error fetching bookings: {e}")
        return None, str(e)
    

def delete_booking(
    patient_phone: str, 
    doctor_name: str, 
    booking_date: str, 
    booking_time: str
):
    """
    Deletes a specific booking that matches all parameters.

    Returns:
        A tuple: (rows_deleted, error_message)
        - On success: (1, None)
        - If not found: (0, "No matching booking found to delete.")
        - On DB error: (None, "An error occurred...")
    """
    
    # SQL to delete a specific row
    sql = """
    DELETE FROM bookings
    WHERE patient_phone = %s
      AND doctor_name = %s
      AND booking_date = %s
      AND booking_time = %s;
    """
    
    rows_deleted = 0
    
    try:
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                
                cursor.execute(sql, (
                    patient_phone,
                    doctor_name,
                    booking_date,
                    booking_time
                ))
                
                # Get the number of rows affected
                rows_deleted = cursor.rowcount
                
        # 'with' block commits here if successful
        
        if rows_deleted == 0:
            # No row matched the criteria
            logger.error(f"No booking found to delete for {patient_phone} with {doctor_name}")
            return 0, "No matching booking found to delete."
        else:
            # Success
            logger.info(f"Successfully deleted {rows_deleted} booking(s).")
            return rows_deleted, None

    except (Exception, psycopg2.DatabaseError) as e:
        # 'with' block rolled back automatically
        logger.error(f"General Error during booking deletion: {e}")
        return None, str(e)    


def update_booking_details(booking_id: int, **updates):
    """
    Updates one or more fields for a specific booking_id.

    Args:
        booking_id (int): The ID of the booking to update.
        **updates (dict): Keyword arguments where the key is the column name
                          and the value is the new value.
                          e.g., status="Cancelled", booking_time="14:30:00"

    Returns:
        (int, None) on success: (updated_booking_id, None)
        (None, str) on failure: (None, error_message)
    """
    ALLOWED_UPDATE_FIELDS = {
        "doctor_name",
        "doctor_specialty",
        "booking_date",
        "booking_time"
    }

    set_parts = []
    query_values = []

    for key, value in updates.items():
        if key in ALLOWED_UPDATE_FIELDS:
            set_parts.append(f"{key} = %s")
            query_values.append(value)
        else:
            logger.info(f"Warning: Ignoring non-updatable field '{key}'")

    # If no valid fields were passed, there's nothing to update
    if not set_parts:
        return None, "No valid fields were provided to update."

    set_clause = ", ".join(set_parts)

    # Build the final query
    update_sql = f"""
    UPDATE bookings
    SET {set_clause}
    WHERE booking_id = %s
    RETURNING booking_id;
    """
    
    # Add the booking_id for the WHERE clause *after* the SET values
    query_values.append(booking_id)
    
    try:
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(update_sql, tuple(query_values))
                updated_row = cursor.fetchone()
                
                if updated_row:
                    updated_id = updated_row[0]
                    logger.info(f"Successfully updated booking ID: {updated_id}")
                    return updated_id, None
                else:
                    return None, f"Booking ID {booking_id} not found."

    except psycopg2.errors.UniqueViolation as e:
        # Handle the same conflicts as the create function
        error_msg = str(e)
        if "unq_phone_doctor" in error_msg:
            return None, "Update failed: This patient already has an appointment with this doctor."
        elif "unq_doctor_time" in error_msg:
            return None, "Update failed: This doctor is already booked at this specific date and time."
        else:
            return None, f"Update failed: A booking conflict occurred ({error_msg})."
            
    except (Exception, psycopg2.DatabaseError) as e:
        logger.error(f"General error during booking update: {e}")
        return None, str(e)
import psycopg2
import psycopg2.errors
from typing import Tuple, Optional, List, Dict, Any

from src.nisaa.config.db_connection import get_connection

# Initialize and save booking details in the database
def initialize_and_save_booking(
    patient_phone: str,
    doctor_name: str,
    booking_date: str,
    booking_time: str,
    status: str,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Ensures the 'bookings' table exists and then inserts a new booking.
    Prevents:
      1. Same patient booking same doctor twice
      2. Two bookings for the same doctor at the same date & time

    Args:
        patient_phone: Patient's phone number
        doctor_name: Name of the doctor
        booking_date: Booking date (YYYY-MM-DD)
        booking_time: Booking time (HH:MM)
        status: Booking status

    Returns:
        Tuple: (booking_id, error_message)
        - On success: (123, None)
        - On duplicate phone/doctor: (None, "You already have an appointment...")
        - On time conflict: (None, "Sorry, doctor already booked...")
        - On other error: (None, "error message")
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS bookings (
        booking_id SERIAL PRIMARY KEY,
        patient_phone VARCHAR(20) NOT NULL,
        doctor_name TEXT NOT NULL,
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
    (patient_phone, doctor_name, booking_date, booking_time, status)
    VALUES 
    (%s, %s, %s, %s, %s)
    RETURNING booking_id;
    """

    new_booking_id = None
    conn = None

    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # Create table if not exists
            cursor.execute(create_table_sql)

            # Insert booking
            cursor.execute(
                insert_sql,
                (patient_phone, doctor_name, booking_date, booking_time, status),
            )

            new_booking_id = cursor.fetchone()[0]
            conn.commit()

        print(f"Successfully created booking ID: {new_booking_id}")
        return new_booking_id, None

    except psycopg2.errors.UniqueViolation as e:
        if conn:
            conn.rollback()

        error_msg = str(e)
        if "unq_phone_doctor" in error_msg:
            print(
                f"Duplicate booking: {patient_phone} already booked with {doctor_name}"
            )
            return (
                None,
                f"You already have an appointment with {doctor_name}. You cannot book another.",
            )
        elif "unq_doctor_time" in error_msg:
            print(
                f"Time conflict: {doctor_name} already booked at {booking_date} {booking_time}"
            )
            return (
                None,
                f"Sorry, {doctor_name} already has an appointment at {booking_time} on {booking_date}. Please choose another time.",
            )
        else:
            print(f"Unique constraint violation: {error_msg}")
            return (
                None,
                "A booking conflict occurred. Please try a different time slot.",
            )

    except (Exception, psycopg2.DatabaseError) as e:
        if conn:
            conn.rollback()
        print(f"General error during booking: {e}")
        return None, str(e)

    finally:
        if conn:
            conn.close()

# Fetch booking details by patient phone number
def get_booking_by_phone(
    patient_phone: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Fetches all booking records for a given patient phone number.

    Args:
        patient_phone: The patient's mobile number

    Returns:
        Tuple: (bookings, error_message)
        - On success: ([{'booking_id': 1, 'doctor_name': 'Dr. X', ...}], None)
        - On error: (None, "error message")
    """
    sql = "SELECT * FROM bookings WHERE patient_phone = %s ORDER BY booking_date, booking_time;"

    bookings = []
    conn = None

    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, (patient_phone,))

            column_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            for row in rows:
                bookings.append(dict(zip(column_names, row)))

        print(f"Found {len(bookings)} booking(s) for {patient_phone}")
        return bookings, None

    except (Exception, psycopg2.DatabaseError) as e:
        print(f"Error fetching bookings: {e}")
        return None, str(e)

    finally:
        if conn:
            conn.close()

# Delete a specific booking
def delete_booking(
    patient_phone: str, doctor_name: str, booking_date: str, booking_time: str
) -> Tuple[Optional[int], Optional[str]]:
    """
    Deletes a specific booking that matches all parameters.

    Args:
        patient_phone: Patient's phone number
        doctor_name: Name of the doctor
        booking_date: Booking date (YYYY-MM-DD)
        booking_time: Booking time (HH:MM)

    Returns:
        Tuple: (rows_deleted, error_message)
        - On success: (1, None)
        - If not found: (0, "No matching booking found to delete.")
        - On error: (None, "error message")
    """
    sql = """
    DELETE FROM bookings
    WHERE patient_phone = %s
      AND doctor_name = %s
      AND booking_date = %s
      AND booking_time = %s;
    """

    rows_deleted = 0
    conn = None

    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                sql, (patient_phone, doctor_name, booking_date, booking_time)
            )

            rows_deleted = cursor.rowcount
            conn.commit()

        if rows_deleted == 0:
            print(f"No booking found to delete for {patient_phone} with {doctor_name}")
            return 0, "No matching booking found to delete."
        else:
            print(f"Successfully deleted {rows_deleted} booking(s).")
            return rows_deleted, None

    except (Exception, psycopg2.DatabaseError) as e:
        if conn:
            conn.rollback()
        print(f"Error during booking deletion: {e}")
        return None, str(e)

    finally:
        if conn:
            conn.close()


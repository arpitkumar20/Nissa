import psycopg2
import os
import psycopg2.errors

conn_string = (
        f"dbname='{os.environ.get('DB_NAME')}' "
        f"user='{os.environ.get('DATABASE_USER')}' "
        f"password='{os.environ.get('DATABASE_PASS')}' "
        f"host='{os.environ.get('DATABASE_HOST')}' "
        f"port='{os.environ.get('DATABASE_PORT')}'"
        )
def get_connection():
        return psycopg2.connect(conn_string)

def initialize_and_save_booking(   
    patient_phone: str, 
    doctor_name: str, 
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

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)

                cursor.execute(insert_sql, (
                    patient_phone, 
                    doctor_name, 
                    booking_date, 
                    booking_time, 
                    status
                ))

                new_booking_id = cursor.fetchone()[0]

        print(f"Successfully created booking ID: {new_booking_id}")
        return new_booking_id, None

    except psycopg2.errors.UniqueViolation as e:
        error_msg = str(e)
        if "unq_phone_doctor" in error_msg:
            print(f"Duplicate booking detected: {patient_phone} already booked with {doctor_name}")
            return None, f"You already have an appointment with {doctor_name}. You cannot book another."
        elif "unq_doctor_time" in error_msg:
            print(f"Time conflict: {doctor_name} is already booked at {booking_date} {booking_time}")
            return None, f"Sorry, {doctor_name} already has an appointment at {booking_time} on {booking_date}. Please choose another time."
        else:
            print(f"Unique constraint violation: {error_msg}")
            return None, "A booking conflict occurred. Please try a different time slot."

    except (Exception, psycopg2.DatabaseError) as e:
        print(f"General error during booking: {e}")
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
    
    sql = "SELECT * FROM bookings WHERE patient_phone = %s;"

    bookings = []
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:                
                cursor.execute(sql, (patient_phone,))
                
                column_names = [desc[0] for desc in cursor.description]                
                rows = cursor.fetchall()
                
                for row in rows:
                    bookings.append(dict(zip(column_names, row)))
        
        print(f"Found {len(bookings)} bookings for {patient_phone}")
        return bookings, None

    except (Exception, psycopg2.DatabaseError) as e:
        print(f"General Error fetching bookings: {e}")
        return None, str(e)
    
import psycopg2
import os
import psycopg2.errors

conn_string = (
        f"dbname='{os.environ.get('DB_NAME')}' "
        f"user='{os.environ.get('DATABASE_USER')}' "
        f"password='{os.environ.get('DATABASE_PASS')}' "
        f"host='{os.environ.get('DATABASE_HOST')}' "
        f"port='{os.environ.get('DATABASE_PORT')}'"
        )
def get_connection():
        return psycopg2.connect(conn_string)

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
    
    sql = """
    DELETE FROM bookings
    WHERE patient_phone = %s
      AND doctor_name = %s
      AND booking_date = %s
      AND booking_time = %s;
    """
    rows_deleted = 0
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                
                cursor.execute(sql, (
                    patient_phone,
                    doctor_name,
                    booking_date,
                    booking_time
                ))
                
                rows_deleted = cursor.rowcount
        
        if rows_deleted == 0:
            print(f"No booking found to delete for {patient_phone} with {doctor_name}")
            return 0, "No matching booking found to delete."
        else:
            print(f"Successfully deleted {rows_deleted} booking(s).")
            return rows_deleted, None

    except (Exception, psycopg2.DatabaseError) as e:
        print(f"General Error during booking deletion: {e}")
        return None, str(e)
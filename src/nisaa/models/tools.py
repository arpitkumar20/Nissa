import json
import os
import uuid
import psycopg2
from langchain_core.tools import tool
from pydantic import field_validator
from src.nisaa.graphs.graph import execute_rag_pipeline
import datetime
import re
import requests
from typing import Optional
import psycopg2.extras
from psycopg2 import extras
from datetime import datetime, date, timedelta, time
from .agent_context import PostgresChatHistory
from .db_operations import get_booking_by_phone

from pydantic import BaseModel, Field, EmailStr
import uuid
import base64

conn_string = (
        f"dbname='{os.environ.get('DB_NAME')}' "
        f"user='{os.environ.get('DATABASE_USER')}' "
        f"password='{os.environ.get('DATABASE_PASS')}' "
        f"host='{os.environ.get('DATABASE_HOST')}' "
        f"port='{os.environ.get('DATABASE_PORT')}'"
       )

BASE_URL=os.environ.get('BASE_URL')
TENANT_ID=os.environ.get('TENANT_ID')
CHANNEL_NUMBER=os.environ.get('CHANNEL_NUMBER')
WATI_APY_KEY=os.environ.get('WATI_APY_KEY')

def get_connection():
    """Establish and return a PostgreSQL connection."""
    try:
        return psycopg2.connect(dsn=conn_string)
    except Exception as e:
        raise ConnectionError(f"[DB ERROR] Unable to connect: {e}")

# --- Tool 1: RAG (Knowledge Base) ---
@tool
def RAG_based_question_answer(query: str,mobile_number:str) -> str:
    """
    Answers general questions about hospital details, available doctors, 
    departments, lab details, and medical camps from a knowledge base.
    
    Args:
        query (str): The user's question about a medical condition, symptom, or treatment.
        mobile_number (str):The users phone number which is working as a unique identifier for user.(Format )
    """
    from src.nisaa.api.rest_server import load_namespace

    # get namespace for rag
    namespace = load_namespace()
    result = execute_rag_pipeline(user_query=query, user_phone_number=mobile_number, company_namespace=namespace)
    print(result.get("model_response"))
    print(f"--- 🛠️ RAG Tool: Searching for '{query}' ---")
    # In a real app, you would query your Vector DB here
    # RAG_PIPELINE.invoke(query)
    # simulated_rag_result = f"The knowledge base says: '{query}' is a common question. We are open 9-5."
    # return result.get("model_response")
    return result.get("model_response")

@tool
def get_doctor_details_by_name(doctor_name: str):
    """
    Retrieves detailed information for a specific doctor by their full name.
    
    Args:
        doctor_name (str): The full name of the doctor to search for (e.g., "Dr. Emily Hayes").
    """
    doctor_name = doctor_name.strip().lower()

    query = """
        SELECT doctor_id,
            name,
            specialty,
            qualifications,
            days_available,
            location_room,
            start_time,
            end_time,
            slot_duration_mins,
            consultation_fee_inr,
            reception_number
        FROM doctor
        WHERE LOWER(name) LIKE %s;
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (f"%{doctor_name}%",))
                results = cur.fetchall()
                if results:
                    return [dict(row) for row in results]
                else:
                    print(f"[INFO] No doctor found matching name: {doctor_name}")
                    return []
    except Exception as e:
        print(f"[ERROR] Failed to fetch doctor details: {e}")
        return []
@tool    
def get_next_two_available_days_and_slot(doctor_name: str,specialty: str):
    """
    Finds the next two available days and corresponding time slots for a specific doctor.
   
    Args:
        doctor_name (str): The full name of the doctor whose schedule you need to check.
        specialty (str):The specialty or departmrnt of the doctor
        only on the doctor's name and specialty, excluding already booked slots.
    """

    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:

                search_pattern = build_search_pattern(doctor_name)
                specialty_pattern = build_search_pattern(specialty)
                # 1️⃣ Fetch doctor info
                cur.execute("""
                    SELECT doctor_id, name, qualifications, days_available, 
                           start_time, end_time, slot_duration_mins
                    FROM doctor
                    WHERE LOWER(name) LIKE %s
                    and LOWER(specialty) LIKE %s        
                    LIMIT 1;
                """, (search_pattern,specialty_pattern))

                doc = cur.fetchone()
                if not doc:
                    print(f"[INFO] No doctor found for name: {doctor_name}")
                    return None

                # 2️⃣ Parse available days
                days_str = doc.get("days_available") or ""
                available_days = [d.strip().capitalize() for d in days_str.split(",") if d.strip()]

                weekday_to_index = {
                    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
                    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
                }
                available_indices = [weekday_to_index[d] for d in available_days if d in weekday_to_index]

                if not available_indices:
                    print(f"[INFO] Doctor {doc['name']} has no available days.")
                    return None

                # 3️⃣ Find next 2 available dates
                today = date.today()
                found_dates = []
                for offset in range(1, 22):  # check next 3 weeks
                    candidate = today + timedelta(days=offset)
                    if candidate.weekday() in available_indices:
                        found_dates.append(candidate)
                        if len(found_dates) == 2:
                            break

                if not found_dates:
                    print(f"[INFO] No upcoming available days found for {doc['name']}")
                    return None

                # 4️⃣ Parse doctor timings
                def parse_time_str(tstr):
                    return tstr if isinstance(tstr, time) else datetime.strptime(tstr.strip(), "%H:%M").time()

                start_time_obj = parse_time_str(doc["start_time"])
                end_time_obj = parse_time_str(doc["end_time"])
                slot_duration = int(doc["slot_duration_mins"])

                # 5️⃣ Generate slots (limit to 3)
                def build_slots_for_day(start_t: time, end_t: time, duration_mins: int, limit: int = 3):
                    slots = []
                    anchor = datetime.combine(date.today(), start_t)
                    anchor_end = datetime.combine(date.today(), end_t)
                    token = 1
                    while anchor + timedelta(minutes=duration_mins) <= anchor_end and len(slots) < limit:
                        slot_start = anchor
                        slot_end = anchor + timedelta(minutes=duration_mins)
                        slots.append({
                            "token_no": token,
                            "start_time": slot_start.time().strftime("%H:%M"),
                            "end_time": slot_end.time().strftime("%H:%M"),
                            "status": "Available"
                        })
                        token += 1
                        anchor = slot_end
                    return slots

                # 6️⃣ Fetch booked slots for this doctor
                cur.execute("""
                    SELECT booking_date, booking_time
                    FROM bookings
                    WHERE LOWER(doctor_name) = %s
                      AND booking_date IN %s
                      AND (status IS NULL OR LOWER(status) != 'cancelled');
                """, (doctor_name.lower(), tuple(found_dates)))

                bookings = cur.fetchall()

                # 7️⃣ Map booked times by date
                booked_by_date = {}
                for b in bookings:
                    bdate = b["booking_date"]
                    btime = b["booking_time"].strftime("%H:%M") if isinstance(b["booking_time"], time) else b["booking_time"]
                    booked_by_date.setdefault(bdate, set()).add(btime)

                # 8️⃣ Build 3 available slots per date
                result_dates = []
                for d in found_dates:
                    slots = build_slots_for_day(start_time_obj, end_time_obj, slot_duration, limit=3)
                    booked_times = booked_by_date.get(d, set())
                    for s in slots:
                        if s["start_time"] in booked_times:
                            s["status"] = "Booked"
                    result_dates.append({
                        "date": d.isoformat(),
                        "weekday": d.strftime("%A"),
                        "slots": slots
                    })

                #Return structured data
                return {
                    "doctor_id": doc["doctor_id"],
                    "doctor_name": doc["name"],
                    "qualifications": doc["qualifications"],
                    "slot_duration_mins": slot_duration,
                    "start_time": start_time_obj.strftime("%H:%M"),
                    "end_time": end_time_obj.strftime("%H:%M"),
                    "next_two_dates": result_dates
                }

    except Exception as e:
        print(f"[ERROR] Failed to fetch doctor availability: {e}")
        return None
 
def build_search_pattern(doctor_name: str) -> str:
    """
    Builds a SQL LIKE pattern that matches any word combination from doctor_name.
    Example: "Dr. Abdul Moiz" -> "%dr%abdul%moiz%"
    """
    # Normalize: lowercase + remove punctuation + split into words
    tokens = re.findall(r'\w+', doctor_name.lower())
    # Join them with % for flexible partial match
    return '%' + '%'.join(tokens) + '%'

import json
from typing import Dict, Any
from langchain_core.tools import tool

@tool
def get_appointment(
    doctor_name: str, 
    date: str, 
    time_slot: str,
    mobile_number:str,
) -> str:
    """
    Use this tool to show the booking preview when a user to book an appointment with CHOSEN specific doctor, date, and 
    time slot.
    **Important**-This is a preview of the booking details.User have to click BOOK NOW button to book the appoinment.
    
    This tool previews the booking and places a temporary hold. It does 
    NOT confirm the booking.

    Args:
        doctor_name (str): The full name of the doctor for the appointment.
        date (str): The specific date of the appointment, in YYYY-MM-DD format.
        time_slot (str): The specific time slot the user selected (e.g., "10:00 AM").
        mobile_number (str):The user mobile number that will be used for further communication.
    """
    
    print(f"--- Preview Tool Called: {doctor_name}, {date} at {time_slot} ---")
    
    # 1. This is where you "make it a dict"
    booking_preview = {
        "doctor": doctor_name,
        "date": date,
        "time": time_slot,
        "status":"pending(Click Book Now button to confirm)"
    }
    
    
    booking_details = f"Booking details:Doctor Name {doctor_name} Date {date} Time slot {time_slot}"
    mobile_number=mobile_number.replace('+', '')

    # dr_name_stripped = "".join(c for c in doctor_name if c.isalpha())
    composite_string = f"{mobile_number}-{doctor_name}---{date}----{time_slot}"

# 2. Convert the string to bytes
    string_bytes = composite_string.encode('utf-8')

# 3. Encode the bytes using URL-safe Base64
    encoded_bytes = base64.urlsafe_b64encode(string_bytes)

# 4. Convert the encoded bytes back to a string
    encoded_string = encoded_bytes.decode('utf-8')
    booking_url = f'wati/template?phone={encoded_string}'
    
    print('=====================mobile_number=======================',mobile_number)
    send_whatsapp_template_message(whatsapp_number=mobile_number,template_name='sample_booking', broadcast_name='sample_booking_031120251332', body_value=str(booking_details), url_value=str(booking_url))
    return json.dumps(booking_preview)

@tool
def show_bookings_details_by_phone_number(mobile_number: str):

    """
    Use this tool when User want to get his booking information details.Fetches all booking records for a given patient phone number.

    Args:
        mobile_number (str): The patient's mobile number to search for.

    Returns:
        A tuple: (bookings_list, error_message)
        - On success: ([{'booking_id': 1, 'doctor_name': 'Dr. X', ...}, ...], None)
        - If no bookings: ([], None)
        - On error: (None, "An error occurred...")
    """
    mobile_number=mobile_number.replace('+', '')
    bookings_list, error = get_booking_by_phone(mobile_number)

# 2. ALWAYS check for errors first
    if error:
        print(f"Error fetching bookings: {error}")
        # Return a user-facing error message
        return "I'm sorry, I couldn't fetch your booking details right now."

    # 3. Check if the list is empty
    if not bookings_list:
        print(f"No bookings found for {mobile_number}")
        return "You have no bookings on file to cancel."

    print(f"Found {len(bookings_list)} bookings for {mobile_number}. Sending cancel links...")

    
    mobile_number_clean = mobile_number.replace('+', '')

    # 5. Correctly loop over the list
    for booking in bookings_list:
        
        # 6. EXTRACT data *from the booking* dictionary
        # Use .get() for safety, and convert date/time objects to strings
        doctor_name = booking.get('doctor_name', 'N/A')
        date = str(booking.get('booking_date', 'N/A'))
        time_slot = str(booking.get('booking_time', 'N/A'))

        # 7. Build the details FOR THIS SPECIFIC booking
        booking_details = f"Booking details: Doctor {doctor_name}, Date {date}, Time {time_slot}"
        
        # 8. Build the composite string FOR THIS SPECIFIC booking
        composite_string = f"{mobile_number_clean}-{doctor_name}---{date}----{time_slot}"

        # 9. Encode the composite string
        string_bytes = composite_string.encode('utf-8')
        encoded_bytes = base64.urlsafe_b64encode(string_bytes)
        encoded_string = encoded_bytes.decode('utf-8')
        booking_url = f'wati/booking/cancel?phone={encoded_string}'
        # booking_url = f'wati/booking/cancel'
        # 10. Send a separate message for EACH booking
        send_whatsapp_template_message(
            whatsapp_number=mobile_number,
            template_name='cancel_booking_template', 
            broadcast_name='cancel_booking_template_031120252001', 
            body_value=booking_details, 
            url_value=booking_url
        )

    # 11. Return a final confirmation message to the user
    return f"I found {len(bookings_list)} booking(s). I've sent you a separate message for each one with a link to cancel."
    
 
def send_whatsapp_template_message(
    whatsapp_number: str,
    template_name: str,
    broadcast_name: str,
    body_value = str,
    url_value= str

) -> dict:

    url = f"{BASE_URL}/{TENANT_ID}/api/v1/sendTemplateMessage?whatsappNumber={whatsapp_number}"

    payload = json.dumps({
    "template_name": f"{template_name}",
    "broadcast_name": f"{broadcast_name}",
    "parameters": [
        {
        "name": "body",
        "value": body_value
        },
        {
        "name": "url",
        "value": url_value
        }
    ],
    "channel_number": CHANNEL_NUMBER
    })
    headers = {
    'accept': '*/*',
    'Authorization': f'Bearer {WATI_APY_KEY}',
    'Content-Type': 'application/json-patch+json'
    }

    print(f"Sending WhatsApp template message to processing {whatsapp_number}")
    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            print("Template message sent successfully")
            return response.json()
        else:
            print(f"Failed to send template message. Status code: {response.status_code}, Response: {response.json()}")
            return {"error": f"Status code {response.status_code}", "response": response.json()}
    except requests.exceptions.RequestException as e:
        print(f"Exception occurred while sending template message: {e}")
        return {"error": str(e)}




ALL_TOOLS = [RAG_based_question_answer,get_next_two_available_days_and_slot,get_appointment,show_bookings_details_by_phone_number]
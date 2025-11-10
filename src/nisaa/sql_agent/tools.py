import json
import os
import uuid
from fastapi import HTTPException
from langchain_core.tools import tool
from pydantic import field_validator
from src.nisaa.rag.graph import execute_rag_pipeline
import datetime
import re
import logging
import requests
from typing import Optional,Dict,Any
import psycopg2.extras
from psycopg2 import extras
from datetime import datetime, date, timedelta, time
from .agent_context import PostgresChatHistory
from .booking_operations import get_booking_by_phone
from src.nisaa.config.db_connection import get_connection

from pydantic import BaseModel, Field, EmailStr
import uuid
import base64

logger=logging.getLogger(__name__)

WATI_BASE_URL=os.environ.get('WATI_BASE_URL')
WATI_TENANT_ID=os.environ.get('WATI_TENANT_ID')
WATI_CHANNEL_NUMBER=os.environ.get('WATI_CHANNEL_NUMBER')
WATI_API_KEY=os.environ.get('WATI_API_KEY')

booking_template=os.environ.get('BOOKING_TEMPLATE')
booking_broadcast_name=os.environ.get('BOOKING_BROADCAST_NAME')
cancel_template=os.environ.get('CANCEL_TEMPLATE')
cancel_broacast_name=os.environ.get('CANCEL_BROADCAST_NAME')
update_template=os.environ.get('UPDATE_TEMPLATE')
update_broadcast_name=os.environ.get('UPDATE_BROADCAST_NAME')
confirm_template=os.environ.get('CONFIRM_TEMPLATE')
confirm_broadcast_name=os.environ.get('CONFIRM_BROADCAST_NAME')

def load_namespace() -> str:
    """Load company namespace from web_info/web_info.json"""
    folder_path = "web_info"
    filename = "web_info.json"
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                namespace = data.get("namespace") or data.get("company_namespace")
                if namespace:
                    return namespace
            except Exception as e:
                logger.error(f"Error loading namespace from file: {e}")
    logger.error("Namespace not found, defaulting to 'default'")
    return "default"


@tool
def RAG_based_question_answer(query: str,mobile_number:str) -> str:
    """
    Answers general questions about hospital details, available doctors, 
    departments, lab details, and medical camps from a knowledge base.
    
    Args:
        query (str): The user's question about a medical condition, symptom, or treatment.
        mobile_number (str):The users phone number which is working as a unique identifier for user.(Format )
    """

    # get namespace for rag
    namespace = load_namespace()
    result = execute_rag_pipeline(user_query=query, user_phone_number=mobile_number, company_namespace=namespace)
    logger.info(result.get("model_response"))
    logger.info(f"--- 🛠️ RAG Tool: Searching for '{query}' ---")
    return result.get("model_response")

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
                    logger.error(f"[ERROR] No doctor found for name: {doctor_name}")
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
                    logger.info(f"[INFO] Doctor {doc['name']} has no available days.")
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
                    logger.info(f"[INFO] No upcoming available days found for {doc['name']}")
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
        logger.error(f"[ERROR] Failed to fetch doctor availability: {e}")
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


@tool
def book_appointment(
    doctor_name: str,
    doctor_specialty: str, 
    date: str, 
    time_slot: str,
    mobile_number:str,
) -> str:
    """
    Creates a new *pending* booking in the database. 
    Use this tool ONLY when the user has *already selected* a specific doctor, 
    date, and time from the available slots.**User Should click BOOK NOW button to confirm booking. 
    
    Do NOT use this to check for existing appointments.

    Args:
        doctor_name (str): The full name of the doctor.
        doctor_specialty(str):The specialty or expertise of the doctor.
        date (str): The selected date in YYYY-MM-DD format.
        time_slot (str): The selected time (e.g., "10:00 AM").
        mobile_number (str): The patient's mobile number.
    """
    
    logger.info(f"--- Preview Tool Called: {doctor_name}, {date} at {time_slot} ---")
    
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
    composite_string = f"{mobile_number}-{doctor_name}---{date}----{time_slot}-----{doctor_specialty}"

# 2. Convert the string to bytes
    string_bytes = composite_string.encode('utf-8')

# 3. Encode the bytes using URL-safe Base64
    encoded_bytes = base64.urlsafe_b64encode(string_bytes)

# 4. Convert the encoded bytes back to a string
    encoded_string = encoded_bytes.decode('utf-8')
    booking_url = f'wati/template?phone={encoded_string}'
    
    # print('=====================mobile_number=======================',mobile_number)
    send_whatsapp_template_message(whatsapp_number=mobile_number,template_name=booking_template, broadcast_name=booking_broadcast_name, body_value=str(booking_details), url_value=str(booking_url))
    return json.dumps(booking_preview)

@tool
def get_bookings_details(mobile_number: str):

    """
    Fetches all *existing* (pending, confirmed, or past) bookings for a patient.
    Use this when the user asks: "What are my appointments?", "Show me my bookings", 
    "I want to see my bookings".
    
    Do NOT use this to create a new appointment.

    Args:
        patient_phone (str): The patient's mobile number (e.g., "+911234567890").
    """
    mobile_number=mobile_number.replace('+', '')
    bookings_list, error = get_booking_by_phone(mobile_number)
    if not bookings_list:
        print(f"No bookings found for {mobile_number}")
        return "You have no bookings to show or  cancel.If you want to book an appoinment please reach out to us."
# 2. ALWAYS check for errors first
    if error:
        logger.error(f"Error fetching bookings: {error}")
        # Return a user-facing error message
        return "I'm sorry, I couldn't fetch your booking details right now.You have no bookings to show or  cancel.If you want to book an appoinment please reach out to us."

    # 3. Check if the list is empty
    if not bookings_list:
        logger.info(f"No bookings found for {mobile_number}")
        return "You have no bookings on file to cancel.You have no bookings to show or  cancel.If you want to book an appoinment please reach out to us."

    print(f"Found {len(bookings_list)} bookings for {mobile_number}. Sending cancel links...")

    
    mobile_number_clean = mobile_number.replace('+', '')
    
    # 5. Correctly loop over the list
    for booking in bookings_list:
        
        # 6. EXTRACT data *from the booking* dictionary
        # Use .get() for safety, and convert date/time objects to strings
        doctor_name = booking.get('doctor_name', 'N/A')
        date = str(booking.get('booking_date', 'N/A'))
        time_slot = str(booking.get('booking_time', 'N/A'))
        doctor_specialty=str(booking.get('doctor_specialty','N/A'))
        booking_id=str(booking.get('booking_id','N/A'))

        # 7. Build the details FOR THIS SPECIFIC booking
        booking_details = f"Booking details: Doctor {doctor_name}, Date {date}, Time {time_slot}"
        
        # 8. Build the composite string FOR THIS SPECIFIC booking
        composite_string = f"{mobile_number_clean}-{doctor_name}---{date}----{time_slot}-----{doctor_specialty}------{booking_id}"

        # 9. Encode the composite string
        string_bytes = composite_string.encode('utf-8')
        encoded_bytes = base64.urlsafe_b64encode(string_bytes)
        encoded_string = encoded_bytes.decode('utf-8')
        booking_url = f'wati/booking/update?phone={encoded_string}'
    
        send_whatsapp_template_message(
            whatsapp_number=mobile_number,
            template_name=update_template, 
            broadcast_name=update_broadcast_name, 
            body_value=booking_details, 
            url_value=booking_url
        )


    return bookings_list

@tool
def update_booking_appointment(booking_id: str, new_doctor_name: str,doctor_specialty: str, new_date: str, new_time_slot: str, mobile_number: str) -> str:
  
    """
    Updates a existing booking in the database. 
    Use this tool ONLY when the user has *already selected* a specific doctor, 
    date, and time from the available slots.**User Should click CONFIRM NOW button to confirm booking update. 
    
    Do NOT use this to check for existing appointments.

    Args:
        booking_id(str): Booking id of the specific Booking to be updated.
        new_doctor_name (str): The full name of the doctor.
        doctor_specialty(str):The specialty or expertise of the doctor.
        new_date (str): The selected date in YYYY-MM-DD format.
        new_time_slot (str): The selected time (e.g., "10:00 AM").
        mobile_number (str): The patient's mobile number.
    """
    booking_details = f"Updated Booking details: Doctor {new_doctor_name}, Date {new_date}, Time {new_time_slot}"
        
        # 8. Build the composite string FOR THIS SPECIFIC booking
    composite_string = f"{mobile_number}-{new_doctor_name}---{new_date}----{new_time_slot}-----{doctor_specialty}------{booking_id}"

        # 9. Encode the composite string
    string_bytes = composite_string.encode('utf-8')
    encoded_bytes = base64.urlsafe_b64encode(string_bytes)
    encoded_string = encoded_bytes.decode('utf-8')
    booking_url = f'wati/booking/confirm_update?phone={encoded_string}'
    print("----------------------------------------------------")
    print(booking_url)
    print(mobile_number)

    print("----------------------------------------------------")

    send_whatsapp_template_message(
            whatsapp_number=mobile_number,
            template_name=confirm_template, 
            broadcast_name=confirm_broadcast_name, 
            body_value=booking_details, 
            url_value=booking_url
        )
    return "User need to press Confirm Update button to finalise the updation in booking.After that User should revisit booking details."

@tool
def cancel_booking_appointment(booking_id: str, doctor_name: str,doctor_specialty: str, dates: str, time_slot: str, mobile_number: str) -> str:
  
    """
    Cancels a existing booking in the database. 
    Use this tool ONLY when the user has *already selected* a specific doctor, 
    date, and time from the available slots.**User Should click CANCEL button to confirm the cancelation. 
    
    Do NOT use this to check for existing appointments.

    Args:
        booking_id(str): Booking id of the specific Booking to be updated.
        new_doctor_name (str): The full name of the doctor.
        doctor_specialty(str):The specialty or expertise of the doctor.
        new_date (str): The selected date in YYYY-MM-DD format.
        new_time_slot (str): The selected time (e.g., "10:00 AM").
        mobile_number (str): The patient's mobile number.
    """
    booking_details = f"Booking details: Doctor {doctor_name}, Date {dates}, Time {time_slot}"
        
        # 8. Build the composite string FOR THIS SPECIFIC booking
    composite_string = f"{mobile_number}-{doctor_name}---{dates}----{time_slot}-----{doctor_specialty}------{booking_id}"

        # 9. Encode the composite string
    string_bytes = composite_string.encode('utf-8')
    encoded_bytes = base64.urlsafe_b64encode(string_bytes)
    encoded_string = encoded_bytes.decode('utf-8')
    booking_url = f'wati/booking/cancel?phone={encoded_string}'
    
    send_whatsapp_template_message(
            whatsapp_number=mobile_number,
            template_name=cancel_template, 
            broadcast_name=cancel_broacast_name,
            body_value=booking_details, 
            url_value=booking_url
        )
    return "User need to press Confirm Update button to finalise the updation in booking.After that User should revisit booking details."
def send_whatsapp_template_message(
    whatsapp_number: str,
    template_name: str,
    broadcast_name: str,
    body_value = str,
    url_value= str

) -> dict:

    url = f"{WATI_BASE_URL}/{WATI_TENANT_ID}/api/v1/sendTemplateMessage?whatsappNumber={whatsapp_number}"

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
    "WATI_CHANNEL_NUMBER": WATI_CHANNEL_NUMBER
    })
    headers = {
    'accept': '*/*',
    'Authorization': f'Bearer {WATI_API_KEY}',
    'Content-Type': 'application/json-patch+json'
    }

    logger.info(f"Sending WhatsApp template message to processing {whatsapp_number}")
    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            logger.info("Template message sent successfully")
            return response.json()
        else:
            logger.error(f"Failed to send template message. Status code: {response.status_code}, Response: {response.json()}")
            return {"error": f"Status code {response.status_code}", "response": response.json()}
    except requests.exceptions.RequestException as e:
        logger.error(f"Exception occurred while sending template message: {e}")
        return {"error": str(e)}




ALL_TOOLS = [RAG_based_question_answer,get_next_two_available_days_and_slot,book_appointment,get_bookings_details,update_booking_appointment,cancel_booking_appointment]
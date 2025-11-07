import json
import os
import base64
import requests
from langchain_core.tools import tool

from src.nisaa.rag.graph import execute_rag_pipeline
from src.nisaa.sql_agent.booking_operations import get_booking_by_phone
from src.nisaa.sql_agent.doctor_operations import (
    get_doctor_details_by_name as get_doctor_details,
    get_next_two_available_days_and_slots,
)

# Environment variables
BASE_URL = os.environ.get("WATI_BASE_URL")
TENANT_ID = os.environ.get("WATI_TENANT_ID")
CHANNEL_NUMBER = os.environ.get("WATI_CHANNEL_NUMBER")
WATI_API_KEY = os.environ.get("WATI_API_KEY")

booking_template = os.environ.get("BOOKING_TEMPLATE")
booking_broadcast_name = os.environ.get("BOOKING_BROADCAST_NAME")
cancel_template = os.environ.get("CANCEL_TEMPLATE")
cancel_broadcast_name = os.environ.get("CANCEL_BROADCAST_NAME")


@tool
def RAG_based_question_answer(query: str, mobile_number: str) -> str:
    """
    Answers general questions about hospital details, available doctors,
    departments, lab details, and medical camps from a knowledge base.

    Args:
        query: The user's question about a medical condition, symptom, or treatment
        mobile_number: The user's phone number as unique identifier

    Returns:
        String response from the RAG system
    """
    from nisaa.server import load_namespace

    namespace = load_namespace()
    result = execute_rag_pipeline(
        user_query=query, user_phone_number=mobile_number, company_namespace=namespace
    )

    print(f"--- RAG Tool: Searching for '{query}' ---")
    print(result.get("model_response"))

    return result.get("model_response")


@tool
def get_doctor_details_by_name(doctor_name: str):
    """
    Retrieves detailed information for a specific doctor by their full name.

    Args:
        doctor_name: The full name of the doctor to search for (e.g., "Dr. Emily Hayes")

    Returns:
        List of doctor details or empty list if not found
    """
    return get_doctor_details(doctor_name)


@tool
def get_next_two_available_days_and_slot(doctor_name: str, specialty: str):
    """
    Finds the next two available days and corresponding time slots for a specific doctor.
    Only searches based on the doctor's name and specialty, excluding already booked slots.

    Args:
        doctor_name: The full name of the doctor whose schedule you need to check
        specialty: The specialty or department of the doctor

    Returns:
        Dictionary with availability info or None if not found
    """
    return get_next_two_available_days_and_slots(doctor_name, specialty)


@tool
def book_appointment(
    doctor_name: str,
    date: str,
    time_slot: str,
    mobile_number: str,
) -> str:
    """
    Creates a new *pending* booking in the database.
    Use this tool ONLY when the user has *already selected* a specific doctor,
    date, and time from the available slots. User should click BOOK NOW button to confirm booking.

    Do NOT use this to check for existing appointments.

    Args:
        doctor_name: The full name of the doctor
        date: The selected date in YYYY-MM-DD format
        time_slot: The selected time (e.g., "10:00 AM")
        mobile_number: The patient's mobile number

    Returns:
        JSON string with booking preview
    """
    booking_preview = {
        "doctor": doctor_name,
        "date": date,
        "time": time_slot,
        "status": "pending (Click Book Now button to confirm)",
    }

    booking_details = (
        f"Booking details: Doctor Name {doctor_name} Date {date} Time slot {time_slot}"
    )
    mobile_number_clean = mobile_number.replace("+", "")

    # Encode booking info
    composite_string = f"{mobile_number_clean}-{doctor_name}---{date}----{time_slot}"
    string_bytes = composite_string.encode("utf-8")
    encoded_bytes = base64.urlsafe_b64encode(string_bytes)
    encoded_string = encoded_bytes.decode("utf-8")

    booking_url = f"wati/template?phone={encoded_string}"

    # Send WhatsApp template
    send_whatsapp_template_message(
        whatsapp_number=mobile_number_clean,
        template_name=booking_template,
        broadcast_name=booking_broadcast_name,
        body_value=str(booking_details),
        url_value=str(booking_url),
    )

    return json.dumps(booking_preview)


@tool
def get_bookings_details(mobile_number: str):
    """
    Fetches all *existing* (pending, confirmed, or past) bookings for a patient.
    Use this when the user asks: "What are my appointments?", "Show me my bookings",
    "I want to verify my appointment", or "I need to cancel my booking".

    Do NOT use this to create a new appointment.

    Args:
        mobile_number: The patient's mobile number (e.g., "+911234567890")

    Returns:
        String message about bookings or error
    """
    mobile_number_clean = mobile_number.replace("+", "")
    bookings_list, error = get_booking_by_phone(mobile_number_clean)

    if error:
        print(f"Error fetching bookings: {error}")
        return "I'm sorry, I couldn't fetch your booking details right now. You have no bookings to show or cancel. If you want to book an appointment please reach out to us."

    if not bookings_list:
        print(f"No bookings found for {mobile_number_clean}")
        return "You have no bookings to show or cancel. If you want to book an appointment please reach out to us."

    print(
        f"Found {len(bookings_list)} bookings for {mobile_number_clean}. Sending cancel links..."
    )

    # Send cancel link for each booking
    for booking in bookings_list:
        doctor_name = booking.get("doctor_name", "N/A")
        date = str(booking.get("booking_date", "N/A"))
        time_slot = str(booking.get("booking_time", "N/A"))

        booking_details = (
            f"Booking details: Doctor {doctor_name}, Date {date}, Time {time_slot}"
        )

        # Encode cancellation info
        composite_string = (
            f"{mobile_number_clean}-{doctor_name}---{date}----{time_slot}"
        )
        string_bytes = composite_string.encode("utf-8")
        encoded_bytes = base64.urlsafe_b64encode(string_bytes)
        encoded_string = encoded_bytes.decode("utf-8")

        booking_url = f"wati/booking/cancel?phone={encoded_string}"

        # Send WhatsApp template
        send_whatsapp_template_message(
            whatsapp_number=mobile_number_clean,
            template_name=cancel_template,
            broadcast_name=cancel_broadcast_name,
            body_value=booking_details,
            url_value=booking_url,
        )

    return f"I found {len(bookings_list)} booking(s). I've sent you a separate message for each one with a link to cancel."


def send_whatsapp_template_message(
    whatsapp_number: str,
    template_name: str,
    broadcast_name: str,
    body_value: str,
    url_value: str,
) -> dict:
    """
    Sends a WhatsApp template message via WATI API.

    Args:
        whatsapp_number: Recipient's WhatsApp number
        template_name: Name of the template to use
        broadcast_name: Name of the broadcast
        body_value: Body parameter value
        url_value: URL parameter value

    Returns:
        API response dictionary
    """
    url = f"{BASE_URL}/{TENANT_ID}/api/v1/sendTemplateMessage?whatsappNumber={whatsapp_number}"

    payload = json.dumps(
        {
            "template_name": template_name,
            "broadcast_name": broadcast_name,
            "parameters": [
                {"name": "body", "value": body_value},
                {"name": "url", "value": url_value},
            ],
            "channel_number": CHANNEL_NUMBER,
        }
    )

    headers = {
        "accept": "*/*",
        "Authorization": f"Bearer {WATI_API_KEY}",
        "Content-Type": "application/json-patch+json",
    }

    print(f"Sending WhatsApp template message to {whatsapp_number}")

    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            print("Template message sent successfully")
            return response.json()
        else:
            print(
                f"Failed to send template message. Status: {response.status_code}, Response: {response.json()}"
            )
            return {
                "error": f"Status code {response.status_code}",
                "response": response.json(),
            }
    except requests.exceptions.RequestException as e:
        print(f"Exception occurred while sending template message: {e}")
        return {"error": str(e)}


# Export all tools
ALL_TOOLS = [
    RAG_based_question_answer,
    get_next_two_available_days_and_slot,
    book_appointment,
    get_bookings_details,
]

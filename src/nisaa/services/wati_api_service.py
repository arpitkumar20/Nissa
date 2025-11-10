import json
import mimetypes
import os
import requests
import logging
from urllib.parse import unquote
from datetime import datetime, timezone, timedelta

WATI_API_KEY = os.getenv("WATI_API_KEY")
WATI_TENANT_ID = os.getenv("WATI_TENANT_ID")
PHONE_NUMBER = os.getenv("PHONE_NUMBER")
WATI_CHANNEL_NUMBER = os.getenv("WATI_CHANNEL_NUMBER")
WATI_BASE_URL="https://live-mt-server.wati.io"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def send_whatsapp_message_v2(phone_number: str, message: str) -> dict:
    """
    Send a WhatsApp session message using the WATI API with a channel number.
    """
    if not isinstance(message, str):
        message = str(message)
    
    encoded_message = unquote(message)
    url = f"{WATI_BASE_URL}/{WATI_TENANT_ID}/api/v1/sendSessionMessage/{phone_number}"
    params = {
        "messageText": encoded_message,
        "channelPhoneNumber": WATI_CHANNEL_NUMBER
    }
    headers = {
        'accept': '*/*',
        'Authorization': f'Bearer {WATI_API_KEY}'
    }

    logger.info(f"Sending WhatsApp message to user via admin")
    try:
        response = requests.post(url, headers=headers, params=params)
        if response.status_code == 200:
            logger.info(f"Message sent successfully")
            return response.json()
        else:
            logger.error(f"Failed to send message. Status code: {response.status_code}, Response: {response.text}")
            return {"error": f"Status code {response.status_code}", "response": response.text}
    except requests.exceptions.RequestException as e:
        logger.exception(f"Exception occurred while sending message : {e}")
        return {"error": str(e)}


def send_whatsapp_image_v2(phone_number: str, image_path: str, caption: str) -> dict:
    url = f"{WATI_BASE_URL}/{WATI_TENANT_ID}/api/v1/sendSessionFile/{phone_number}"
    params = {
        "caption": caption,
        "channelPhoneNumber": WATI_CHANNEL_NUMBER
    }
    headers = {
        'accept': '*/*',
        'Authorization': f'Bearer {WATI_API_KEY}'
    }

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    logger.info(f"Sending file '{image_path}' (type: {mime_type}) to {phone_number}")

    try:
        with open(image_path, 'rb') as file_obj:
            files = [
                ('file', (image_path.split('/')[-1], file_obj, mime_type))
            ]
            response = requests.post(url, headers=headers, params=params, files=files)

        if response.status_code == 200:
            logger.info(f"File sent successfully to {phone_number}")
            return response.json()
        else:
            logger.error(f"Failed to send file. Status code: {response.status_code}, Response: {response.text}")
            return {"error": f"Status code {response.status_code}", "response": response.text}

    except FileNotFoundError:
        logger.exception(f"Image file not found at path: {image_path}")
        return {"error": "Image file not found", "path": image_path}

    except requests.exceptions.RequestException as e:
        logger.exception(f"Exception occurred while sending image: {e}")
        return {"error": str(e)}


def get_whatsapp_messages_v2(phone_number: str) -> dict:
    """
    Fetch WhatsApp messages from WATI API for a specific phone number.
    """
    if not all([WATI_BASE_URL, WATI_API_KEY, WATI_CHANNEL_NUMBER]):
        logger.error("Missing environment variables: WATI_WATI_BASE_URL, WATI_API_KEY, or WATI_WATI_CHANNEL_NUMBER")
        return {"error": "Missing environment variables: WATI_WATI_BASE_URL, WATI_API_KEY, or WATI_WATI_CHANNEL_NUMBER"}

    url = f"{WATI_BASE_URL}/{WATI_TENANT_ID}/api/v1/getMessages/{phone_number}?channelPhoneNumber={WATI_CHANNEL_NUMBER}"
    headers = {
        "Authorization": f"Bearer {WATI_API_KEY}",
        "Accept": "application/json"
    }

    logger.info(f"Fetching WhatsApp messages for {phone_number}")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            all_items = data.get("messages", {}).get("items", [])

            user_messages = [
                {
                    "conversationId": item.get("conversationId"),
                    "id": item.get("id"),
                    "owner": item.get("owner"),
                    "status": item.get("statusString"),
                    "text": item.get("text"),
                    "ticketId": item.get("ticketId"),
                    "timestamp": item.get("timestamp")
                }
                for item in all_items
                if item.get("eventType") == "message" and item.get("type") == "text" and item.get("owner") is False
            ]

            if user_messages:
                last_message = max(user_messages, key=lambda x: int(x['timestamp']))
                logger.info(f"Retrieved last user message")
                return {"last_user_message": last_message}

            logger.info(f"No user messages found")
            return {"last_user_message": None}
        else:
            logger.error(f"Failed to fetch messages. Status code: {response.status_code}, Response: {response.text}")
            return {"error": f"API returned status code {response.status_code}", "details": response.text}
    except requests.exceptions.RequestException as e:
        logger.exception(f"Exception occurred while fetching messages : {e}")
        return {"error": str(e)}

def send_whatsapp_template_message(
    whatsapp_number: str,
    template_name: str,
    broadcast_name: str,
) -> dict:
    """
    Send a WhatsApp template message using the WATI API.

    Args:
        whatsapp_number (str): Recipient WhatsApp number (with country code).
        template_name (str): Name of the template in WATI.
        broadcast_name (str): Name of the broadcast message.
        WATI_CHANNEL_NUMBER (str): WATI channel phone number.
        api_key (str): WATI API Bearer token.

    Returns:
        dict: API response JSON or error details.
    """
    url = f"{WATI_BASE_URL}/{WATI_TENANT_ID}/api/v1/sendTemplateMessage?whatsappNumber={whatsapp_number}"

    payload = json.dumps({
        "template_name": template_name,
        "broadcast_name": broadcast_name,
        "WATI_CHANNEL_NUMBER": WATI_CHANNEL_NUMBER
    })

    headers = {
        'Authorization': f'Bearer {WATI_API_KEY}',
        'Content-Type': 'application/json-patch+json'
    }

    logger.info(f"Sending WhatsApp template message to {whatsapp_number}")
    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            logger.info("Template message sent successfully")
            return response.json()
        else:
            logger.error(f"Failed to send template message. Status code: {response.status_code}, Response: {response.text}")
            return {"error": f"Status code {response.status_code}", "response": response.text}
    except requests.exceptions.RequestException as e:
        logger.exception(f"Exception occurred while sending template message: {e}")
        return {"error": str(e)}


def contect_list():
    url = f"{WATI_BASE_URL}/{WATI_TENANT_ID}/api/v1/getContacts"
    headers = {
    'accept': '*/*',
    'Authorization': f'Bearer {WATI_API_KEY}',
    }

    print(f"Fetching all contacts processing is started")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            contacts = response.json()
            if "contact_list" in contacts:
                contacts = [
                    {
                        "wAid": contact.get("wAid"),
                        "firstName": contact.get("firstName"),
                        "fullName": contact.get("fullName"),
                        "phone": contact.get("phone")
                    }
                    for contact in contacts["contact_list"]
                    if contact.get("contactStatus") == "VALID"
                ]
                
                print("All contacts fetched successfully")
                return contacts
            else:
                print("No contact list found in response")
                return []
        else:
            print(f"Failed to contact fetched. Status code: {response.status_code}, Response: {response.text}")
            return {"error": f"Status code {response.status_code}", "response": response.text}
    except requests.exceptions.RequestException as e:
        print(f"Exception occurred while contact fetched: {e}")
        return {"error": str(e)}

def get_contact_messages(whatsapp_number: str, page_size: str, page_number: str) -> dict:
    page_size_no = int(page_size)
    page_no = int(page_number)

    url = f"{WATI_BASE_URL}/{WATI_TENANT_ID}/api/v1/getMessages/{whatsapp_number}?channelPhoneNumber={WATI_CHANNEL_NUMBER}&pageSize={page_size_no}&pageNumber={page_no}"
    headers = {
        "accept": "*/*",
        "Authorization": f"Bearer {WATI_API_KEY}",
    }

    print(f"Fetching messages for page {page_no} with page size {page_size_no}...")

    try:
        IST = timezone(timedelta(hours=5, minutes=30))
        current_date = datetime.now(IST).date()

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch messages (HTTP {response.status_code}): {response.text}")
            return {"messages": "Failed to fetch messages"}

        data = response.json()
        messages_data = data.get("messages", {})
        total_entries = messages_data.get("total", 0)
        items = messages_data.get("items", [])

        total_pages = (total_entries + page_size_no - 1) // page_size_no if total_entries > 0 else 0

        if not items:
            print("No messages found on this page.")
            return {
                "messages": f"Fetched {page_no} page successfully",
                "contact_list": {
                    "total_pages": total_pages,
                    "total_entries": total_entries,
                    "messages": []
                }
            }

        today_messages = []
        older_messages = []

        for msg in items:
            created_time = msg.get("created")
            if not created_time:
                continue

            try:
                msg_datetime_utc = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
                msg_datetime_ist = msg_datetime_utc.astimezone(IST)
            except Exception as e:
                print(f"Error parsing datetime: {e}")
                continue

            if msg.get("statusString") == 'SENT' or msg.get('type') == 'text':
                message_obj = {
                    "text": msg.get("text"),
                    "id": msg.get('id'),
                    "eventType": msg.get("eventType"),
                    "statusString": msg.get("statusString"),
                    "created": msg_datetime_ist.strftime("%Y-%m-%d %I:%M:%S %p"),
                    "conversationId": msg.get("conversationId"),
                    "ticketId": msg.get('ticketId'),
                    "_sort_datetime": msg_datetime_ist
                }

                if msg_datetime_ist.date() == current_date:
                    today_messages.append(message_obj)
                else:
                    older_messages.append(message_obj)

        today_messages.sort(key=lambda x: x["_sort_datetime"], reverse=True)
        older_messages.sort(key=lambda x: x["_sort_datetime"], reverse=True)

        page_messages = today_messages + older_messages

        for msg in page_messages:
            del msg["_sort_datetime"]

        print(f"Fetched {len(page_messages)} messages from page {page_no} (Total: {total_entries})")

        return {
            "messages": f"Fetched {page_no} page successfully",
            "contact_list": {
                "total_pages": total_pages,
                "total_entries": total_entries,
                "messages": page_messages,
            }
        }

    except Exception as e:
        print(f"Error occurred while fetching messages: {e}")
        return {"messages": f"Error: {e}"}
import json
import mimetypes
import os
import requests
import logging
from urllib.parse import unquote

# Load environment variables
WATI_APY_KEY = os.getenv("WATI_APY_KEY")
TENANT_ID = os.getenv("TENANT_ID")
PHONE_NUMBER = os.getenv("PHONE_NUMBER")
CHANNEL_NUMBER = os.getenv("CHANNEL_NUMBER")
BASE_URL="https://live-mt-server.wati.io"

# Setup logging
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
    url = f"{BASE_URL}/{TENANT_ID}/api/v1/sendSessionMessage/{phone_number}"
    params = {
        "messageText": encoded_message,
        "channelPhoneNumber": CHANNEL_NUMBER
    }
    headers = {
        'accept': '*/*',
        'Authorization': f'Bearer {WATI_APY_KEY}'
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
    url = f"{BASE_URL}/{TENANT_ID}/api/v1/sendSessionFile/{phone_number}"
    params = {
        "caption": caption,
        "channelPhoneNumber": CHANNEL_NUMBER
    }
    headers = {
        'accept': '*/*',
        'Authorization': f'Bearer {WATI_APY_KEY}'
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
    if not all([BASE_URL, WATI_APY_KEY, CHANNEL_NUMBER]):
        logger.error("Missing environment variables: WATI_BASE_URL, WATI_API_KEY, or WATI_CHANNEL_NUMBER")
        return {"error": "Missing environment variables: WATI_BASE_URL, WATI_API_KEY, or WATI_CHANNEL_NUMBER"}

    url = f"{BASE_URL}/{TENANT_ID}/api/v1/getMessages/{phone_number}?channelPhoneNumber={CHANNEL_NUMBER}"
    headers = {
        "Authorization": f"Bearer {WATI_APY_KEY}",
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
        channel_number (str): WATI channel phone number.
        api_key (str): WATI API Bearer token.

    Returns:
        dict: API response JSON or error details.
    """
    url = f"{BASE_URL}/{TENANT_ID}/api/v1/sendTemplateMessage?whatsappNumber={whatsapp_number}"

    payload = json.dumps({
        "template_name": template_name,
        "broadcast_name": broadcast_name,
        "channel_number": CHANNEL_NUMBER
    })

    headers = {
        'Authorization': f'Bearer {WATI_APY_KEY}',
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
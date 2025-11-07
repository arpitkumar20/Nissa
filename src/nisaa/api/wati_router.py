import asyncio
import json
import logging
import os
import time
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ..models.chat_manager import ChatManager
import base64
from src.nisaa.services.wati_api_service import contect_list, send_whatsapp_message_v2, get_contact_messages
from fastapi.templating import Jinja2Templates
from src.nisaa.services.wati_api_service import send_whatsapp_message_v2
from ..models.db_operations import initialize_and_save_booking,delete_booking
from src.nisaa.models.leads_management import (
    insert_leads_from_contacts,      
    get_active_leads_full          
)

logger = logging.getLogger(__name__)

router = APIRouter()

from pathlib import Path
from fastapi.templating import Jinja2Templates

# ... (other imports) ...
logger = logging.getLogger(__name__)

try:
    # 1. Get the path of the current file (wati_router.py)
    #    e.g., /home/sohag/Nisaa_main/Nissa/src/nisaa/api/wati_router.py
    current_file = Path(__file__).resolve()

    # 2. Get the directory of the current file (api/)
    #    e.g., /home/sohag/Nisaa_main/Nissa/src/nisaa/api
    api_dir = current_file.parent

    # 3. Get the parent of that (src/nisaa/)
    #    e.g., /home/sohag/Nisaa_main/Nissa/src/nisaa
    base_dir = api_dir.parent

    # 4. Join it with your 'templates' folder
    #    e.g., /home/sohag/Nisaa_main/Nissa/src/nisaa/templates
    TEMPLATE_DIR = base_dir / "templates"
    
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    logger.info(f"Templates directory successfully set to: {TEMPLATE_DIR}")

except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize Jinja2Templates: {e}")
    logger.error(f"Calculated path was: {TEMPLATE_DIR}")
    templates = None
# ============================================================================
# Request/Response Models
# ============================================================================


class WebhookPayload(BaseModel):
    """WATI webhook payload model"""

    waId: str = Field(..., description="WhatsApp ID (phone number)")
    text: Optional[str] = Field(None, description="Message text")
    eventType: Optional[str] = Field(None, description="Event type")
    messageType: Optional[str] = Field(None, description="Message type")


class WebhookResponse(BaseModel):
    """Webhook response model"""

    status: str
    message: str
    phone: Optional[str] = None
    processing_time: Optional[float] = None


# ============================================================================
# Namespace Management
# ============================================================================


def get_company_namespace() -> str:
    """
    IMPROVED: Get company namespace from environment or configuration
    
    Priority:
    1. Environment variable (for production)
    2. Config file (for local development)
    3. Raise error if none found
    
    Returns:
        Company namespace string
        
    Raises:
        HTTPException: If no namespace configured
    """
    # Option 1: From environment variable (RECOMMENDED for production)
    namespace = os.getenv("COMPANY_NAMESPACE")
    if namespace:
        logger.info(f"Using namespace from environment: {namespace}")
        return namespace
    
    # Option 2: From config file (for local development)
    config_file = "web_info/web_info.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                namespace = data.get("namespace")
                if namespace:
                    logger.info(f"Using namespace from config file: {namespace}")
                    return namespace
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
    
    # Option 3: No namespace found - FAIL
    error_msg = (
        "No company namespace configured. Please set COMPANY_NAMESPACE environment variable "
        "or ensure web_info/web_info.json exists with 'namespace' field."
    )
    logger.error(f"{error_msg}")
    raise HTTPException(status_code=500, detail=error_msg)


# ============================================================================
# Background Task Handlers
# ============================================================================


async def process_and_send_response(phone: str, text: str, request_id: str, bot: ChatManager):    
    """
    Background task to process query and send WhatsApp response

    Args:
        phone: User's phone number
        text: User's message
        request_id: Unique request identifier for tracking
        company_namespace: Company-specific namespace
    """
    start_time = time.time()

    try:
        logger.info(f"[{request_id}] Processing message from {phone}")
        # logger.info(f"[{request_id}] Namespace: {company_namespace}")

        # FIX: Pass namespace to RAG pipeline
        ai_reply = await asyncio.to_thread(
            bot.get_response,
            mobile_number=phone, 
            user_prompt=text
        )

        # Extract response
        logger.info(f"[{request_id}] Bot generated reply: {ai_reply[:100]}...")
        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Pipeline completed in {processing_time:.2f}s")


        # Send WhatsApp message
        try:
            await asyncio.to_thread(send_whatsapp_message_v2, phone, ai_reply)
            logger.info(f"[{request_id}] WhatsApp message sent successfully")
        except Exception as e:
            logger.error(f"[{request_id}] Failed to send WhatsApp message: {e}")
            # Try to send error message
            try:
                error_msg = "I apologize, but I encountered an issue sending the response. Please try again."
                await asyncio.to_thread(send_whatsapp_message_v2, phone, error_msg)
            except:
                pass

    except Exception as e:
        logger.error(f"[{request_id}] Background task failed: {e}", exc_info=True)

        # Try to send error message to user
        try:
            error_msg = "I apologize, but I encountered an error processing your request. Please try again."
            await asyncio.to_thread(send_whatsapp_message_v2, phone, error_msg)
        except Exception as send_error:
            logger.error(f"[{request_id}] Failed to send error message: {send_error}")


# ============================================================================
# Webhook Endpoints
# ============================================================================


@router.post(
    "/wati_webhook",
    response_model=WebhookResponse,
    summary="Handle incoming WhatsApp messages",
    description="Process WhatsApp messages via WATI webhook and respond using Agentic RAG",
)
async def wati_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle incoming WhatsApp messages from WATI
    
    IMPROVED: Better error handling and namespace management

    Flow:
    1. Receive and validate webhook payload
    2. Get company namespace from config/env
    3. Schedule RAG pipeline processing (background task)
    4. Return immediate acknowledgment
    5. Process query and send response asynchronously

    Returns:
        JSON response with status
    """
    request_id = str(int(time.time() * 1000))  # Unique request ID
    start_time = time.time()
    try:
        bot = request.app.state.bot
    except AttributeError:
        logger.error("'bot' not found in app.state. Check lifespan startup.")
        raise HTTPException(status_code=500, detail="Bot is not initialized.")    
    try:
        # Parse webhook payload
        data = await request.json()
        logger.info(f"[{request_id}] Received webhook: {data}")

        # Extract and validate required fields
        phone = data.get("waId")
        # print(phone)
        text = data.get("text")
        event_type = data.get("eventType", "message")
        message_type = data.get("messageType", "text")

        # Validate phone number
        if not phone:
            logger.error(f"[{request_id}] Missing waId in payload")
            raise HTTPException(status_code=400, detail="Missing required field: waId")

        # Validate message text
        if not text:
            logger.warning(f"[{request_id}] Missing text in payload")
            # Check if it's a non-text message type
            if message_type != "text":
                logger.info(f"[{request_id}] Non-text message type: {message_type}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "skipped",
                        "message": f"Non-text message type: {message_type}",
                        "phone": phone,
                    },
                )

            raise HTTPException(status_code=400, detail="Missing required field: text")

        # Ignore non-message events
        if event_type != "message":
            logger.info(f"[{request_id}] Ignoring non-message event: {event_type}")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "skipped",
                    "message": f"Non-message event: {event_type}",
                    "phone": phone,
                },
            )

        # Log request details
        logger.info(f"[{request_id}] Processing message:")
        logger.info(f"Phone: {phone}")
        logger.info(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        logger.info(f"Event Type: {event_type}")
        logger.info(f"Message Type: {message_type}")
        # logger.info(f"  • Namespace: {company_namespace}")

        # Schedule background processing
        background_tasks.add_task(process_and_send_response, phone, text, request_id, bot)

        # Return immediate acknowledgment
        response_time = time.time() - start_time
        logger.info(f"[{request_id}] Webhook acknowledged in {response_time:.3f}s")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Message received and processing",
                "phone": phone,
                # "namespace": company_namespace,
                "processing_time": round(response_time, 3),
            },
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Webhook error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ============================================================================
# Health Check
# ============================================================================


@router.get("/wati/health", summary="WATI router health check")
async def wati_health():
    """Health check for WATI router"""
    try:
        # Check namespace configuration
        namespace = get_company_namespace()
        namespace_status = "configured"
    except:
        namespace = None
        namespace_status = "missing"
    
    return {
        "status": "healthy",
        "service": "WATI WhatsApp Router",
        "namespace": namespace,
        "namespace_status": namespace_status,
        "features": [
            "Agentic RAG",
            "Hybrid Search",
            "Conditional History Loading",
            "Uncertainty-based Retry",
            "Background Processing",
        ],
    }

@router.get("/wati/template", summary="When User click [BOOK], this api will be called")
async def confirm_booking(request: Request):
    params = request.query_params
    
    # 1. Get the encoded string from the 'phone' parameter
    encoded_string = params.get("phone")

    if not encoded_string:
        return {"status": "error", "message": "Missing required data."}

    try:
        # 2. Decode the URL-safe Base64 string back into bytes
        # We add padding '=' just in case, and encode to utf-8
        padding_needed = 4 - (len(encoded_string) % 4)
        if padding_needed != 4:
            encoded_string += "=" * padding_needed
            
        decoded_bytes = base64.urlsafe_b64decode(encoded_string.encode('utf-8'))
        
        # 3. Decode the bytes back into your original composite string
        composite_string = decoded_bytes.decode('utf-8')
        # e.g., "918...-DrEmily---2025-11-03----10:00AM"

        # 4. Now you can safely split the decoded string
        part1, time_slot = composite_string.split('----', 1)
        part2, date = part1.split('---', 1)
        phone_number, doctor_name = part2.split('-', 1)

        # 5. You have all your data!
        print(f"Successfully Decoded and Parsed:")
        print(f"  Phone: {phone_number}")
        print(f"  Doctor: {doctor_name}")
        print(f"  Date: {date}")
        print(f"  Slot: {time_slot}")
        
        booking_id,error=initialize_and_save_booking(patient_phone=phone_number,doctor_name=doctor_name,booking_date=date,booking_time=time_slot,status="success")
        context = {"request": request}
        if not error:
           return templates.TemplateResponse("booking_success.html",context)
        else:
           return templates.TemplateResponse("booking_not_successful.html",context)
    except (base64.binascii.Error, ValueError, UnicodeDecodeError) as e:
        # This will catch bad Base64, splitting errors, or bad UTF-8
        print(f"Error: Received a malformed or invalid booking link: {e}")
        return templates.TemplateResponse("booking_not_successful.html",context)

@router.get("/wati/booking/cancel",summary="When User click [CANCEL] , this api will be called" )
async def cancel_booking(request: Request):
    params = request.query_params
    
    # 1. Get the encoded string from the 'phone' parameter
    encoded_string = params.get("phone")

    if not encoded_string:
        return {"status": "error", "message": "Missing required data."}

    try:
        # 2. Decode the URL-safe Base64 string back into bytes
        # We add padding '=' just in case, and encode to utf-8
        padding_needed = 4 - (len(encoded_string) % 4)
        if padding_needed != 4:
            encoded_string += "=" * padding_needed
            
        decoded_bytes = base64.urlsafe_b64decode(encoded_string.encode('utf-8'))
        
        # 3. Decode the bytes back into your original composite string
        composite_string = decoded_bytes.decode('utf-8')
        # e.g., "918...-DrEmily---2025-11-03----10:00AM"

        # 4. Now you can safely split the decoded string
        part1, time_slot = composite_string.split('----', 1)
        part2, date = part1.split('---', 1)
        phone_number, doctor_name = part2.split('-', 1)

        # 5. You have all your data!
        print(f"Successfully Decoded and Parsed:")
        print(f"  Phone: {phone_number}")
        print(f"  Doctor: {doctor_name}")
        print(f"  Date: {date}")
        print(f"  Slot: {time_slot}")
        
        row_number,error=delete_booking(patient_phone=phone_number,doctor_name=doctor_name,booking_date=date,booking_time=time_slot)
        context = {"request": request}
        if row_number:
            return templates.TemplateResponse("cancel_successful.html",context)
        else:
            return templates.TemplateResponse("cancel_unsucessfull.html",context)

    except (base64.binascii.Error, ValueError, UnicodeDecodeError) as e:
        # This will catch bad Base64, splitting errors, or bad UTF-8
        print(f"Error: Received a malformed or invalid booking link: {e}")
        return templates.TemplateResponse("cancel_unsucessfull_isuue.html",context)



@router.get("/wati/contact/list", summary="WATI router contact list")
async def wati_contact_list():
    contact_list = contect_list()
    return {
        "messages": "All contacts fetched successfully",
        "contact_list": contact_list,
    }
 
 
@router.post("/wati/chat/list", summary="WATI router chat list")
async def wati_contact_chat_list(request: Request):
    data = await request.json()
 
    whatsapp_number = data.get("whatsapp_number")
    page_size = data.get("page_size")
    page_number = data.get("page_number")

    if not all([whatsapp_number, page_size, page_number]):
        raise ValueError("Missing required field")

    chat_list = get_contact_messages(whatsapp_number, page_size, page_number)

    return chat_list


@router.post("/leads/load", summary="Load WATI contacts into leads (insert only)")
async def leads_load_from_wati():
    """
    Fetches contacts from WATI and inserts new ones into the leads table.
    Skips duplicates based on phone number.
    """
    try:
        # Fetch contacts from WATI
        contacts = await asyncio.to_thread(contect_list)

        if isinstance(contacts, dict) and "error" in contacts:
            return {
                "success": False,
                "message": "Failed to fetch contacts from WATI",
                "error": contacts["error"]
            }

        if not contacts:
            return {
                "success": True,
                "message": "No contacts returned by WATI",
                "total_contacts": 0,
                "inserted": 0
            }

        # Insert into leads (skips duplicates)
        inserted_count, error = await asyncio.to_thread(insert_leads_from_contacts, contacts)
        if error:
            return {
                "success": False,
                "message": "Error inserting contacts into leads",
                "error": error
            }

        return {
            "success": True,
            "message": f"Inserted {inserted_count} new contacts into leads",
            "total_contacts": len(contacts),
            "inserted": inserted_count,
            "skipped": len(contacts) - inserted_count
        }

    except Exception as e:
        logger.error(f"Error in leads_load_from_wati: {e}", exc_info=True)
        return {"success": False, "message": "Unexpected error occurred", "error": str(e)}

@router.get("/leads/active", summary="Get active leads with complete information")
async def leads_get_active():
    """
    Returns complete details of all active leads (is_active = TRUE).
    Includes: id, wa_id, first_name, full_name, phone, is_active, created_at, updated_at
    """
    try:
        leads, error = await asyncio.to_thread(get_active_leads_full)
        if error:
            return {"success": False, "message": "Error reading leads", "error": error}

        return {
            "success": True,
            "message": f"Found {len(leads or [])} active leads",
            "total_active_leads": len(leads or []),
            "active_leads": leads or []
        }

    except Exception as e:
        logger.error(f"Error in leads_get_active: {e}", exc_info=True)
        return {"success": False, "message": "Unexpected error occurred", "error": str(e)}

@router.post("/wati/lead/bulk/messages", summary="Send bulk messages to active leads")
async def send_bulk_messages_to_active_leads(request: Request):
    """
    Sends a WhatsApp message to all active leads (is_active = TRUE).
    Simply sends to all leads with is_active status = true, no other checks.
    
    """
    try:
        data = await request.json()
        company_name = data.get("company_name")
        message = data.get("message")

        if not company_name:
            raise HTTPException(status_code=400, detail="Missing required field: company_name")
        
        if not message:
            raise HTTPException(status_code=400, detail="Missing required field: message")

        logger.info(f"Starting bulk message send for company: {company_name}")
        logger.info(f"Message: {message[:100]}...")

        # Get all active leads from the leads table
        leads, error = await asyncio.to_thread(get_active_leads_full)
        
        if error:
            logger.error(f"Failed to fetch active leads: {error}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch leads: {error}")

        if not leads:
            return {
                "success": True,
                "message": "No active leads found",
                "company_name": company_name,
                "total_active_leads": 0,
                "total_sent": 0,
                "failed": 0
            }

        logger.info(f"Found {len(leads)} active leads")

        # Prepare tasks for sending messages to ALL active leads
        tasks = []
        phones_for_tasks = []

        for lead in leads:
            phone = lead.get("phone")
            if phone:
                tasks.append(asyncio.to_thread(send_whatsapp_message_v2, phone, message))
                phones_for_tasks.append(phone)
            logger.info(f"Queued message for {phone}")

        # Send messages concurrently
        if tasks:
            logger.info(f"Sending messages to {len(phones_for_tasks)} leads...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = 0
            failed_count = 0
            
            for phone, result in zip(phones_for_tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"[{phone}] Failed to send: {result}")
                    failed_count += 1
                else:
                    logger.info(f"[{phone}] ✅ WhatsApp message sent successfully")
                    success_count += 1

            return {
                "success": True,
                "message": "Bulk message sending completed",
                "company_name": company_name,
                "total_active_leads": len(leads),
                "total_sent": success_count,
                "failed": failed_count
            }
        else:
            return {
                "success": True,
                "message": "No valid phone numbers found in active leads",
                "company_name": company_name,
                "total_active_leads": len(leads),
                "total_sent": 0,
                "failed": 0
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk message sending: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
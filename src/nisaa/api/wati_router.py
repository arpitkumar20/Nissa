"""
WATI WhatsApp Router - FIXED VERSION

✅ Key Changes:
1. Better namespace management (no file reading)
2. Proper error handling for missing namespace
3. More robust webhook validation
4. Better logging and debugging
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.nisaa.graphs.graph import execute_rag_pipeline
from src.nisaa.services.wati_api_service import send_whatsapp_message_v2

logger = logging.getLogger(__name__)

router = APIRouter()


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
    ✅ IMPROVED: Get company namespace from environment or configuration
    
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
    logger.error(f"❌ {error_msg}")
    raise HTTPException(status_code=500, detail=error_msg)


# ============================================================================
# Background Task Handlers
# ============================================================================


async def process_and_send_response(
    phone: str, 
    text: str, 
    request_id: str, 
    company_namespace: str
):
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
        logger.info(f"[{request_id}] Namespace: {company_namespace}")

        # ✅ FIX: Pass namespace to RAG pipeline
        result = await asyncio.to_thread(
            execute_rag_pipeline, 
            user_query=text, 
            user_phone_number=phone, 
            company_namespace=company_namespace
        )

        # Extract response
        agent_reply = result.get("model_response", "")

        if not agent_reply:
            agent_reply = (
                "I apologize, but I couldn't generate a response. Please try again."
            )
            logger.warning(f"[{request_id}] Empty response from RAG pipeline")

        # Log pipeline results
        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Pipeline completed in {processing_time:.2f}s:")
        logger.info(f"  • Search Type: {result.get('search_type', 'unknown')}")
        logger.info(f"  • Documents: {result.get('num_documents', 0)}")
        logger.info(f"  • Used History: {result.get('needs_history', False)}")
        logger.info(f"  • Response Length: {len(agent_reply)} chars")

        if result.get("error"):
            logger.error(f"[{request_id}] Pipeline error: {result['error']}")

        # Send WhatsApp message
        try:
            await asyncio.to_thread(send_whatsapp_message_v2, phone, agent_reply)
            logger.info(f"[{request_id}] ✅ WhatsApp message sent successfully")
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Failed to send WhatsApp message: {e}")
            # Try to send error message
            try:
                error_msg = "I apologize, but I encountered an issue sending the response. Please try again."
                await asyncio.to_thread(send_whatsapp_message_v2, phone, error_msg)
            except:
                pass

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Background task failed: {e}", exc_info=True)

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
    
    ✅ IMPROVED: Better error handling and namespace management

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
        # Parse webhook payload
        data = await request.json()
        logger.info(f"[{request_id}] Received webhook: {data}")

        # Extract and validate required fields
        phone = data.get("waId")
        text = data.get("text")
        event_type = data.get("eventType", "message")
        message_type = data.get("messageType", "text")

        # ✅ FIX: Get namespace from config/env (not from file)
        try:
            company_namespace = get_company_namespace()
        except HTTPException as e:
            logger.error(f"[{request_id}] {e.detail}")
            raise

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
        logger.info(f"  • Phone: {phone}")
        logger.info(f"  • Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        logger.info(f"  • Event Type: {event_type}")
        logger.info(f"  • Message Type: {message_type}")
        logger.info(f"  • Namespace: {company_namespace}")

        # Schedule background processing
        background_tasks.add_task(
            process_and_send_response, 
            phone, 
            text, 
            request_id,
            company_namespace
        )

        # Return immediate acknowledgment
        response_time = time.time() - start_time
        logger.info(f"[{request_id}] Webhook acknowledged in {response_time:.3f}s")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Message received and processing",
                "phone": phone,
                "namespace": company_namespace,
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
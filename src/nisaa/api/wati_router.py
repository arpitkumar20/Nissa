import asyncio
import logging
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
# Background Task Handlers
# ============================================================================


async def process_and_send_response(phone: str, text: str, request_id: str):
    """
    Background task to process query and send WhatsApp response

    Args:
        phone: User's phone number
        text: User's message
        request_id: Unique request identifier for tracking
    """
    start_time = time.time()

    try:
        logger.info(f"[{request_id}] Processing message from {phone}")

        # Execute RAG pipeline
        result = await asyncio.to_thread(
            execute_rag_pipeline, user_query=text, user_phone_number=phone
        )

        # Extract response
        agent_reply = result.get("model_response", "")

        if not agent_reply:
            agent_reply = (
                "I apologize, but I couldn't generate a response. Please try again."
            )
            logger.warning(f"[{request_id}] Empty response from RAG pipeline")

        print(agent_reply)

        # Log pipeline results
        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Pipeline completed in {processing_time:.2f}s:")
        logger.info(f"  • Search Type: {result.get('search_type', 'unknown')}")
        logger.info(f"  • Documents: {result.get('num_documents', 0)}")
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

    Flow:
    1. Receive and validate webhook payload
    2. Schedule RAG pipeline processing (background task)
    3. Return immediate acknowledgment
    4. Process query and send response asynchronously

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

        # Schedule background processing
        background_tasks.add_task(process_and_send_response, phone, text, request_id)

        # Return immediate acknowledgment
        response_time = time.time() - start_time
        logger.info(f"[{request_id}] Webhook acknowledged in {response_time:.3f}s")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Message received and processing",
                "phone": phone,
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
    return {
        "status": "healthy",
        "service": "WATI WhatsApp Router",
        "features": [
            "Agentic RAG",
            "Hybrid Search",
            "Chat History",
            "Background Processing",
        ],
    }
 
import threading
from fastapi import APIRouter, Request

from src.nisaa.helpers.logger import logger
from src.nisaa.graphs.graph import chat_bot_graph
from nisaa.services.wati_api_service import send_whatsapp_message_v2

router = APIRouter()

@router.post("/wati_webhook")
async def wati_webhook(request: Request):
    workflow = chat_bot_graph()
    try:
        data = await request.json()
        message_id = data.get('id')
        phone = data.get('waId')
        text = data.get('text')        
        response = {"status": "received"}

        if not all([message_id, phone, text]):
            return {"error": "Invalid payload"}

        response = workflow.invoke({"user_query": text})
        threading.Thread(target=send_whatsapp_message_v2, args=(phone, response.get('model_response'))).start()
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"error": str(e)}
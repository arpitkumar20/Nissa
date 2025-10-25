# import threading
# from fastapi import APIRouter, Request

# from src.nisaa.helpers.logger import logger
# from src.nisaa.helpers.db import save_message
# from src.nisaa.graphs.graph import chat_bot_graph
# from src.nisaa.helpers.agent_factory import create_chat_agent
# from nisaa.services.wati_api_service import send_whatsapp_message_v2

# router = APIRouter()
# agent = create_chat_agent()

# @router.post("/wati_webhook")
# async def wati_webhook(request: Request):
#     workflow = chat_bot_graph()
#     try:
#         data = await request.json()
#         message_id = data.get('id')
#         phone = data.get('waId')
#         text = data.get('text')        
#         response = {"status": "received"}

#         if not all([message_id, phone, text]):
#             return {"error": "Invalid payload"}

#         config = {"configurable": {"thread_id": str(phone)}}
#         inputs = {"messages": [{"role": "user", "content": text}]}

#         agent_result = agent.invoke(inputs, config)
        
#         if isinstance(agent_result, dict):
#             messages = agent_result.get("messages", [])
#             if messages:
#                 last_msg = messages[-1]
#                 reply = getattr(last_msg, "content", str(last_msg))
#             else:
#                 reply = ""
#         elif hasattr(agent_result, "content"):
#             reply = agent_result.content
#         else:
#             reply = str(agent_result)
        
#         save_message(str(phone), "user", text)
#         save_message(str(phone), "assistant", reply)

#         response = workflow.invoke({"user_query": text}, config)

#         save_message(str(phone), "assistant", response.get("model_response"))
#         threading.Thread(target=send_whatsapp_message_v2, args=(phone, response.get('model_response'))).start()
#         return {"status": "message sent successfully"}
#     except Exception as e:
#         logger.error(f"Webhook error: {e}")
#         return {"error": str(e)}



# src/nisaa/api/wati_router.py
import threading
from fastapi import APIRouter, Request
from src.nisaa.helpers.logger import logger
from src.nisaa.graphs.graph import chat_bot_graph
from src.nisaa.helpers.agent_factory import create_chat_agent
from src.nisaa.helpers.long_term_memory import PostgresMemoryStore
from nisaa.services.wati_api_service import send_whatsapp_message_v2

router = APIRouter()
agent = create_chat_agent(short_term_memory=True)

@router.post("/wati_webhook")
async def wati_webhook(request: Request):
    try:
        data = await request.json()
        message_id = data.get('id')
        phone = data.get('waId')
        text = data.get('text')
        if not all([message_id, phone, text]):
            return {"error": "Invalid payload"}

        # Initialize long-term memory for this thread
        long_memory = PostgresMemoryStore(thread_id=str(phone))
        long_memory.put("user", text)

        # Short-term memory config for agent
        config = {"configurable": {"thread_id": str(phone)}}
        inputs = {"messages": [{"role": "user", "content": text}]}
        agent_result = agent.invoke(inputs, config)

        # Extract agent reply
        if isinstance(agent_result, dict):
            messages = agent_result.get("messages", [])
            reply = messages[-1].content if messages else ""
        elif hasattr(agent_result, "content"):
            reply = agent_result.content
        else:
            reply = str(agent_result)

        # Save assistant reply in long-term memory
        long_memory.put("assistant", reply)

        # Run LangGraph workflow if needed
        workflow = chat_bot_graph()
        response = workflow.invoke({"user_query": text}, config)

        # Save workflow reply in long-term memory
        if response.get("model_response"):
            long_memory.put("assistant", response.get("model_response"))

        # Send WhatsApp message asynchronously
        threading.Thread(
            target=send_whatsapp_message_v2, 
            args=(phone, response.get('model_response'))
        ).start()

        return {"status": "message sent successfully"}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"error": str(e)}
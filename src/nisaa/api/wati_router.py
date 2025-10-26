import threading
from fastapi import APIRouter, Request
from src.nisaa.helpers.logger import logger
from src.nisaa.api.chatbot_router import chat_bot_graph
from src.nisaa.helpers.postgres_store import PostgresMemoryStore
from src.nisaa.helpers.short_term_memory import create_chat_agent
from nisaa.services.wati_api_service import send_whatsapp_message_v2

router = APIRouter()
agent = create_chat_agent()

def store_short_and_long_memory(phone: str, user_text: str, agent_reply: str):
    try:
        long_memory = PostgresMemoryStore(thread_id=str(phone))
        long_memory.put("user", user_text)
        long_memory.put("assistant", agent_reply)

        config = {"configurable": {"thread_id": str(phone)}}
        inputs = {"messages": [{"role": "user", "content": user_text}]}
        agent.invoke(inputs, config)

    except Exception as e:
        logger.error(f"Memory storage error: {e}")


@router.post("/wati_webhook")
async def wati_webhook(request: Request):
    workflow = chat_bot_graph()
    try:
        data = await request.json()
        phone = data.get('waId')
        text = data.get('text')

        if not all([phone, text]):
            return {"error": "Invalid payload"}

        config = {"configurable": {"thread_id": str(phone)}}
        inputs = {"messages": [{"role": "user", "content": text}]}

        agent_result = agent.invoke(inputs, config)
        agent_reply = ""
        if isinstance(agent_result, dict) and "messages" in agent_result:
            messages = agent_result["messages"]
            for msg in reversed(messages):
                if hasattr(msg, "role") and msg.role == "assistant":
                    agent_reply = getattr(msg, "content", "")
                    break
        elif hasattr(agent_result, "content"):
            agent_reply = agent_result.content
        else:
            agent_reply = str(agent_result)
        
        initial_state = {
            "user_query": text,
            "user_phone_number": phone
        }

        response = workflow.invoke(initial_state, config)
        agent_reply = response.get("model_response")
        threading.Thread(
            target=store_short_and_long_memory,
            args=(phone, text, agent_reply)
        ).start()

        threading.Thread(
            target=send_whatsapp_message_v2,
            args=(phone, agent_reply)
        ).start()

        return {"status": "message sent successfully"}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"error": str(e)}
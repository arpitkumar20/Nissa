import json
from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.nisaa.graphs.graph import chat_bot_graph

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        # Accept connection bypassing origin check
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print("WebSocket disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/")
async def websocket_chatbot(websocket: WebSocket):
    await manager.connect(websocket)
    workflow = chat_bot_graph()
    try:
        while True:
            data = await websocket.receive_text()
            # If data is a JSON string, parse it
            if isinstance(data, str):
                try:
                    data_dict = json.loads(data)
                except json.JSONDecodeError:
                    data_dict = {"user_query": data}
            else:
                data_dict = data
            try:
                response = workflow.invoke({"user_query": data_dict.get("user_query", "")})
            except Exception as e:
                response = f"Error processing your message: {str(e)}"
            await manager.send_personal_message(
                json.dumps(
                    {"user_query": response} if isinstance(response, str) else response.get("user_query", {"error": "No response generated."})
                ),
                websocket
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
# File: chat_manager.py
from langchain_core.messages import HumanMessage
from .agent_context import PostgresChatHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
# Note: We are NOT importing the OpenAI client here.

FEW_SHOT_EXAMPLES = [
    HumanMessage(content="Tell me about the cafeteria services."),
    AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="RAG_based_question_answer",
                args={"query": "cafeteria services and hours","mobile_number":"918777745480"},
                id="tool_call_rag_2"
            )
        ]
    ),
    ToolMessage(
        content='{"data": "The main cafeteria is on Floor 2, open 7 AM - 7 PM. A cafe is in the lobby, open 24/7.", "source": "visitor_guide.pdf"}',
        tool_call_id="tool_call_rag_2"
    ),
    
    # --- Example 2: Slot Tool ---
    HumanMessage(content="Are there any appointments for Dr. XYZ who is a ABC specialist next week?"),
    AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="get_next_two_available_days_and_slot",
                args={"doctor_name": "Dr. XYZ","specialty":"ABC"},
                id="tool_call_slot_1"
            )
        ]
    ),
    ToolMessage(
        content='{"doctor": "Dr. XYZ", "availability": {"2025-11-04": ["10:00 AM"], "2025-11-05": ["02:00 PM"]}}',
        tool_call_id="tool_call_slot_1"
    ),

    # --- Example 3: Confirmation Tool ---
    HumanMessage(content="Hi, I need to cancel my appointment. The ID is B-12345."),
    AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="confirmation_status_tool",
                args={"operation": "cancel", "booking_id": "B-12345"},
                id="tool_call_confirm_1"
            )
        ]
    ),
    ToolMessage(
        content='{"status": "Booking B-12345 has been successfully cancelled."}',
        tool_call_id="tool_call_confirm_1"
    )
]

class ChatManager:
    """
    Orchestrates the chat flow using an Agent and a History Manager.
    """
    def __init__(self, agent, history_manager: PostgresChatHistory):
        """
        Initializes the chat manager using Dependency Injection.
        It is "given" the agent and the history manager.
        """
        self.agent = agent
        self.history = history_manager
        self.few_shot_example=FEW_SHOT_EXAMPLES

    def get_response(self, mobile_number: str, user_prompt: str) -> str:
        """
        This is the main function that orchestrates the entire flow.
        """
        print(f"\n--- 1. Fetching history for thread: {mobile_number} ---")
        chat_history = self.few_shot_example+self.history.get_history(mobile_number)
        
        print(f"--- 2. Found {len(chat_history)} messages in context. ---")
        
        # --- DO NOT build full_conversation here ---
        
        print(f"--- 3. Invoking Agent... ---")
        try:
            config = {"configurable": {"mobile_number": "temp-session"}}
            
            # --- FIX 1: Pass history and input separately ---
            response_state = self.agent.invoke(
                {
                    "chat_history": chat_history, # Pass history list
                    "input": user_prompt,
                    "mobile_number":mobile_number          # Pass new user prompt string
                }, 
                config
            )
            
            # --- FIX 2: Get response from the 'output' key ---
            ai_content = response_state['output']
            
            print(f"--- 4. Received response from Agent. ---")

            # 5. Save the new context (this part was already correct)
            self.history.save_message(mobile_number, "user", user_prompt)
            self.history.save_message(mobile_number, "assistant", ai_content)
            
            print(f"--- 5. Saved new context to PostgreSQL. ---")

            return ai_content
            
        except Exception as e:
            print(f"Error during agent invocation: {e}")
            return "Sorry, I encountered an error."
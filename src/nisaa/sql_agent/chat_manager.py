import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.messages.tool import ToolCall

from src.nisaa.sql_agent.agent_context import PostgresChatHistory

logger = logging.getLogger(__name__)

def _get_few_shot_examples(mobile_number: str) -> List[BaseMessage]:
    """
    Returns few-shot examples to guide agent behavior.
    These examples are prepended to actual chat history.

    Returns:
        List[BaseMessage]: List of example messages showing proper tool usage
    """

    few_shot_examples = [
        HumanMessage(content="What are the hospital's visiting hours?"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="RAG_based_question_answer",
                    args={
                        "query": "hospital visiting hours",
                        "mobile_number": mobile_number # Change this number with user's mobile_number while tool calling 
                    },
                    id="call_rag_visiting_hours"
                )
            ]
        ),
        ToolMessage(
            content='"The hospital visiting hours are from 4:00 PM to 6:00 PM for general wards."',
            tool_call_id="call_rag_visiting_hours" # (example tool_call_id)
        ),
        AIMessage(content="The hospital visiting hours are from 4:00 PM to 6:00 PM for general wards.")
        ]
    # (Continuing the list from above)
    few_shot_examples.extend([
        HumanMessage(content="I need to book an appointment with Dr. Priya Sharma, she's a cardiologist."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="get_next_two_available_days_and_slot",
                    args={
                        "doctor_name": "Dr. Priya Sharma",
                        "specialty": "Cardiology"
                    },
                    id="call_get_slots_priya"
                )
            ]
        ),
        ToolMessage(
            content="""{
                "doctor_id": "D123",
                "doctor_name": "Dr. Priya Sharma",
                "qualifications": "MD, Cardiology",
                "slot_duration_mins": 15,
                "start_time": "10:00",
                "end_time": "14:00",
                "next_two_dates": [
                    {"date": "2025-11-12", "weekday": "Wednesday", "slots": [{"token_no": 1, "start_time": "10:00", "end_time": "10:15", "status": "Available"}, ...]},
                    {"date": "2025-11-14", "weekday": "Friday", "slots": [{"token_no": 1, "start_time": "10:00", "end_time": "10:15", "status": "Available"}, ...]}
                ]
            }""",
            tool_call_id="call_get_slots_priya"
        ),
        AIMessage(content="Dr. Priya Sharma is available on Wednesday, November 12th, with slots starting at 10:00 AM, and Friday, November 14th, with slots starting at 10:00 AM. Which date and time would you prefer?"),
        HumanMessage(content="November 12th at 10:00 AM sounds good."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="book_appointment",
                    args={
                        "doctor_name": "Dr. Priya Sharma",
                        "doctor_specialty": "Cardiology",
                        "date": "2025-11-12",
                        "time_slot": "10:00",
                        "mobile_number": mobile_number # Change this number with user's mobile_number while tool calling 
                    },
                    id="call_book_appt_priya"
                )
            ]
        ),
        ToolMessage(
            content='{"doctor": "Dr. Priya Sharma", "date": "2025-11-12", "time": "10:00", "status": "pending(Click Book Now button to confirm)"}',
            tool_call_id="call_book_appt_priya"
        ),
        AIMessage(content="Great, I have a pending appointment for you with Dr. Priya Sharma on November 12, 2025, at 10:00 AM. Please check your WhatsApp for a confirmation message and click the 'Book Now' button to finalize it.")
        ])
    # (Continuing the list from above)
    few_shot_examples.extend([
    HumanMessage(content="I need to cancel my appointment."),
    AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="get_bookings_details",
                args={"mobile_number": mobile_number}, # Change this number with user's mobile_number while tool calling
                id="call_get_bookings_cancel"  # <-- Uses the variable
            )
        ]
    ),
    ToolMessage(
        content='[{"booking_id": "B-1001", "doctor_name": "Dr. Arjun Gupta", "doctor_specialty": "Orthopedics", "booking_date": "2025-11-15", "booking_time": "11:00:00"}]',
        tool_call_id="call_get_bookings_cancel"
    ),
    AIMessage(content="I see you have one appointment with Dr. Arjun Gupta (Orthopedics) on November 15, 2025, at 11:00 AM. Are you sure you want to cancel this?"),
    HumanMessage(content="Yes, please cancel it."),
    AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="cancel_booking_appointment",
                args={
                    "booking_id": "B-1001",
                    "doctor_name": "Dr. Arjun Gupta",
                    "doctor_specialty": "Orthopedics",
                    "dates": "2025-11-15",
                    "time_slot": "11:00:00",
                    "mobile_number": mobile_number # Change this number with user's mobile_number while tool calling 
                },
                id="call_cancel_tool"
            )
        ]
    ),
    ToolMessage(
        content='"User need to press Confirm Update button to finalise the updation in booking.After that User should revisit booking details."', # (Using your tool's exact response)
        tool_call_id="call_cancel_tool"
    ),
    AIMessage(content="Okay, I have started the cancellation for your appointment with Dr. Arjun Gupta. Please check your WhatsApp for a message and click the button to confirm the cancellation.")
    ])
    # (Continuing the list from above)
    few_shot_examples.extend([
        HumanMessage(content="I want to reschedule my appointment."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="get_bookings_details",
                    args={"mobile_number": mobile_number}, # Change this number with user's mobile_number while tool calling # <-- Uses the variable
                    id="call_get_bookings_update"
                )
            ]
        ),
        ToolMessage(
            content='[{"booking_id": "B-2005", "doctor_name": "Dr. Rohan Joshi", "doctor_specialty": "Dermatology", "booking_date": "2025-11-20", "booking_time": "14:00:00"}]',
            tool_call_id="call_get_bookings_update"
        ),
        AIMessage(content="I see you have one appointment with Dr. Rohan Joshi (Dermatology) on November 20, 2025, at 2:00 PM. Would you like to change the date, time, or see a different doctor?"),
        HumanMessage(content="I want to see a different doctor."),
        AIMessage(content="Certainly. What is the name of the new doctor you would like to see?"),
        HumanMessage(content="How about Dr. Amit Singh?"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="RAG_based_question_answer",
                    args={
                        "query": "Is Dr. Amit Singh a valid doctor at this hospital?",
                        "mobile_number": mobile_number # Change this number with user's mobile_number while tool calling 
                    },
                    id="call_rag_validate_amit"
                )
            ]
        ),
        ToolMessage(
            content='"I\'m sorry, but I do not have any information on a Dr. Amit Singh working in this hospital."',
            tool_call_id="call_rag_validate_amit"
        ),
        # This next AIMessage triggers the *next* tool call, as per your prompt's strict flow.
        AIMessage(
            content="I'm sorry, I couldn't find Dr. Amit Singh in our system. Let me find other doctors in the Dermatology department for you.",
            tool_calls=[
                ToolCall(
                    name="RAG_based_question_answer",
                    args={
                        "query": "Who are the other doctors in the Dermatology department?",
                        "mobile_number": mobile_number # Change this number with user's mobile_number while tool calling 
                    },
                    id="call_rag_alternatives"
                )
            ]
        ),
        ToolMessage(
            content='"Other doctors in the Dermatology department include Dr. Meera Nair and Dr. Sanjay Verma."',
            tool_call_id="call_rag_alternatives"
        ),
        AIMessage(content="Other available doctors in the Dermatology department are Dr. Meera Nair and Dr. Sanjay Verma. Would you like to book with one of them?"),
        HumanMessage(content="Let's go with Dr. Meera Nair."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="get_next_two_available_days_and_slot",
                    args={
                        "doctor_name": "Dr. Meera Nair",
                        "specialty": "Dermatology" # (Inferred from original booking)
                    },
                    id="call_get_slots_meera"
                )
            ]
        ),
        ToolMessage(
            content='{"doctor_id": "D456", ..., "next_two_dates": [{"date": "2025-11-21", "weekday": "Friday", "slots": [{"token_no": 1, "start_time": "11:00", ...}]}]}',
            tool_call_id="call_get_slots_meera"
        ),
        AIMessage(content="Dr. Meera Nair is available on Friday, November 21st, with slots starting at 11:00 AM. Would you like to take that?"),
        HumanMessage(content="Yes, please."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="update_booking_appointment",
                    args={
                        "booking_id": "B-2005",
                        "new_doctor_name": "Dr. Meera Nair",
                        "doctor_specialty": "Dermatology",
                        "new_date": "2025-11-21",
                        "new_time_slot": "11:00",
                        "mobile_number": mobile_number # Change this number with user's mobile_number while tool calling 
                    },
                    id="call_update_tool"
                )
            ]
        ),
        ToolMessage(
            content='"User need to press Confirm Update button to finalise the updation in booking.After that User should revisit booking details."',
            tool_call_id="call_update_tool"
        ),
        AIMessage(content="Okay, I've prepared the update. Your original appointment with Dr. Joshi will be changed to Dr. Meera Nair on November 21st at 11:00 AM. Please check your WhatsApp and press the 'Confirm Update' button to finalize this change.")
        ])
    return few_shot_examples
#     [
#     HumanMessage(content="Tell me about the cafeteria services."),
#     AIMessage(
#         content="",
#         tool_calls=[
#             ToolCall(
#                 name="RAG_based_question_answer",
#                 args={"query": "cafeteria services and hours","mobile_number":"91800045480"},
#                 id="tool_call_rag_2"
#             )
#         ]
#     ),
#     ToolMessage(
#         content='{"data": "The main cafeteria is on Floor 2, open 7 AM - 7 PM. A cafe is in the lobby, open 24/7.", "source": "visitor_guide.pdf"}',
#         tool_call_id="tool_call_rag_2"
#     ),
    
#     # --- Example 2: Slot Tool ---
#     HumanMessage(content="Are there any appointments for Dr. XYZ who is a ABC specialist next week?"),
#     AIMessage(
#         content="",
#         tool_calls=[
#             ToolCall(
#                 name="get_next_two_available_days_and_slot",
#                 args={"doctor_name": "Dr. XYZ","specialty":"ABC"},
#                 id="tool_call_slot_1"
#             )
#         ]
#     ),
#     ToolMessage(
#         content='{"doctor": "Dr. XYZ", "availability": {"2025-11-04": ["10:00 AM"], "2025-11-05": ["02:00 PM"]}}',
#         tool_call_id="tool_call_slot_1"
#     ),

#     # --- Example 3: Confirmation Tool ---
#     HumanMessage(content="Hi, I need to cancel my appointment. The ID is B-12345."),
#     AIMessage(
#         content="",
#         tool_calls=[
#             ToolCall(
#                 name="confirmation_status_tool",
#                 args={"operation": "cancel", "booking_id": "B-12345"},
#                 id="tool_call_confirm_1"
#             )
#         ]
#     ),
#     ToolMessage(
#         content='{"status": "Booking B-12345 has been successfully cancelled."}',
#         tool_call_id="tool_call_confirm_1"
#     )
# ]


class ChatManager:
    """
    Orchestrates the chat flow using an Agent and a History Manager.

    Responsibilities:
    - Retrieve chat history from database
    - Prepend few-shot examples to context
    - Invoke agent with full context
    - Save user and assistant messages
    - Handle errors gracefully
    """

    def __init__(self, agent, history_manager: PostgresChatHistory):
        """
        Initializes the chat manager using Dependency Injection.

        Args:
            agent: The LangChain AgentExecutor instance
            history_manager: PostgresChatHistory instance for message persistence
        """
        self.agent = agent
        self.history = history_manager


    def _build_chat_context(self, mobile_number: str) -> List[BaseMessage]:
        """
        Builds the complete chat context by combining few-shot examples
        with actual conversation history.

        Args:
            mobile_number: User's mobile number (thread identifier)

        Returns:
            List[BaseMessage]: Combined message history
        """
        self.few_shot_examples = _get_few_shot_examples(mobile_number)
        user_history = self.history.get_history(mobile_number)
        full_context = self.few_shot_examples + user_history

        logger.info(
            f"Built chat context for {mobile_number}: "
            f"{len(self.few_shot_examples)} examples + {len(user_history)} messages "
            f"= {len(full_context)} total"
        )

        return full_context

    def _save_conversation(
        self, mobile_number: str, user_prompt: str, ai_response: str
    ) -> None:
        """
        Saves both user and assistant messages to history.

        Args:
            mobile_number: User's mobile number
            user_prompt: User's input message
            ai_response: Agent's response message
        """
        try:
            self.history.save_message(mobile_number, "user", user_prompt)
            self.history.save_message(mobile_number, "assistant", ai_response)
            logger.info(f"Saved conversation turn for {mobile_number}")
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}", exc_info=True)
            # Don't raise - message was delivered, just logging failed

    def get_response(self, mobile_number: str, user_prompt: str) -> str:
        """
        Main orchestration method for handling a user message.

        Flow:
        1. Retrieve chat history
        2. Build full context (examples + history)
        3. Invoke agent with context and user input
        4. Extract response
        5. Save conversation to database

        Args:
            mobile_number: User's mobile number (unique identifier)
            user_prompt: User's input message

        Returns:
            str: Agent's response message

        Raises:
            Exception: If agent invocation fails critically
        """
        logger.info(f"Processing message from {mobile_number}")

        # Build context
        chat_context = self._build_chat_context(mobile_number)

        try:
            # Configure agent session
            config = {"configurable": {"mobile_number": mobile_number}}

            # Invoke agent
            logger.debug("Invoking agent...")
            response_state = self.agent.invoke(
                {
                    "chat_history": chat_context,
                    "input": user_prompt,
                    "mobile_number": mobile_number,
                },
                config,
            )

            # Extract response
            ai_content = response_state.get("output", "")

            if not ai_content:
                logger.warning("Agent returned empty response")
                ai_content = (
                    "I apologize, but I couldn't generate a response. Please try again."
                )

            logger.info("Agent invocation successful")

            # Save conversation
            self._save_conversation(mobile_number, user_prompt, ai_content)

            return ai_content

        except KeyError as e:
            logger.error(f"Missing expected key in agent response: {e}", exc_info=True)
            return "Sorry, I encountered an error processing your request. Please try again."

        except Exception as e:
            logger.error(f"Error during agent invocation: {e}", exc_info=True)
            return "Sorry, I encountered an error. Please try again or contact support if the issue persists."


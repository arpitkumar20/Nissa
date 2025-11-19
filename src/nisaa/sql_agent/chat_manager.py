import logging
from typing import List

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.messages.tool import ToolCall

from src.nisaa.sql_agent.agent_context import PostgresChatHistory

logger = logging.getLogger(__name__)

def _get_few_shot_examples(mobile_number: str) -> List[BaseMessage]:
    """
    Returns few-shot examples to guide agent behavior.
    These examples demonstrate the PATTERN of interaction, not real data.
    The agent learns the flow and structure from these examples.

    Returns:
        List[BaseMessage]: List of example messages showing proper tool usage patterns
    """

    few_shot_examples = []

    # ============================================================================
    # EXAMPLE 1: RAG Query (General Information)
    # Purpose: Show how to answer general questions using RAG
    # ============================================================================
    few_shot_examples.extend([
        HumanMessage(content="What are the hospital's visiting hours?"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="RAG_based_question_answer",
                    args={
                        "query": "hospital visiting hours",
                        "mobile_number": mobile_number
                    },
                    id="call_rag_visiting_hours"
                )
            ]
        ),
        ToolMessage(
            content='"The hospital visiting hours are from 4:00 PM to 6:00 PM for general wards."',
            tool_call_id="call_rag_visiting_hours"
        ),
        AIMessage(content="The hospital visiting hours are from 4:00 PM to 6:00 PM for general wards.")
    ])

    # ============================================================================
    # EXAMPLE 2: New Appointment Booking Flow
    # Purpose: Demonstrate complete booking process and that booking_id comes from tool response
    # ============================================================================
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
                    id="call_get_slots_1"
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
                    {
                        "date": "2025-11-12",
                        "weekday": "Wednesday",
                        "slots": [
                            {"token_no": 1, "start_time": "10:00", "end_time": "10:15", "status": "Available"},
                            {"token_no": 2, "start_time": "10:15", "end_time": "10:30", "status": "Available"}
                        ]
                    },
                    {
                        "date": "2025-11-14",
                        "weekday": "Friday",
                        "slots": [
                            {"token_no": 1, "start_time": "10:00", "end_time": "10:15", "status": "Available"}
                        ]
                    }
                ]
            }""",
            tool_call_id="call_get_slots_1"
        ),
        AIMessage(content="Dr. Priya Sharma is available on Wednesday, November 12th, and Friday, November 14th. Both days have slots starting at 10:00 AM. Which date and time would you prefer?"),
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
                        "mobile_number": mobile_number
                    },
                    id="call_book_1"
                )
            ]
        ),
        ToolMessage(
            content='{"booking_id": 31, "doctor": "Dr. Priya Sharma", "date": "2025-11-12", "time": "10:00", "status": "pending(Click Book Now button to confirm)"}',
            tool_call_id="call_book_1"
        ),
        AIMessage(content="Great! I've created a pending appointment (Booking ID: 31) with Dr. Priya Sharma on November 12, 2025, at 10:00 AM. Please check your WhatsApp for a confirmation message and click the 'Book Now' button to finalize your appointment.")
    ])

    # ============================================================================
    # EXAMPLE 3: View Existing Bookings
    # Purpose: Show that booking_id always comes from get_bookings_details
    # ============================================================================
    few_shot_examples.extend([
        HumanMessage(content="Can you show me my appointments?"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="get_bookings_details",
                    args={"mobile_number": mobile_number},
                    id="call_get_bookings_1"
                )
            ]
        ),
        ToolMessage(
            content='[{"booking_id": 31, "doctor_name": "Dr. Priya Sharma", "doctor_specialty": "Cardiology", "booking_date": "2025-11-12", "booking_time": "10:00:00"}, {"booking_id": 101, "doctor_name": "Dr. Arjun Gupta", "doctor_specialty": "Orthopedics", "booking_date": "2025-11-15", "booking_time": "11:00:00"}]',
            tool_call_id="call_get_bookings_1"
        ),
        AIMessage(content="You have 2 appointments:\n\n1. **Booking ID: 31** - Dr. Priya Sharma (Cardiology) on November 12, 2025, at 10:00 AM\n2. **Booking ID: 101** - Dr. Arjun Gupta (Orthopedics) on November 15, 2025, at 11:00 AM")
    ])

    # ============================================================================
    # EXAMPLE 4: Cancel Appointment
    # Purpose: Show that booking_id MUST come from get_bookings_details first
    # ============================================================================
    few_shot_examples.extend([
        HumanMessage(content="I need to cancel my appointment with Dr. Arjun Gupta."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="get_bookings_details",
                    args={"mobile_number": mobile_number},
                    id="call_get_bookings_2"
                )
            ]
        ),
        ToolMessage(
            content='[{"booking_id": 101, "doctor_name": "Dr. Arjun Gupta", "doctor_specialty": "Orthopedics", "booking_date": "2025-11-15", "booking_time": "11:00:00"}]',
            tool_call_id="call_get_bookings_2"
        ),
        AIMessage(content="I found your appointment: **Booking ID: 101** with Dr. Arjun Gupta (Orthopedics) on November 15, 2025, at 11:00 AM. Are you sure you want to cancel this appointment?"),
        HumanMessage(content="Yes, please cancel it."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="cancel_booking_appointment",
                    args={
                        "booking_id": "101",
                        "doctor_name": "Dr. Arjun Gupta",
                        "doctor_specialty": "Orthopedics",
                        "dates": "2025-11-15",
                        "time_slot": "11:00:00",
                        "mobile_number": mobile_number
                    },
                    id="call_cancel_1"
                )
            ]
        ),
        ToolMessage(
            content='"User need to press Confirm Update button to finalise the updation in booking.After that User should revisit booking details."',
            tool_call_id="call_cancel_1"
        ),
        AIMessage(content="I've initiated the cancellation for your appointment (Booking ID: 101) with Dr. Arjun Gupta. Please check your WhatsApp and click the 'Confirm Cancellation' button to complete the process.")
    ])

    # ============================================================================
    # EXAMPLE 5: Reschedule Appointment (Change Date/Time - Same Doctor)
    # Purpose: Show how to get booking_id, then use it for updates
    # ============================================================================
    few_shot_examples.extend([
        HumanMessage(content="I want to reschedule my appointment with Dr. Priya Sharma to a different date."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="get_bookings_details",
                    args={"mobile_number": mobile_number},
                    id="call_get_bookings_3"
                )
            ]
        ),
        ToolMessage(
            content='[{"booking_id": 31, "doctor_name": "Dr. Priya Sharma", "doctor_specialty": "Cardiology", "booking_date": "2025-11-12", "booking_time": "10:00:00"}]',
            tool_call_id="call_get_bookings_3"
        ),
        AIMessage(content="I found your appointment: **Booking ID: 31** with Dr. Priya Sharma (Cardiology) on November 12, 2025, at 10:00 AM. Let me check available slots for rescheduling."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="get_next_two_available_days_and_slot",
                    args={
                        "doctor_name": "Dr. Priya Sharma",
                        "specialty": "Cardiology"
                    },
                    id="call_get_slots_2"
                )
            ]
        ),
        ToolMessage(
            content='{"doctor_id": "D123", "doctor_name": "Dr. Priya Sharma", "next_two_dates": [{"date": "2025-11-14", "weekday": "Friday", "slots": [{"token_no": 1, "start_time": "10:00", "end_time": "10:15", "status": "Available"}, {"token_no": 3, "start_time": "10:30", "end_time": "10:45", "status": "Available"}]}, {"date": "2025-11-16", "weekday": "Sunday", "slots": [{"token_no": 2, "start_time": "11:00", "end_time": "11:15", "status": "Available"}]}]}',
            tool_call_id="call_get_slots_2"
        ),
        AIMessage(content="Dr. Priya Sharma has availability on:\n- Friday, November 14th at 10:00 AM or 10:30 AM\n- Sunday, November 16th at 11:00 AM\n\nWhich slot would you prefer?"),
        HumanMessage(content="Friday at 10:30 AM works better for me."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="update_booking_appointment",
                    args={
                        "booking_id": "31",
                        "new_doctor_name": "Dr. Priya Sharma",
                        "doctor_specialty": "Cardiology",
                        "new_date": "2025-11-14",
                        "new_time_slot": "10:30",
                        "mobile_number": mobile_number
                    },
                    id="call_update_1"
                )
            ]
        ),
        ToolMessage(
            content='"User need to press Confirm Update button to finalise the updation in booking.After that User should revisit booking details."',
            tool_call_id="call_update_1"
        ),
        AIMessage(content="Perfect! I've updated your appointment (Booking ID: 31). Your new appointment with Dr. Priya Sharma is scheduled for Friday, November 14, 2025, at 10:30 AM. Please check your WhatsApp and click the 'Confirm Update' button to finalize this change.")
    ])

    # ============================================================================
    # EXAMPLE 6: Change Doctor (Complex Update Flow with Validation)
    # Purpose: Show doctor validation and alternative suggestions
    # ============================================================================
    few_shot_examples.extend([
        HumanMessage(content="I want to change my dermatology appointment to see a different doctor."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="get_bookings_details",
                    args={"mobile_number": mobile_number},
                    id="call_get_bookings_4"
                )
            ]
        ),
        ToolMessage(
            content='[{"booking_id": 205, "doctor_name": "Dr. Rohan Joshi", "doctor_specialty": "Dermatology", "booking_date": "2025-11-20", "booking_time": "14:00:00"}]',
            tool_call_id="call_get_bookings_4"
        ),
        AIMessage(content="I found your dermatology appointment: **Booking ID: 205** with Dr. Rohan Joshi on November 20, 2025, at 2:00 PM. Which doctor would you like to see instead?"),
        HumanMessage(content="Is Dr. Amit Singh available?"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="RAG_based_question_answer",
                    args={
                        "query": "Is Dr. Amit Singh a dermatologist at this hospital?",
                        "mobile_number": mobile_number
                    },
                    id="call_rag_1"
                )
            ]
        ),
        ToolMessage(
            content='"I\'m sorry, but I do not have any information on a Dr. Amit Singh working in this hospital."',
            tool_call_id="call_rag_1"
        ),
        AIMessage(
            content="I couldn't find Dr. Amit Singh in our system. Let me find other available dermatologists for you.",
            tool_calls=[
                ToolCall(
                    name="RAG_based_question_answer",
                    args={
                        "query": "Who are the dermatologists available at the hospital?",
                        "mobile_number": mobile_number
                    },
                    id="call_rag_2"
                )
            ]
        ),
        ToolMessage(
            content='"Other doctors in the Dermatology department include Dr. Meera Nair and Dr. Sanjay Verma."',
            tool_call_id="call_rag_2"
        ),
        AIMessage(content="We have Dr. Meera Nair and Dr. Sanjay Verma available in Dermatology. Would you like to switch to one of them?"),
        HumanMessage(content="Let's go with Dr. Meera Nair."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="get_next_two_available_days_and_slot",
                    args={
                        "doctor_name": "Dr. Meera Nair",
                        "specialty": "Dermatology"
                    },
                    id="call_get_slots_3"
                )
            ]
        ),
        ToolMessage(
            content='{"doctor_id": "D456", "doctor_name": "Dr. Meera Nair", "next_two_dates": [{"date": "2025-11-21", "weekday": "Friday", "slots": [{"token_no": 1, "start_time": "11:00", "end_time": "11:15", "status": "Available"}]}, {"date": "2025-11-22", "weekday": "Saturday", "slots": [{"token_no": 2, "start_time": "09:00", "end_time": "09:15", "status": "Available"}]}]}',
            tool_call_id="call_get_slots_3"
        ),
        AIMessage(content="Dr. Meera Nair is available on:\n- Friday, November 21st at 11:00 AM\n- Saturday, November 22nd at 9:00 AM\n\nWhich slot would you prefer?"),
        HumanMessage(content="Friday at 11:00 AM works for me."),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="update_booking_appointment",
                    args={
                        "booking_id": "205",
                        "new_doctor_name": "Dr. Meera Nair",
                        "doctor_specialty": "Dermatology",
                        "new_date": "2025-11-21",
                        "new_time_slot": "11:00",
                        "mobile_number": mobile_number
                    },
                    id="call_update_2"
                )
            ]
        ),
        ToolMessage(
            content='"User need to press Confirm Update button to finalise the updation in booking.After that User should revisit booking details."',
            tool_call_id="call_update_2"
        ),
        AIMessage(content="Excellent! I've updated your appointment (Booking ID: 205) from Dr. Rohan Joshi to Dr. Meera Nair on Friday, November 21, 2025, at 11:00 AM. Please check your WhatsApp and click the 'Confirm Update' button to finalize this change.")
    ])

    return few_shot_examples


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
        logger.info("ChatManager initialized with agent and history manager")

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
import re
import psycopg2
import psycopg2.extras
from datetime import date, datetime, timedelta, time
from typing import Optional, Dict, List, Any

from src.nisaa.config.db_connection import get_dict_cursor, get_connection

def build_search_pattern(search_term: str) -> str:
    """
    Builds a SQL LIKE pattern that matches any word combination from search_term.

    Args:
        search_term: The search string (e.g., "Dr. Abdul Moiz")

    Returns:
        SQL LIKE pattern (e.g., "%dr%abdul%moiz%")
    """
    tokens = re.findall(r"\w+", search_term.lower())
    return "%" + "%".join(tokens) + "%"

# Retrieve doctor details by full name
def get_doctor_details_by_name(doctor_name: str) -> List[Dict[str, Any]]:
    """
    Retrieves detailed information for a specific doctor by their full name.

    Args:
        doctor_name: The full name of the doctor to search for

    Returns:
        List of doctor dictionaries with full details, or empty list if not found
    """
    doctor_name = doctor_name.strip().lower()
    search_pattern = f"%{doctor_name}%"

    query = """
        SELECT doctor_id,
            name,
            specialty,
            qualifications,
            days_available,
            location_room,
            start_time,
            end_time,
            slot_duration_mins,
            consultation_fee_inr,
            reception_number
        FROM doctor
        WHERE LOWER(name) LIKE %s;
    """

    try:
        with get_dict_cursor(use_pool=False) as cur:
            cur.execute(query, (search_pattern,))
            results = cur.fetchall()

            if results:
                return [dict(row) for row in results]
            else:
                print(f"[INFO] No doctor found matching name: {doctor_name}")
                return []

    except Exception as e:
        print(f"[ERROR] Failed to fetch doctor details: {e}")
        return []

# Find next two available days and slots for a doctor
def get_next_two_available_days_and_slots(
    doctor_name: str, specialty: str
) -> Optional[Dict[str, Any]]:
    """
    Finds the next two available days and corresponding time slots for a specific doctor.
    Excludes already booked slots.

    Args:
        doctor_name: The full name of the doctor
        specialty: The specialty or department of the doctor

    Returns:
        Dictionary with doctor info and available slots, or None if not found
    """
    conn = None

    try:
        conn = get_connection()

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Find the doctor
            search_pattern = build_search_pattern(doctor_name)
            specialty_pattern = build_search_pattern(specialty)

            cur.execute(
                """
                SELECT doctor_id, name, qualifications, days_available, 
                       start_time, end_time, slot_duration_mins
                FROM doctor
                WHERE LOWER(name) LIKE %s
                  AND LOWER(specialty) LIKE %s        
                LIMIT 1;
            """,
                (search_pattern, specialty_pattern),
            )

            doc = cur.fetchone()
            if not doc:
                print(
                    f"[INFO] No doctor found for name: {doctor_name}, specialty: {specialty}"
                )
                return None

            # Parse available days
            days_str = doc.get("days_available") or ""
            available_days = [
                d.strip().capitalize() for d in days_str.split(",") if d.strip()
            ]

            weekday_to_index = {
                "Monday": 0,
                "Tuesday": 1,
                "Wednesday": 2,
                "Thursday": 3,
                "Friday": 4,
                "Saturday": 5,
                "Sunday": 6,
            }
            available_indices = [
                weekday_to_index[d] for d in available_days if d in weekday_to_index
            ]

            if not available_indices:
                print(f"[INFO] Doctor {doc['name']} has no available days.")
                return None

            # Find next two available dates
            today = date.today()
            found_dates = []
            for offset in range(1, 22):
                candidate = today + timedelta(days=offset)
                if candidate.weekday() in available_indices:
                    found_dates.append(candidate)
                    if len(found_dates) == 2:
                        break

            if not found_dates:
                print(f"[INFO] No upcoming available days found for {doc['name']}")
                return None

            # Parse time objects
            def parse_time_str(tstr):
                return (
                    tstr
                    if isinstance(tstr, time)
                    else datetime.strptime(tstr.strip(), "%H:%M").time()
                )

            start_time_obj = parse_time_str(doc["start_time"])
            end_time_obj = parse_time_str(doc["end_time"])
            slot_duration = int(doc["slot_duration_mins"])

            # Build slots for a day
            def build_slots_for_day(
                start_t: time, end_t: time, duration_mins: int, limit: int = 3
            ):
                slots = []
                anchor = datetime.combine(date.today(), start_t)
                anchor_end = datetime.combine(date.today(), end_t)
                token = 1

                while (
                    anchor + timedelta(minutes=duration_mins) <= anchor_end
                    and len(slots) < limit
                ):
                    slot_start = anchor
                    slot_end = anchor + timedelta(minutes=duration_mins)
                    slots.append(
                        {
                            "token_no": token,
                            "start_time": slot_start.time().strftime("%H:%M"),
                            "end_time": slot_end.time().strftime("%H:%M"),
                            "status": "Available",
                        }
                    )
                    token += 1
                    anchor = slot_end

                return slots

            # Get existing bookings
            cur.execute(
                """
                SELECT booking_date, booking_time
                FROM bookings
                WHERE LOWER(doctor_name) = %s
                  AND booking_date IN %s
                  AND (status IS NULL OR LOWER(status) != 'cancelled');
            """,
                (doctor_name.lower(), tuple(found_dates)),
            )

            bookings = cur.fetchall()

            # Map booked times by date
            booked_by_date = {}
            for b in bookings:
                bdate = b["booking_date"]
                btime = (
                    b["booking_time"].strftime("%H:%M")
                    if isinstance(b["booking_time"], time)
                    else b["booking_time"]
                )
                booked_by_date.setdefault(bdate, set()).add(btime)

            # Build result with availability status
            result_dates = []
            for d in found_dates:
                slots = build_slots_for_day(
                    start_time_obj, end_time_obj, slot_duration, limit=3
                )
                booked_times = booked_by_date.get(d, set())

                for s in slots:
                    if s["start_time"] in booked_times:
                        s["status"] = "Booked"

                result_dates.append(
                    {"date": d.isoformat(), "weekday": d.strftime("%A"), "slots": slots}
                )

            return {
                "doctor_id": doc["doctor_id"],
                "doctor_name": doc["name"],
                "qualifications": doc["qualifications"],
                "slot_duration_mins": slot_duration,
                "start_time": start_time_obj.strftime("%H:%M"),
                "end_time": end_time_obj.strftime("%H:%M"),
                "next_two_dates": result_dates,
            }

    except Exception as e:
        print(f"[ERROR] Failed to fetch doctor availability: {e}")
        return None

    finally:
        if conn:
            conn.close()

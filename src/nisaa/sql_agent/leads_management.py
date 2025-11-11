import psycopg2
import psycopg2.errors
from typing import List, Dict, Tuple, Optional
from ..config.db_connection import get_connection,get_pooled_connection



def insert_leads_from_contacts(contacts: List[Dict]) -> Tuple[int, Optional[str]]:
    """
    Inserts new leads from WATI contact list.
    Skips contacts that already exist (based on phone number).
    
    Args:
        contacts: List of contact dictionaries with keys: wAid, firstName, fullName, phone
    
    Returns:
        Tuple of (number_of_records_inserted, error_message)
    """
    if not contacts:
        return 0, "No contacts to process"
    
    insert_sql = """
    INSERT INTO leads (wa_id, first_name, full_name, phone, is_active)
    VALUES (%s, %s, %s, %s, TRUE)
    ON CONFLICT (phone) DO NOTHING;
    """
    
    inserted_count = 0
    
    try:
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                for contact in contacts:
                    phone = contact.get("phone")
                    if not phone:
                        continue
                    
                    cursor.execute(insert_sql, (
                        contact.get("wAid"),
                        contact.get("firstName"),
                        contact.get("fullName"),
                        phone
                    ))
                    if cursor.rowcount > 0:
                        inserted_count += 1
        
        print(f"Successfully inserted {inserted_count} new contacts into leads table")
        return inserted_count, None
    
    except (Exception, psycopg2.DatabaseError) as e:
        print(f"Error inserting leads: {e}")
        return inserted_count, str(e)


def get_active_leads_full() -> Tuple[Optional[List[dict]], Optional[str]]:
    """
    Retrieves full details of all active leads (is_active = TRUE).

    Returns:
        Tuple of (list_of_leads_as_dicts, error_message)
    """
    select_sql = """
    SELECT id::text, wa_id, first_name, full_name, phone, is_active, 
           created_at::text, updated_at::text
    FROM leads
    WHERE is_active = TRUE
    ORDER BY created_at DESC;
    """

    try:
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_sql)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                leads = [dict(zip(columns, row)) for row in rows]

        print(f"Found {len(leads)} active leads with full info")
        return leads, None

    except (Exception, psycopg2.DatabaseError) as e:
        print(f"Error fetching active leads: {e}")
        return None, str(e)
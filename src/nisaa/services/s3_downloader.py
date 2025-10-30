import os
import boto3
import urllib.parse
import logging

from dotenv import load_dotenv

from src.nisaa.helpers.s3_client import get_s3_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables from .env file (if exists)
load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")


def normalize_filename(name: str) -> str:
    """Lowercase, URL-decode, replace '+' with space, strip, and take basename."""
    if not isinstance(name, str):
        raise TypeError(f"normalize_filename expected str, got {type(name)}")
    
    # Decode URL encoding (handles %20, %28, %29, etc.)
    decoded = urllib.parse.unquote_plus(name)  # unquote_plus handles both %20 AND +
    
    # Return normalized basename
    return os.path.basename(decoded).strip().lower()

def download_all_files_from_s3(file_list: list, company_name: str):
    s3 = get_s3_client()
    allowed_extensions = {"pdf", "txt", "xml", "csv", "docx", "xlsx", "xls", "json"}
    base_directory = "data"
    company_directory = os.path.join(base_directory, company_name)

    if not file_list or len(file_list) == 0:
        logger.warning("No files provided in file_list")
        return []

    # Filter out empty strings
    file_list = [f for f in file_list if f and f.strip()]
    if not file_list:
        logger.warning("file_list contained only empty strings")
        return []
    
    os.makedirs(company_directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {company_directory}")

    downloaded_files = []
    
    try:
        # Build a map of S3 objects by their full key (normalized)
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        all_objects = response.get("Contents", [])
        logger.info(f"Found {len(all_objects)} objects in S3 bucket")

        # Create map with normalized FULL PATHS as keys
        s3_files_map = {
            obj["Key"].strip().lower(): obj["Key"]  # Normalize full path
            for obj in all_objects
        }

        for file_name in file_list:
            # Normalize the incoming full path
            normalized_key = file_name.strip().lower()
            
            # Try exact match first
            s3_key = s3_files_map.get(normalized_key)
            
            if not s3_key:
                # Fallback: try matching by basename only
                requested_basename = normalize_filename(os.path.basename(file_name))
                for full_key in s3_files_map.values():
                    if normalize_filename(os.path.basename(full_key)) == requested_basename:
                        s3_key = full_key
                        logger.info(f"Matched by basename: {file_name} -> {s3_key}")
                        break
            
            if s3_key:
                extension = os.path.splitext(s3_key)[1].lstrip(".").lower()
                if extension in allowed_extensions:
                    # Use only the filename (not full path) for local storage
                    local_filename = os.path.basename(s3_key)
                    file_path = os.path.join(company_directory, local_filename)
                    
                    s3.download_file(BUCKET_NAME, s3_key, file_path)
                    downloaded_files.append(file_path)
                    logger.info(f"✅ Downloaded: {s3_key} -> {file_path}")
                else:
                    logger.warning(f"⚠️ Skipped (invalid extension .{extension}): {s3_key}")
            else:
                logger.error(f"❌ No match found in S3 for: {file_name}")

    except Exception as e:
        logger.error(f'❌ Error while downloading files: {e}', exc_info=True)

    logger.info(f"📦 Total files downloaded: {len(downloaded_files)}")
    return downloaded_files
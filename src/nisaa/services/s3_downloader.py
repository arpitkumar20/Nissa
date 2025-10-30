import os
import boto3
import urllib.parse

AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv('BUCKET_NAME')

def normalize_filename(name: str) -> str:
    """Normalize filenames for matching."""
    return urllib.parse.unquote(name).replace("+", " ").strip().lower()

def download_all_files_from_s3(file_list: list, company_name: str):
    """Download files from S3 bucket."""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    allowed_extensions = {"pdf", "txt", "xml", "csv", "docx", "xlsx", "xls", "json"}
    base_directory = "data"
    company_directory = os.path.join(base_directory, company_name)
    os.makedirs(company_directory, exist_ok=True)
    print(f"Ensured directory exists: {company_directory}")

    downloaded_files = []

    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        all_objects = response.get("Contents", [])

        s3_files_map = {
            normalize_filename(os.path.basename(obj["Key"])): obj["Key"]
            for obj in all_objects
        }

        for file_name in file_list:
            normalized_name = normalize_filename(os.path.basename(file_name))
            s3_key = s3_files_map.get(normalized_name)

            if s3_key:
                extension = os.path.splitext(s3_key)[1].lstrip(".").lower()
                if extension in allowed_extensions:
                    file_path = os.path.join(company_directory, os.path.basename(s3_key))
                    s3.download_file(BUCKET_NAME, s3_key, file_path)
                    downloaded_files.append(file_path)
                    print(f"Downloaded: {s3_key} → {file_path}")
                else:
                    print(f"Skipped (invalid extension): {s3_key}")
            else:
                print(f"No match found for: {file_name}")

    except Exception as e:
        print(f"Error while downloading files: {e}")
        raise e

    return downloaded_files 
import requests
import pandas as pd
from pathlib import Path
from src.nisaa.helpers.logger import logger

zh_login_url = 'https://accounts.zoho.in'
zh_service_url = "https://www.zohoapis.in"
zh_application_url = "https://creator.zoho.in"
zh_login_grant = 'refresh_token'

class Zoho:
    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        try:
            response = requests.post(zh_login_url + '/oauth/v2/token', data={
                'refresh_token': refresh_token,
                'client_id': client_id,
                'client_secret': client_secret,
                'grant_type': zh_login_grant,
            })

            if response.status_code != 200:
                raise Exception('Zoho login unsuccessful.')

            self.__class__.zh_login_token = response.json().get('access_token')
            logger.info({'message': 'Zoho connection established.'})
        except Exception as error:
            logger.error({'message': 'Zoho connection error occurred', 'error': error})

    def connection_test(self):
        try:
            if self.__class__.zh_login_token is None or zh_service_url is None:
                raise Exception('Zoho Connection error occurred.')
            
            return {'message': 'Zoho connection established.'}, 200
        except Exception as error:
            logger.error({'message': 'Zoho login failed with an error', 'error': error})
            return {'message': 'Zoho login failed with an error.', 'description': str(error)}, 401
    

    def get_all_applications(self):
        try:
            url = f"{zh_service_url}/creator/v2.1/meta/applications"
            headers = {
                "Authorization": f"Zoho-oauthtoken {self.__class__.zh_login_token}"
            }
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                print("Error fetching applications:", response.text)
                return None

            data = response.json()
            apps = data.get("applications", [])

            result = []
            for app in apps:
                result.append({
                    "owner_name": app.get("workspace_name") or app.get("created_by"),
                    "app_link_name": app.get("link_name"),
                })
            return result

        except Exception as e:
            logger.error({'message': 'Error getting applications', 'error': e})
            return {'message': 'Error getting applications', "description": str(e)}, 500

    def fetch_reports_list(self, owner_name: str, app_link_name: str):
        try:
            url = f"{zh_service_url}/creator/v2.1/meta/{owner_name}/{app_link_name}/reports"
            headers = {
                "Authorization": f"Zoho-oauthtoken {self.__class__.zh_login_token}"
            }

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                reports = data.get("reports", [])
                
                result = []
                for rpt in reports:
                    result.append({
                        "owner_name": owner_name,
                        "app_link_name": app_link_name,
                        "report_link_name": rpt.get("link_name"),
                        "report_display_name": rpt.get("display_name"),
                        "type": rpt.get("type")
                    })

                logger.info("Reports metadata fetched successfully")
                return {"success": True, "reports": result}, 200
            else:
                logger.error(f"Error {response.status_code}: {response.text}")
                return {"success": False, "message": response.text}, response.status_code

        except Exception as error:
            logger.error("Zoho Creator report metadata fetch error", exc_info=True)
            return {"success": False, "message": str(error)}, 500

    def fetch_report_deatils(self, owner_name: str, app_link_name: str, report_link_name: str):
        try:
            url = f"{zh_application_url}/api/v2/{owner_name}/{app_link_name}/report/{report_link_name}"
            headers = {
                "Authorization": f"Zoho-oauthtoken {self.__class__.zh_login_token}"
            }
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                logger.info("Report data fetched successfully")
                # Extract 'data' if present in JSON, else assume response is a list
                records = data.get('data') if isinstance(data, dict) else data

                if isinstance(records, list) and len(records) > 0 and isinstance(records[0], dict):
                    # Create directory if not exists
                    folder_path = Path("zoho_details")
                    folder_path.mkdir(exist_ok=True)

                    # Create DataFrame from records
                    df = pd.DataFrame(records)

                    # Save to CSV with report_link_name as filename
                    csv_file_path = folder_path / f"{report_link_name}.csv"
                    df.to_csv(csv_file_path, index=False, encoding='utf-8')

                    logger.info(f"Report data stored successfully in CSV: {csv_file_path}")

                else:
                    logger.warning(f"No valid records found for report: {report_link_name}. Skipping CSV creation.")
                return {'operation_type': 'report', 'records': data}, 200
            else:
                logger.error(f"Error {response.status_code}: {response.text}")
                return {'error': True, 'message': f"Report fetch failed with status {response.status_code}", 'description': response.text}, response.status_code
        except Exception as error:
            logger.error({'message': 'Zoho Creator report fetch error occurred', 'error': error})
            return {'error': True, 'message': 'Zoho Creator report fetch error occurred.', 'description': str(error)}, 401
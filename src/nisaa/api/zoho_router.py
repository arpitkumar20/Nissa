from fastapi import APIRouter, Request, HTTPException

from src.nisaa.helpers.logger import logger
from src.nisaa.services.zoho_connector import Zoho

router = APIRouter(prefix="/zoho", tags=["Zoho"])

@router.post("/fetch_report_details")
async def fetch_zoho_report(request: Request):
    try:
        data = await request.json()
        zoho = Zoho(
            client_id=data.get('client_id'),
            client_secret=data.get('client_secret'),
            refresh_token=data.get('refresh_token'),
        )

        test_response, code = zoho.connection_test()
        if code != 200:
            raise HTTPException(status_code=401, detail=test_response)

        applications = zoho.get_all_applications()
        if not applications or not isinstance(applications, list):
            raise HTTPException(status_code=404, detail="No applications found in Zoho account")

        logger.info(f"Applications found: {[a.get('app_link_name') for a in applications]}")

        selected_app = next(
            (a for a in applications if a.get("app_link_name") == data.get('app_link_name')), None
        )
        if not selected_app:
            raise HTTPException(
                status_code=404,
                detail=f"Application '{data.get('app_link_name')}' not found in Zoho account"
            )

        owner_name = selected_app.get("owner_name")
        app_link_name = selected_app.get("app_link_name")

        reports_resp, status_code = zoho.fetch_reports_list(
            owner_name=owner_name,
            app_link_name=app_link_name
        )

        if status_code != 200 or not reports_resp.get("success"):
            raise HTTPException(
                status_code=404,
                detail=f"Failed to fetch reports for app '{app_link_name}'"
            )

        reports = reports_resp.get("reports", [])
        if not isinstance(reports, list) or not reports:
            logger.info(f"No valid reports found for app '{app_link_name}'")
            return {
                "success": True,
                "app_link_name": app_link_name,
                "total_reports": 0,
                "reports": []
            }

        all_report_data = []

        for report in reports:
            link_name = report.get("report_link_name")
            if not link_name:
                continue

            report_data, status = zoho.fetch_report_deatils(
                owner_name=owner_name,
                app_link_name=app_link_name,
                report_link_name=link_name
            )

            if status != 200 or not report_data:
                continue

            all_report_data.append({
                "report_metadata": report,
                "report_data": report_data
            })

        logger.info(f"Fetched {len(all_report_data)} reports for app '{app_link_name}'")

        return {
            "success": True,
            "app_link_name": app_link_name,
            "total_reports": len(all_report_data),
            "reports": all_report_data
        }
    except HTTPException as e:
        raise e
    except Exception as err:
        logger.error(f"Zoho report fetch failed: {err}")
        raise HTTPException(
            status_code=500,
            detail=f"Zoho report fetch failed: {err}"
        )
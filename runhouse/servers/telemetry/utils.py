from runhouse.logger import logger


def send_status_to_den_succeeded(send_to_den_resp: dict):
    send_to_den_status_code = send_to_den_resp.get("status_code")

    if send_to_den_status_code == 404:
        logger.info(
            "Cluster has not yet been saved to Den, cannot update status or logs."
        )

    elif send_to_den_status_code != 200:
        logger.warning(
            f"{send_to_den_status_code}: Failed to send cluster status to den. Please, check cluster logs for more info."
        )

    return send_to_den_status_code == 200

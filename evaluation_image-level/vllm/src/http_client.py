import logging
import time

import requests


def wait_for_server_ready(
    base_url: str,
    headers: dict,
    max_retries: int,
    retry_interval: int,
    timeout: int,
    logger: logging.Logger,
) -> bool:
    ping_url = base_url.rstrip("/") + "/health"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url=ping_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            logger.info("Server is ready.")
            return True
        except requests.exceptions.Timeout:
            logger.warning(
                f"Server not ready ({attempt}/{max_retries}): request timed out after {timeout}s"
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Server not ready ({attempt}/{max_retries}): {e}")
        if attempt < max_retries:
            time.sleep(retry_interval)

    logger.error("Server did not respond after max retries.")
    return False


def send_inference_request(
    endpoint: str,
    headers: dict,
    payload: dict,
    logger: logging.Logger,
) -> tuple[str, str] | None:
    try:
        response = requests.post(
            url=endpoint, headers=headers, json=payload, timeout=None
        )
        response.raise_for_status()
        completion = response.json()
        choice = completion["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "unknown")
        return content, finish_reason
    except Exception as e:
        logger.warning(f"Request failed: {e}")
    return None

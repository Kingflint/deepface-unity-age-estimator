"""HTTP webhook dispatcher with exponential-backoff retry."""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, Optional

from .models import DeliveryAttempt, WebhookEvent
from .signing import SIGNATURE_HEADER, TIMESTAMP_HEADER, compute_signature

logger = logging.getLogger("deepface_server.webhooks")

HttpClient = Callable[..., object]


class WebhookDispatcher:
    """Sends :class:`WebhookEvent` payloads to configured endpoints."""

    def __init__(
        self,
        endpoints: Iterable[str],
        secret: str = "",
        *,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        backoff_factor: float = 2.0,
        timeout: float = 5.0,
        http_client: Optional[HttpClient] = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.endpoints = list(endpoints)
        self.secret = secret
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self._http_client = http_client
        self._sleeper = sleeper

    def dispatch(self, event: WebhookEvent) -> list[DeliveryAttempt]:
        attempts: list[DeliveryAttempt] = []
        for url in self.endpoints:
            attempts.extend(self._deliver(url, event))
        return attempts

    def _deliver(self, url: str, event: WebhookEvent) -> list[DeliveryAttempt]:
        results: list[DeliveryAttempt] = []
        body = event.to_json()
        timestamp = event.created_at
        signature = compute_signature(self.secret, body, timestamp)
        headers = {
            "Content-Type": "application/json",
            SIGNATURE_HEADER: signature,
            TIMESTAMP_HEADER: timestamp,
            "X-Event-Id": event.id,
            "X-Event-Type": event.type,
        }
        delay = self.backoff_seconds
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                status = self._send(url, body, headers)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                result = DeliveryAttempt(
                    event_id=event.id,
                    url=url,
                    status_code=status,
                    duration_ms=elapsed_ms,
                    attempt=attempt,
                )
            except Exception as exc:  # noqa: BLE001 - network failure
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                result = DeliveryAttempt(
                    event_id=event.id,
                    url=url,
                    error=str(exc),
                    duration_ms=elapsed_ms,
                    attempt=attempt,
                )
            results.append(result)
            if result.succeeded:
                return results
            if attempt < self.max_attempts:
                result.next_retry_in_seconds = delay
                self._sleeper(delay)
                delay *= self.backoff_factor
        return results

    def _send(self, url: str, body: str, headers: dict[str, str]) -> int:
        if self._http_client is not None:
            return int(self._http_client(url=url, data=body, headers=headers, timeout=self.timeout))
        try:
            from urllib import request as urlrequest

            req = urlrequest.Request(
                url, data=body.encode("utf-8"), headers=headers, method="POST"
            )
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:  # nosec - opt-in webhook
                return int(resp.status)
        except Exception:
            raise

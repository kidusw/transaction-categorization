"""Small typed wrapper around the txcat HTTP API, used by the dashboard."""

import os

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        http: httpx.Client | None = None,
    ):
        """`http` lets tests pass a FastAPI TestClient (itself an httpx.Client)."""
        # Read at call time (not import time) so TXCAT_API_URL set by a container/test applies.
        base_url = base_url or os.environ.get("TXCAT_API_URL", DEFAULT_API_URL)
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"cannot reach the API at {self.base_url} ({exc})") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"{method} {path} failed ({response.status_code}): {detail}")
        return response.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def model(self) -> dict:
        return self._request("GET", "/model")

    def accounts(self) -> list[dict]:
        return self._request("GET", "/accounts")

    def insights(self, account_id: str, months: int = 12, anomalies: int = 20) -> dict:
        params = {"months": months, "anomalies": anomalies}
        return self._request("GET", f"/accounts/{account_id}/insights", params=params)

    def categorize(self, rows: list[dict]) -> dict:
        return self._request("POST", "/categorize", json={"transactions": rows})

    def review_queue(self, status: str = "pending", limit: int = 50) -> list[dict]:
        return self._request("GET", "/review", params={"status": status, "limit": limit})

    def review(self, item_id: int, category: str) -> dict:
        return self._request("POST", f"/review/{item_id}", json={"category": category})

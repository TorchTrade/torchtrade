"""FXMacroData macro-event transform."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tensordict import NonTensorData, TensorDictBase
from torchrl.envs.transforms import Transform

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


def fetch_fxmacrodata_event_dates(
    currency: str = "usd",
    *,
    limit: int = 100,
    min_tier: int | None = 1,
    api_key: str | None = None,
) -> set[str]:
    """Fetch release dates from FXMacroData for observation features."""

    limit_count = max(1, min(int(limit), 100))
    params: dict[str, str] = {"limit": str(limit_count)}
    token = api_key or os.getenv("FXMACRODATA_API_KEY")
    if token:
        params["api_key"] = token

    url = f"{FXMACRODATA_BASE_URL}/calendar/{currency.lower()}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "torchtrade-fxmacrodata/1.0"})
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)

    events = payload.get("data", [])
    if min_tier is not None:
        events = [
            event
            for event in events
            if int(event.get("market_tier") or 99) <= min_tier
        ]
    return {event["date"] for event in events[:limit_count] if event.get("date")}


class FXMacroDataEventTransform(Transform):
    """Add a boolean macro-event-day flag to each step/reset."""

    def __init__(
        self,
        event_dates: set[str],
        *,
        out_key: str = "is_macro_event_day",
    ):
        super().__init__(in_keys=[], out_keys=[])
        self._event_dates = event_dates
        self._out_key = out_key

    @classmethod
    def from_fxmacrodata(
        cls,
        currency: str = "usd",
        *,
        limit: int = 100,
        min_tier: int | None = 1,
        out_key: str = "is_macro_event_day",
    ) -> "FXMacroDataEventTransform":
        """Build a transform using FXMacroData release-calendar dates."""

        return cls(
            fetch_fxmacrodata_event_dates(
                currency=currency,
                limit=limit,
                min_tier=min_tier,
            ),
            out_key=out_key,
        )

    def _flag(self) -> NonTensorData:
        today = datetime.now(timezone.utc).date().isoformat()
        return NonTensorData(today in self._event_dates)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset.set(self._out_key, self._flag())
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        next_tensordict.set(self._out_key, self._flag())
        return next_tensordict

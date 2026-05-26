from __future__ import annotations

import aiohttp


class Embedder:
    """OpenAI-compatible embedding API client."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        timeout: int = 30,
    ) -> None:
        self._url = api_url.rstrip("/") + "/v1/embeddings"
        self._api_key = api_key
        self._model = model
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def embed(self, text: str) -> list[float]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {"model": self._model, "input": text}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["data"][0]["embedding"]

import os
import time
import requests
from typing import Optional


class NIMClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or os.environ.get(
            "NIM_BASE_URL",
            "https://integrate.api.nvidia.com/v1"
        )
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")

        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set.")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 6,
    ) -> str:
        temp = self.temperature if temperature is None else temperature
        mtok = self.max_tokens if max_tokens is None else max_tokens

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []

        if system:
            messages.append({
                "role": "system",
                "content": system
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": mtok,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=180
                )

                if response.status_code == 429:
                    wait = 2 ** attempt
                    print(f"Rate limited. Retrying in {wait} seconds.")
                    time.sleep(wait)
                    continue
                if response.status_code in [500, 502, 503, 504]:
                    wait = 2 ** attempt
                    print(f"NIM server error {response.status_code}. Retrying in {wait} seconds.")
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    raise RuntimeError(
                        f"NIM error {response.status_code}: {response.text}"
                    )

                data = response.json()
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as error:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"Request failed. Retrying in {wait} seconds.")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"NIM request failed: {error}")

        raise RuntimeError("NIM request failed after retries.")
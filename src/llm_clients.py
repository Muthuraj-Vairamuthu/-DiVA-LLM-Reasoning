import os
import random
import time
from typing import Optional

import requests


class APIChatClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        *,
        provider_name: str,
        base_url: str,
        api_key_env: str,
        max_tokens_field: str = "max_tokens",
        extra_payload: Optional[dict] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_name = provider_name
        self.base_url = base_url
        self.api_key = os.environ.get(api_key_env)
        self.max_tokens_field = max_tokens_field
        self.extra_payload = extra_payload or {}

        self.timeout_seconds = int(os.environ.get("LLM_TIMEOUT_SECONDS", "240"))
        self.default_max_retries = int(os.environ.get("LLM_MAX_RETRIES", "10"))
        self.backoff_base = float(os.environ.get("LLM_BACKOFF_BASE", "1"))
        self.backoff_max = float(os.environ.get("LLM_BACKOFF_MAX", "90"))
        self.backoff_jitter = float(os.environ.get("LLM_BACKOFF_JITTER", "0.5"))

        if not self.api_key:
            raise RuntimeError(f"{api_key_env} is not set.")

    def _retry_wait(self, attempt: int) -> float:
        base_wait = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        jitter = random.uniform(0, self.backoff_jitter)
        return base_wait + jitter

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        temp = self.temperature if temperature is None else temperature
        mtok = self.max_tokens if max_tokens is None else max_tokens
        retries = self.default_max_retries if max_retries is None else max_retries

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
            self.max_tokens_field: mtok,
        }
        payload.update(self.extra_payload)

        for attempt in range(retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds
                )

                if response.status_code == 429:
                    wait = self._retry_wait(attempt)
                    print(f"{self.provider_name} rate limited. Retrying in {wait:.1f} seconds.", flush=True)
                    time.sleep(wait)
                    continue

                if response.status_code in [500, 502, 503, 504]:
                    wait = self._retry_wait(attempt)
                    print(
                        f"{self.provider_name} server error {response.status_code}. Retrying in {wait:.1f} seconds.",
                        flush=True
                    )
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    raise RuntimeError(
                        f"{self.provider_name} error {response.status_code}: {response.text}"
                    )

                data = response.json()
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as error:
                if attempt < retries - 1:
                    wait = self._retry_wait(attempt)
                    print(f"{self.provider_name} request failed. Retrying in {wait:.1f} seconds.", flush=True)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"{self.provider_name} request failed: {error}")

        raise RuntimeError(f"{self.provider_name} request failed after retries.")


class NIMClient(APIChatClient):
    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if api_key is not None:
            os.environ["NVIDIA_API_KEY"] = api_key

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_name="NIM",
            base_url=base_url or os.environ.get(
                "NIM_BASE_URL",
                "https://integrate.api.nvidia.com/v1/chat/completions"
            ),
            api_key_env="NVIDIA_API_KEY",
            max_tokens_field="max_tokens",
        )


class GroqClient(APIChatClient):
    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if api_key is not None:
            os.environ["GROQ_API_KEY"] = api_key

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_name="Groq",
            base_url=base_url or os.environ.get(
                "GROQ_BASE_URL",
                "https://api.groq.com/openai/v1/chat/completions"
            ),
            api_key_env="GROQ_API_KEY",
            max_tokens_field="max_completion_tokens",
            extra_payload={
                "top_p": float(os.environ.get("GROQ_TOP_P", "1")),
                "reasoning_effort": os.environ.get("GROQ_REASONING_EFFORT", "medium"),
                "stream": False,
                "stop": None,
            },
        )


def get_llm_client(
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 512,
):
    provider = os.environ.get("LLM_PROVIDER", "nim").strip().lower()

    if provider == "groq":
        return GroqClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return NIMClient(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

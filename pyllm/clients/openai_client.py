import os
import requests
import json

from typing import Optional
from dataclasses import asdict

from pyllm.clients import Client
from pyllm.types import SamplingParams


class OpenAIChatClient(Client):
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        org_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        if api_key is not None:
            self.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            self.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise KeyError(
                "No OpenAI API key provided. Make sure to either pass it to the constructor or save it as an environment variable with the name OPENAI_API_KEY"
            )
        self.model_name = model_name
        self.org_id = org_id
        self.base_url = base_url

    def query(self, prompt, params: SamplingParams = SamplingParams()) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.org_id:
            headers[""]
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **asdict(params),
        }

        res = requests.post(self.base_url, json=body, headers=headers)

        if res.status_code != 200:
            raise requests.RequestException(res.content)

        return json.loads(res.content)["choices"][0]["message"]["content"]

import os
import requests
import json

from typing import Dict, Optional
from dataclasses import asdict

from pyllm.clients import Client
from pyllm.utils.types import SamplingParams
from pyllm.utils.registry import CLIENT_REGISTRY


@CLIENT_REGISTRY.register("azure")
class AzureChatClient(Client):
    """
    A client for querying an OpenAI API endpoint hosted on Azure, specifically designed for chat completions.

    This client handles the setup and execution of requests to an OpenAI API, allowing for
    easy querying of the model for chat-based completions.

    https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
    """

    def __init__(
        self,
        url: Optional[str] = None,
        resource_name: Optional[str] = None,
        deployment_id: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if api_key is not None:
            self.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            self.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise KeyError(
                "No OpenAI API key provided. Make sure to either pass it to the constructor or save it as an environment variable with the name OPENAI_API_KEY"
            )

        if url:
            self.url = url
        else:
            if (
                sum([bool(arg) for arg in [resource_name, deployment_id, api_version]])
                < 3
            ):
                raise ValueError(
                    "You must pass all of resource_name, deployment_id, and api_version when no URL is provided"
                )

            self.url = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}"

        self.model_name = self.url

    def query(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Dict] = None,
        sampling_params: SamplingParams = SamplingParams(),
    ) -> str:
        """
        Queries the OpenAI API endpoint with a given prompt and sampling parameters.

        Args:
            prompt (str): The prompt to send to the model.
            params (SamplingParams): An instance of SamplingParams specifying parameters
                for the query, such as temperature, max tokens, etc.

        Returns:
            str: The content of the message returned by the model as a response to the query.

        Raises:
            requests.RequestException: If the request to the OpenAI API fails or returns a
                non-200 status code, with the response content included in the exception message.
        """
        if (not prompt and not messages) or (prompt and messages):
            raise ValueError("Pass either a string prompt or messages dict")

        headers = {"api-key": f"{self.api_key}"}

        body = {
            "messages": messages if messages else [{"role": "user", "content": prompt}],
            **asdict(sampling_params),
        }

        res = requests.post(self.url, json=body, headers=headers)

        if res.status_code != 200:
            raise requests.RequestException(res.content)

        return json.loads(res.content)["choices"][0]["message"]["content"]

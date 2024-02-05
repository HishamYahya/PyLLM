import os
import requests
import json

from typing import Optional
from dataclasses import asdict

from pyllm.clients import Client
from pyllm.types import SamplingParams


class OpenAIChatClient(Client):
    """
    A client for querying an OpenAI API endpoint, specifically designed for chat completions.

    This client handles the setup and execution of requests to an OpenAI API, allowing for
    easy querying of the model for chat-based completions.

    Attributes:
        api_key (str): The API key used for authentication with the server.
        model_name (str): The name of the model to query. Defaults to 'gpt-3.5-turbo'.
        org_id (Optional[str]): The organization ID for OpenAI, if applicable.
        base_url (str): The base URL for the OpenAI chat completions API endpoint.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        base_url: str = "https://api.openai.com",
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
    ):
        """
        Initializes the OpenAIChatClient with API key, model name, organization ID, and base URL.

        The API key can be provided directly or set as an environment variable.

        Args:
            model_name (str): The name of the model to be queried. Defaults to 'gpt-3.5-turbo'.
            base_url (str): The base URL for the chat completions API endpoint. Defaults to
                the standard OpenAI chat completions endpoint.
            api_key (Optional[str]): Optional API key for authentication. If not provided,
                attempts to retrieve it from the environment variable OPENAI_API_KEY.
            org_id (Optional[str]): Optional organization ID for usage with OpenAI's API.

        Raises:
            KeyError: If no API key is provided directly or found in the environment variables.
        """
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
        self.base_url = base_url if base_url[-1] == "/" else base_url + "/"

    def query(self, prompt, sampling_params: SamplingParams = SamplingParams()) -> str:
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
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.org_id:
            headers[""]
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **asdict(sampling_params),
        }

        completions_url = self.base_url + "v1/chat/completions"
        res = requests.post(completions_url, json=body, headers=headers)

        if res.status_code != 200:
            raise requests.RequestException(res.content)

        return json.loads(res.content)["choices"][0]["message"]["content"]

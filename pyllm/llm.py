import json
import os

from dataclasses import asdict
from typing import Optional, List, Tuple, Callable
from random import randint
from filelock import FileLock
from requests import RequestException
from appdirs import user_cache_dir

from pyllm.clients import Client, OpenAIChatClient
from pyllm.parsers import ParserBase, RegExParser
from pyllm.templates import PromptTemplate
from pyllm.types import SamplingParams, Function
from pyllm.exceptions import TooManyRetries


os.makedirs(user_cache_dir("PyLLM"), exist_ok=True)


class CacheHandler:
    CACHE_FILE = os.path.join(user_cache_dir("PyLLM"), "cached_functions.json")

    def __init__(self, mode: str = "r"):
        self.mode = mode

    def __enter__(self):
        self.lock = FileLock(f"{self.CACHE_FILE}.lock")
        self.lock.acquire()
        try:
            if not os.path.exists(self.CACHE_FILE):
                with open(self.CACHE_FILE, "w") as f:
                    json.dump({}, f)
            else:
                try:
                    with open(self.CACHE_FILE, "r") as f:
                        json.load(f)
                # If the file is corrupted, empty it
                except json.JSONDecodeError:
                    with open(self.CACHE_FILE, "w") as f:
                        json.dump({}, f)
        finally:
            self.lock.release()

        self.file = open(self.CACHE_FILE, self.mode)
        return self.file

    def __exit__(self, *_):
        self.file.close()
        self.lock.release()


class CodeLLM:
    def __init__(
        self,
        client: Client = OpenAIChatClient(),
        parser: ParserBase = RegExParser(),
        prompt_template: PromptTemplate = PromptTemplate(),
    ):
        self.client = client
        self.parser = parser
        self.prompt_template = prompt_template

    def _unit_test(self, function: Callable, unit_tests: List[Tuple], *args):
        for x, y in unit_tests:
            if type(x) is tuple:
                assert function(*x) == y
            else:
                assert function(x) == y

    def def_function(
        self,
        prompt: str,
        input_types: Optional[List] = None,
        output_types: Optional[List] = None,
        unit_tests: Optional[List[Tuple]] = None,
        use_cached: bool = True,
        n_retries: int = 1,
        sampling_params: SamplingParams = SamplingParams(),
    ) -> Function:
        model_response = None
        # If cached, read it
        if use_cached:
            with CacheHandler("r") as f:
                cache = json.load(f)
                if prompt in cache:
                    model_response = cache[prompt]["model_response"]
                    sampling_params = cache[prompt]["sampling_params"]
                    function = self.parser.parse_function(model_response)

        # Query the model if not cached
        if model_response is None:
            formatted_prompt = self.prompt_template.apply(
                prompt=prompt,
                object_type="function",
                input_types=input_types,
                output_types=output_types,
                unit_tests=unit_tests,
            )
            seed = randint(0, 2 ** 62)
            sampling_params.seed = seed
            for _ in range(n_retries):
                try:

                    model_response = self.client.query(
                        formatted_prompt, params=sampling_params
                    )
                    sampling_params = asdict(sampling_params)
                except RequestException:
                    # retry if server fails to give 200 response
                    continue

                try:
                    function = self.parser.parse_function(model_response)
                except SyntaxError:
                    # retry if parsing fails
                    continue

                try:
                    if unit_tests:
                        self._unit_test(function, unit_tests)
                except AssertionError:
                    # retry if any unit test fails
                    continue

                # Break when code passes all tests
                break

            else:
                raise TooManyRetries(f"{n_retries=} exceeded.")

        # Cache the response
        with CacheHandler("r") as f:
            new_cache = json.load(f)

        new_cache[prompt] = {
            "model_response": model_response,
            "sampling_params": sampling_params,
            "parser": self.parser.__class__.__name__,
        }

        with CacheHandler("w") as f:
            json.dump(new_cache, f)

        return Function(
            function=function,
            source=model_response,
            model_name=self.client.model_name,
            sampling_params=sampling_params,
        )

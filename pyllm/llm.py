import json
import os
import logging

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
        client: Optional[Client] = None,
        parser: Optional[ParserBase] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        if client is None:
            client = OpenAIChatClient()
        self.client = client

        if parser is None:
            parser = RegExParser()
        self.parser = parser

        if prompt_template is None:
            prompt_template = PromptTemplate()
        self.prompt_template = prompt_template

    def _unit_test(self, function: Callable, unit_tests: List[Tuple], *args):
        for x, y in unit_tests:
            if type(x) is tuple:
                yhat = function(*x)
            else:
                yhat = function(x)

            error_message = f"Unit test {x} -> {y} failed, got {yhat} instead."
            assert yhat == y, error_message

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
        parser = self.parser
        # If cached, read it
        if use_cached:
            with CacheHandler("r") as f:
                cache = json.load(f)
                if prompt in cache:
                    model_response = cache[prompt]["model_response"]
                    sampling_params = cache[prompt]["sampling_params"]
                    from pyllm import parsers

                    parser = getattr(parsers, cache[prompt]["parser"])()
                    function = parser.parse_function(model_response)

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
            for cur_try in range(n_retries):
                try:

                    model_response = self.client.query(
                        formatted_prompt, params=sampling_params
                    )
                    sampling_params = asdict(sampling_params)
                    
                except RequestException as e:
                    # retry if server fails to give 200 response
                    error_message = json.loads(e.args[0].decode())
                    logging.warning(f'Try #{cur_try}, model query failed: {error_message}')
                    continue

                try:
                    function = self.parser.parse_function(model_response)
                except SyntaxError as e:
                    # retry if parsing fails
                    error_message = e.msg
                    logging.warning(f'Try #{cur_try}, function parsing failed: {e}')
                    continue

                try:
                    if unit_tests:
                        self._unit_test(function, unit_tests)
                except AssertionError as e:
                    # retry if any unit test fails
                    logging.warning(f'Try #{cur_try}, unit testing failed: {e}')
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
            "parser": parser.__class__.__name__,
        }

        with CacheHandler("w") as f:
            json.dump(new_cache, f)

        return Function(
            function=function,
            source=model_response,
            model_name=self.client.model_name,
            sampling_params=sampling_params,
            parser=parser,
        )

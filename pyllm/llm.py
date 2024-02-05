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
    """
    Manages access to a cache file with thread-safe read and write operations.
    
    This class ensures that the cache file is created if it doesn't exist and
    handles the locking mechanism to avoid concurrent write conflicts.
    
    Attributes:
        _CACHE_FILE (str): The path to the cache file used for storing function
            definitions and responses.
    
    Args:
        mode (str): The mode in which to open the cache file ('r' for read,
            'w' for write, etc.). Defaults to 'r'.
    """

    _CACHE_FILE = os.path.join(user_cache_dir("PyLLM"), "cached_functions.json")

    def __init__(self, mode: str = "r"):
        self.mode = mode

    def __enter__(self):
        """
        Manages access to a cache file with thread-safe read and write operations.
        
        This class ensures that the cache file is created if it doesn't exist and
        handles the locking mechanism to avoid concurrent write conflicts.
        
        Attributes:
            _CACHE_FILE (str): The path to the cache file used for storing function
                definitions and responses.
        
        Args:
            mode (str): The mode in which to open the cache file ('r' for read,
                'w' for write, etc.). Defaults to 'r'.
        """
        self.lock = FileLock(f"{self._CACHE_FILE}.lock")
        self.lock.acquire()
        try:
            if not os.path.exists(self._CACHE_FILE):
                with open(self._CACHE_FILE, "w") as f:
                    json.dump({}, f)
            else:
                try:
                    with open(self._CACHE_FILE, "r") as f:
                        json.load(f)
                # If the file is corrupted, empty it
                except json.JSONDecodeError:
                    with open(self._CACHE_FILE, "w") as f:
                        json.dump({}, f)
        finally:
            self.lock.release()

        self.file = open(self._CACHE_FILE, self.mode)
        return self.file

    def __exit__(self, *_):
        """
        Closes the cache file and releases the file lock on exiting the
        context manager.
        """
        self.file.close()
        self.lock.release()


class CodeLLM:
    """
    Facilitates the definition of functions using a language model, with
    capabilities to cache responses, parse model outputs, and validate through
    unit tests.
    
    Attributes:
        client (Client): The client interface to query the language model.
        parser (ParserBase): The parser to convert model responses into
            executable functions.
        prompt_template (PromptTemplate): The template used to format prompts
            sent to the model.
    """

    def __init__(
        self,
        client: Optional[Client] = None,
        parser: Optional[ParserBase] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        """
        Args:
            client (Optional[Client]): A client for interacting with the language
                model. Defaults to OpenAIChatClient if none is provided.
            parser (Optional[ParserBase]): A parser for interpreting model
                responses. Defaults to RegExParser if none is provided.
            prompt_template (Optional[PromptTemplate]): A template for generating
                prompts. Defaults to PromptTemplate if none is provided.
        """
        if client is None:
            client = OpenAIChatClient()
        self.client = client

        if parser is None:
            parser = RegExParser()
        self.parser = parser

        if prompt_template is None:
            prompt_template = PromptTemplate()
        self.prompt_template = prompt_template

    def _unit_test(self, function: Callable, unit_tests: List[Tuple]):
        """
        Executes unit tests on a given function to validate its correctness.
        
        Args:
            function (Callable): The function to be tested.
            unit_tests (List[Tuple]): A list of tuples, where each tuple
                contains input(s) and the expected output.        
        Raises:
            AssertionError: If a unit test fails, indicating that the function
                did not produce the expected output.
        """
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
        """
        Defines a function based on a given prompt, optionally using cached
        responses, and validates the function through unit testing.
        
        Args:
            prompt (str): The prompt describing the function to be defined.
            input_types (Optional[List]): A list of input types for the function.
            output_types (Optional[List]): A list of output types for the function.
            unit_tests (Optional[List[Tuple]]): A list of tuples for unit testing
                the function, where each tuple contains input(s) and expected output.
            use_cached (bool): Whether to use cached responses. Defaults to True.
            n_retries (int): The number of retries if querying the model or
                parsing the response fails. Defaults to 1.
            sampling_params (SamplingParams): Parameters for sampling the model's
                response. Defaults to an instance of SamplingParams with default values.
        
        Returns:
            Function: A Function object encapsulating the defined function,
                its source model response, and metadata.
        
        Raises:
            TooManyRetries: If the number of retries exceeds `n_retries` without
                successful definition and validation of the function.
        """
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
                logging.debug(f"Try {cur_try}")
                try:
                    model_response = self.client.query(
                        formatted_prompt, sampling_params=sampling_params
                    )
                    logging.debug("Model response: ", model_response)
                    sampling_params = asdict(sampling_params)

                except RequestException as e:
                    # retry if server fails to give 200 response
                    error_message = json.loads(e.args[0].decode())
                    logging.warning(
                        f"Try #{cur_try}, model query failed: {error_message}"
                    )
                    continue

                try:
                    function = self.parser.parse_function(model_response)
                except SyntaxError as e:
                    # retry if parsing fails
                    error_message = e.msg
                    logging.warning(f"Try #{cur_try}, function parsing failed: {e}")
                    continue

                try:
                    if unit_tests:
                        self._unit_test(function, unit_tests)
                except AssertionError as e:
                    # retry if any unit test fails
                    logging.warning(f"Try #{cur_try}, unit testing failed: {e}")
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

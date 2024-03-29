import json
import os
import logging
import timeout_decorator

from dataclasses import asdict
from typing import Optional, List, Tuple, Callable
from random import randint
from requests import RequestException

from pyllm.clients import Client, OpenAIChatClient
from pyllm.parsers import Parser, RegExParser
from pyllm.templates import PromptTemplate
from pyllm.utils.exceptions import TooManyRetries, NothingToParseError
from pyllm.interfaces import CodeGenerator
from pyllm.utils.types import SamplingParams, Function
from pyllm.utils.caching import CacheHandler
from pyllm.utils.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("baseline")
class CodeLLM(CodeGenerator):
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
        parser: Optional[Parser] = None,
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
        failures = []

        function = timeout_decorator.timeout(5, use_signals=False)(function)

        for x, y in unit_tests:
            if type(x) is tuple:
                yhat = function(*x)
            else:
                yhat = function(x)
            if y != yhat:
                error_message = f"{x} -> {yhat}, expected {y}."
                failures.append(error_message)
        assert (
            not failures
        ), f"{len(failures)}/{len(unit_tests)} tests failed:\n" + "\n".join(failures)

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
                    sampling_params = SamplingParams(**cache[prompt]["sampling_params"])
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
            for cur_try in range(n_retries):
                seed = randint(0, 2**62)
                sampling_params.seed = seed
                logging.debug(f"Try {cur_try}")
                try:
                    model_response = self.client.query(
                        formatted_prompt, sampling_params=sampling_params
                    )
                    logging.debug(f"Model response: {model_response}")
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
                except NothingToParseError as e:
                    logging.warning(f"Try #{cur_try}, {e}")
                    logging.debug(
                        f"No function found in the following model response:\n{model_response}"
                    )
                    continue

                if unit_tests:
                    unit_test_results = self.unit_test(function, unit_tests)
                    print(model_response)
                    print(unit_test_results)
                    if failures := [
                        result for result in unit_test_results if result.failed
                    ]:
                        error_message = (
                            f"{len(failures)}/{len(unit_test_results)} test failed."
                        )
                        for result in failures:
                            if result.error:
                                error_message += f"\n{result.x} -> {result.y}, got error {result.error}"
                            else:
                                error_message += f"\n{result.x} -> {result.y}, got {result.yhat} instead."

                        # retry if any unit test fails
                        logging.warning(
                            f"Try #{cur_try}, unit testing failed.\n{error_message}"
                        )
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
            "sampling_params": asdict(sampling_params),
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

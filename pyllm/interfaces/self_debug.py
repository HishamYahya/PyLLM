import json
import os
import logging

from dataclasses import asdict, dataclass
from typing import Optional, List, Tuple, Callable
from random import randint
from requests import RequestException
from enum import Enum


from pyllm.clients import Client, OpenAIChatClient
from pyllm.parsers import Parser, RegExParser
from pyllm.templates import PromptTemplate
from pyllm.utils.exceptions import TooManyRetries, NothingToParseError
from pyllm.interfaces import CodeGenerator
from pyllm.utils.types import SamplingParams, Function
from pyllm.utils.caching import CacheHandler
from pyllm.utils.registry import METHOD_REGISTRY

SELF_DEBUG_TEMPLATE = """Define a function for completing the following task in Python:
{{prompt}}
{% if unit_tests != None %}
The following are the unit tests that the function has to pass:
{% for unit_test in unit_tests %}
Input: {{unit_test[0]}}
Output: {{unit_test[1]}}
{% endfor %}
{% endif %}
"""


class FeedbackMode(str, Enum):
    SIMPLE = "simple"
    UT = "ut"
    UT_EXPL = "ut+expl"
    UT_TRACE = "ut+trace"


@dataclass
class UnitTestResult:
    id: int
    x: int
    y: any
    yhat: any
    error: any

    @property
    def failed(self) -> bool:
        return self.y != self.yhat or self.error


class SelfDebugLLM(CodeGenerator):
    def __init__(
        self,
        feedback_mode: FeedbackMode,
        client: Optional[Client] = None,
        parser: Optional[Parser] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        if client is None:
            client = OpenAIChatClient()
        self.client = client

        if parser is None:
            parser = RegExParser()
        self.parser = parser

        if prompt_template is None:
            prompt_template = PromptTemplate(SELF_DEBUG_TEMPLATE)
        self.prompt_template = prompt_template

        self.feedback_mode: FeedbackMode = feedback_mode

    def _unit_test(
        self, function: Callable, unit_tests: List[Tuple]
    ) -> List[UnitTestResult]:
        """
        Executes unit tests on a given function to validate its correctness.

        Args:
            function (Callable): The function to be tested.
            unit_tests (List[Tuple]): A list of tuples, where each tuple
                contains input(s) and the expected output.
        Returns:
            results (List[UnitTestResult])
        """
        results = []
        for i, (x, y) in enumerate(unit_tests):
            try:
                if type(x) is tuple:
                    yhat = function(*x)
                else:
                    yhat = function(x)
                results.append(
                    UnitTestResult(
                        **{"id": i, "error": None, "x": x, "y": y, "yhat": yhat}
                    )
                )
            except Exception as e:
                results.append(
                    UnitTestResult(
                        **{"id": i, "error": e, "x": x, "y": y, "yhat": None}
                    )
                )

        return results

    def _get_unit_test_feedback(
        self, failures: List[UnitTestResult], with_trace: bool = False
    ) -> str:
        feedback = "The code above fails for the following unit test(s):\n"
        for failure in failures:
            if failure.error:
                feedback += "{x} -> {y}, raised an error {error}\n".format(
                    **failure.__dict__
                )
            else:
                feedback += "{x} -> {y}, got {yhat} instead\n".format(
                    **failure.__dict__
                )
        if with_trace:
            feedback += f"Trace the execution of the function on input {failure.x}"
        else:
            feedback += "Please fix the Python code."
        return feedback

    def def_function(
        self,
        prompt: str,
        unit_tests: List[Tuple],
        max_turns: int = 2,
        use_cached: bool = True,
        n_retries: int = 1,
        sampling_params: SamplingParams = SamplingParams(),
    ) -> Function:
        messages: List = [
            {"role": "system", "content": "You are an expert programming assistant"},
            {
                "role": "user",
                "content": self.prompt_template.apply(prompt, unit_tests=unit_tests),
            },
        ]

        for cur_try in range(n_retries):
            seed = randint(0, 2**62)
            sampling_params.seed = seed
            logging.debug(f"Try {cur_try}")

            success_feedback = False
            success_turn_function = None
            for turn in range(max_turns):
                try:
                    if success_feedback:
                        success_turn_model_response = self.client.query(
                            messages=messages, sampling_params=sampling_params
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": success_turn_model_response,
                            }
                        )
                    else:
                        model_response = self.client.query(
                            messages=messages, sampling_params=sampling_params
                        )
                        messages.append(
                            {"role": "assistant", "content": model_response}
                        )

                    logging.debug(f"Model response: {model_response}")
                except RequestException as e:
                    # retry if server fails to give 200 response
                    error_message = json.loads(e.args[0].decode())
                    logging.warning(
                        f"Try #{cur_try}, model query failed: {error_message}"
                    )
                    if not success_feedback:
                        break

                try:
                    if success_feedback:
                        success_turn_function = self.parser.parse_function(
                            model_response
                        )
                    else:
                        function = self.parser.parse_function(model_response)
                except SyntaxError as e:
                    # retry if parsing fails
                    error_message = e.msg
                    logging.warning(f"Try #{cur_try}, function parsing failed: {e}")
                    if not success_feedback:
                        break
                except NothingToParseError as e:
                    logging.warning(f"Try #{cur_try}, {e}")
                    logging.debug(
                        f"No function found in the following model response:\n{model_response}"
                    )
                    if not success_feedback:
                        break

                # If one of the unit tests fails, return the function generated at the previous turn (already validated)
                if success_feedback and success_turn_function:
                    unit_test_results = self._unit_test(
                        success_turn_function, unit_tests=unit_tests
                    )
                    for test in unit_test_results:
                        if test.failed:
                            return Function(
                                function=function,
                                source=model_response,
                                model_name=self.client.model_name,
                                sampling_params=sampling_params,
                                parser=self.parser,
                            )

                    return Function(
                        function=success_turn_function,
                        source=success_turn_model_response,
                        model_name=self.client.model_name,
                        sampling_params=sampling_params,
                        parser=self.parser,
                    )

                unit_test_results = self._unit_test(function, unit_tests)
                failures = [test for test in unit_test_results if test.failed]

                # If all unit tests passed, go into the last success feedback turn
                if not failures:
                    success_feedback = True
                    messages.append(
                        {
                            "role": "user",
                            "content": "Is the code above correct? If not, please fix it.",
                        }
                    )
                    continue

                if self.feedback_mode == FeedbackMode.SIMPLE:
                    messages.append(
                        {
                            "role": "user",
                            "content": "The code above is wrong. Please fix it.",
                        }
                    )

                elif self.feedback_mode == FeedbackMode.UT:
                    feedback = self._get_unit_test_feedback(failures)
                    messages.append({"role": "user", "content": feedback})

                elif self.feedback_mode == FeedbackMode.UT_EXPL:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Explain the Python code line by line",
                        }
                    )
                    try:
                        explanation = self.client.query(
                            messages=messages, sampling_params=sampling_params
                        )
                    except RequestException as e:
                        error_message = json.loads(e.args[0].decode())
                        logging.warning(
                            f"Try #{cur_try}, model query failed: {error_message}"
                        )
                        break
                    feedback = self._get_unit_test_feedback(failures)
                    messages += [
                        {"role": "assistant", "content": explanation},
                        {"role": "user", "content": feedback},
                    ]

                elif self.feedback_mode == FeedbackMode.UT_TRACE:
                    feedback = self._get_unit_test_feedback(failures, with_trace=True)

                    messages.append({"role": "user", "content": feedback})
                    try:
                        trace = self.client.query(
                            messages=messages, sampling_params=sampling_params
                        )
                    except RequestException as e:
                        error_message = json.loads(e.args[0].decode())
                        logging.warning(
                            f"Try #{cur_try}, model query failed: {error_message}"
                        )
                        break
                    messages += [
                        {"role": "assistant", "content": trace},
                        {"role": "user", "content": "Please fix the Python code."},
                    ]


@METHOD_REGISTRY.register("self-debug-simple")
class SelfDebugLLMSimple(SelfDebugLLM):
    def __init__(
        self,
        client: Client | None = None,
        parser: Parser | None = None,
        prompt_template: PromptTemplate | None = None,
    ):
        super().__init__(FeedbackMode.SIMPLE, client, parser, prompt_template)


@METHOD_REGISTRY.register("self-debug-ut")
class SelfDebugLLMUT(SelfDebugLLM):
    def __init__(
        self,
        client: Client | None = None,
        parser: Parser | None = None,
        prompt_template: PromptTemplate | None = None,
    ):
        super().__init__(FeedbackMode.UT, client, parser, prompt_template)


@METHOD_REGISTRY.register("self-debug-ut-expl")
class SelfDebugLLMUTExpl(SelfDebugLLM):
    def __init__(
        self,
        client: Client | None = None,
        parser: Parser | None = None,
        prompt_template: PromptTemplate | None = None,
    ):
        super().__init__(FeedbackMode.UT_EXPL, client, parser, prompt_template)


@METHOD_REGISTRY.register("self-debug-ut-trace")
class SelfDebugLLMUTExpl(SelfDebugLLM):
    def __init__(
        self,
        client: Client | None = None,
        parser: Parser | None = None,
        prompt_template: PromptTemplate | None = None,
    ):
        super().__init__(FeedbackMode.UT_TRACE, client, parser, prompt_template)

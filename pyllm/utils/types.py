import re

from dataclasses import dataclass
from typing import Optional, Union, List, Callable

from pyllm.parsers import Parser


@dataclass
class SamplingParams:
    """
    Generation parameters following OpenAI's API
    """

    temperature: float = 0
    top_p: float = 1.0
    n: int = 1
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None


class Function:
    """
    A wrapper class for functions generated dynamically, typically from language model output.

    This class encapsulates a generated function, its source code, the model name from which
    it was generated, the parameters used for sampling, and the parser used to interpret the
    model's output. It allows the function to be called directly, and provides easy access
    to its source code and other metadata.

    Attributes:
        function (Callable): The callable function object that is wrapped by this class.
        source (str): The source code of the function as returned by the model.
        model_name (str): The name of the model that generated the function.
        sampling_params (SamplingParams): The parameters used for sampling when the function
            was generated.
        parser (ParserBase): The parser used to interpret the model's output and generate the
            function.
    """

    function: Callable
    source: str
    model_name: str
    sampling_params: SamplingParams
    parser: Parser

    def __init__(
        self,
        function: Callable,
        source: str,
        model_name: str,
        sampling_params: SamplingParams,
        parser: Parser,
    ):
        """
        Args:
            function (Callable): The generated function to be wrapped.
            source (str): The source code of the generated function.
            model_name (str): The name of the model that generated the function.
            sampling_params (SamplingParams): The sampling parameters used for generating
                the function.
            parser (ParserBase): The parser that was used to generate the function from
                the model's output.
        """
        self.function = function
        self.source = source
        self.model_name = model_name
        self.sampling_params = sampling_params
        self.parser = parser

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __str__(self):
        """
        Returns the formatted source code of the wrapped function.

        Returns:
            str: The first found function definition in the source code of the wrapped
                function. If multiple functions are present, returns the first one.
        """

        pattern = r"(def .+:.*\n(?:\s+.+\n)*)"
        return re.findall(pattern, self.source)[0]

    def __repr__(self):
        """
        Provides a concise representation of the Function instance for debugging.

        Returns:
            str: A string representation indicating that this is a generated function,
                including the model name and the seed used for sampling, if available.
        """
        return f"<Function generated by {self.model_name} seed={self.sampling_params.seed}>"

    def __doc__(self):
        """
        Returns the docstring of the wrapped function, if available.

        Returns:
            str: The docstring of the wrapped function.
        """
        return self.function.__doc__

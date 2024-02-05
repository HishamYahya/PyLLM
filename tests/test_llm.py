from typing import Callable

from pyllm.clients import Client
from pyllm.types import SamplingParams
from pyllm.llm import CodeLLM
from pyllm.types import Function


class MockClient(Client):
    model_name: str = "test"

    def __init__(self, response: str) -> None:
        super().__init__()
        self._response = response

    def query(self, input_string: str, sampling_params: SamplingParams) -> str:
        return self._response


def test_simple_function_generation():
    # A function that swaps two numbers
    response = 'The following is the definition of a function that swaps two numbers:<START-OF-CODE>\ndef swap_numbers(a, b):\n    """\n    Swaps the values of two numbers.\n    \n    Args:\n    a (int): The first number.\n    b (int): The second number.\n    \n    Returns:\n    tuple: A tuple containing the swapped values of the two numbers.\n    """\n    return b, a\n<END-OF-CODE>'

    llm = CodeLLM(client=MockClient(response))

    function = llm.def_function("", use_cached=False)

    assert isinstance(function, Function)
    assert isinstance(function.function, Callable)

    output = function(1, 10)

    assert output == (10, 1)


def test_function_generation_with_import():
    # a function that import defaultdict and returns an instance with the default value set to an input number
    response = 'Here is the requested function:\n\n```python\nfrom collections import defaultdict\n\ndef initialize_defaultdict(number):\n    """\n    Initializes a defaultdict with the given input value \'number\'.\n    \n    Parameters:\n    number (any): The input value to initialize the defaultdict with.\n    \n    Returns:\n    defaultdict: The initialized defaultdict with \'number\' as its default value.\n    """\n    my_defaultdict = defaultdict(lambda: number)\n    return my_defaultdict\n```\n\nNote: To run this code, you need to have the `collections` package imported.'

    llm = CodeLLM(client=MockClient(response))

    function = llm.def_function("", use_cached=False)

    assert isinstance(function, Function)
    assert isinstance(function.function, Callable)

    output = function(1)

    assert output["test"] == 1

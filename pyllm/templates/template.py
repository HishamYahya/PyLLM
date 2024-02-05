import jinja2

from typing import Optional, List, Literal, Tuple

from pyllm.templates.jinja import DEFAULT_FUNCTION_JINJA_TEMPLATE


class PromptTemplate:
    """
    A template handler for generating prompts using Jinja2 templates.

    This class is responsible for applying a Jinja2 template to a given set of parameters
    to generate prompts that are sent to a language model for function generation or other
    tasks that can be defined in the future.

    Attributes:
        jinja_template (Template): A Jinja2 template object initialized with a template string.
    """

    def __init__(self, jinja_template_string=DEFAULT_FUNCTION_JINJA_TEMPLATE):
        """
        Initializes the PromptTemplate with a Jinja2 environment and template.

        Args:
            jinja_template_string (str): A string containing the Jinja2 template for
                generating prompts. Defaults to DEFAULT_FUNCTION_JINJA_TEMPLATE.
        """
        environment = jinja2.Environment()
        self.jinja_template = environment.from_string(DEFAULT_FUNCTION_JINJA_TEMPLATE)

    def apply(
        self,
        prompt: str,
        # For now, only function generation is supported
        object_type: Literal["function"],
        input_types: Optional[List] = None,
        output_types: Optional[List] = None,
        unit_tests: Optional[List[Tuple]] = None,
    ) -> str:
        """
        Applies the Jinja2 template to the given parameters to generate a prompt.

        This method renders a prompt using the initialized Jinja2 template with the specified
        parameters.

        Args:
            prompt (str): The base prompt to which the template will be applied.
            object_type (Literal["function"]): The type of object to generate, currently
                supports only "function".
            input_types (Optional[List]): A list of types for the function inputs, if applicable.
            output_types (Optional[List]): A list of types for the function outputs, if applicable.
            unit_tests (Optional[List[Tuple]]): A list of tuples representing unit tests for the
                function, where each tuple contains the input(s) and the expected output.

        Returns:
            str: The generated prompt after applying the template with the provided parameters.
        """
        return self.jinja_template.render(
            prompt=prompt,
            object_type=object_type,
            input_types=input_types,
            output_types=output_types,
            unit_tests=unit_tests,
        )

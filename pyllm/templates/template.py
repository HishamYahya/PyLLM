import jinja2

from typing import Optional, List, Literal, Tuple

from pyllm.templates.jinja import DEFAULT_FUNCTION_JINJA_TEMPLATE


class PromptTemplate:
    def __init__(self, jinja_template_string=DEFAULT_FUNCTION_JINJA_TEMPLATE):
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
        return self.jinja_template.render(
            prompt=prompt,
            object_type=object_type,
            input_types=input_types,
            output_types=output_types,
            unit_tests=unit_tests,
        )

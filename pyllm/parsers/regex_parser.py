import ast
import re
from typing import Callable

from pyllm.parsers import ParserBase


class RegExParser(ParserBase):
    """
    A parser that extracts Python code blocks from an input string using regular expressions.

    This parser specifically focuses on identifying import statements and function definitions
    within the provided string. It dynamically executes these code blocks within the current
    namespace, allowing for the runtime definition of functions based on language model output
    or other dynamically generated Python code.
    """

    def parse_function(self, input_string: str) -> Callable:
        """
        Parses and dynamically executes Python code to define functions from the input string.

        This method uses regular expressions to extract import statements and function
        definitions from the provided input string. It then executes these statements and
        definitions within the current namespace, effectively defining any functions included
        in the input.

        Args:
            input_string (str): The input string containing the Python code to be parsed
                and executed. This string can include import statements and one or more
                function definitions.

        Returns:
            callable: The first function defined in the input string, if any. If the input
                does not define any functions, returns None.

        Raises:
            SyntaxError: If the input string contains syntax errors that prevent the parsing
                or execution of the code.
            NameError: If the execution of the code tries to access undefined variables or
                functions.
        """
        # Escape all \ characters
        input_string = re.sub(r"\\", r"\\\\", input_string)

        # import all needed packages provided by the LLM
        import_statements = []
        import_pattern = (
            r"(from [\w\.]+ import [\w\., ]+)|(import [\w\.]+(?:, [\w\.]+)*)"
        )
        matches = re.finditer(import_pattern, input_string, re.MULTILINE)

        for match in matches:
            block = match.group()
            import_statements.append(block)

        # Share the current namespace
        namespace = globals()

        for import_statement in import_statements:
            parsed_ast = ast.parse(import_statement, mode="exec")
            code = compile(parsed_ast, filename="import_statements", mode="exec")
            exec(code, namespace)

        # Define all the functions in the LLM output
        function_blocks = []
        function_pattern = r"(def .+:\n(?:\s+.+\n)*)"
        matches = re.finditer(function_pattern, input_string, re.MULTILINE)

        for match in matches:
            block = match.group()
            function_blocks.append(block)

        for code_str in function_blocks:
            parsed_ast = ast.parse(code_str, mode="exec")

            # Keep track of names of parsed functions
            function_names = set()
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.FunctionDef):
                    function_names.add(node.name)

            # Compile the code and execute it
            code = compile(parsed_ast, filename="compiled_generated_code", mode="exec")
            exec(code, namespace)
            # Return the first function that has a detected name
            for name, value in namespace.items():
                if name in function_names:
                    return value

        return None

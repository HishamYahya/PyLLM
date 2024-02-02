import ast
import re

from pyllm.parsers import ParserBase


class RegExParser(ParserBase):
    """
        A default parser that extracts python code blocks from the input
    """

    def parse_function(self, input_string: str) -> callable:
        # Escape all \ characters
        input_string = re.sub(r"\\", r"\\\\", input_string)

        code_blocks = []
        pattern = r"(def .+:\n(?:\s+.+\n)*)"

        matches = re.finditer(pattern, input_string, re.MULTILINE)

        for match in matches:
            block = match.group()
            code_blocks.append(block)

        for code_str in code_blocks:

            parsed_ast = ast.parse(code_str, mode="exec")

            # Keep track of names of parsed functions
            function_names = set()
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.FunctionDef):
                    function_names.add(node.name)

            # Compile the code and execute it
            code = compile(parsed_ast, filename="blah", mode="exec")
            # Share the current namespace
            namespace = globals()
            exec(code, namespace)
            # Return the first function that has a detected name
            for name, value in namespace.items():
                if name in function_names:
                    return value

        return None

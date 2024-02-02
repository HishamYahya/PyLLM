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

        # import all needed packages provided by the LLM
        import_statements = []
        import_pattern = r"(import .*)\n"
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

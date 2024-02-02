# PyLLM: In-Line  Executable Code Generation Using LLMs
PyLLM harnesses the power of Large Language Models (LLMs) to generate and execute code during runtime. Whether you're automating routine tasks, solving complex problems, or just experimenting with AI-generated code, PyLLM provides a seamless and intuitive interface for code generation and execution.

## Features
- Dynamic Code Generation: Generate functions on-the-fly in one line.
- Unit Testing: Specify unit tests to ensure the generated code meets your requirements.
- Caching Mechanism: Avoid re-generating code for repeated requests with an optional caching feature.
- Customizable: Easily tune your prompt template and sampling parameters.

## Installation
```
pip install pyllm-code
```

## Quick Start
Here's a simple example to get you started with PyLLM. The following code generates a function to swap two numbers:

```
from pyllm import CodeLLM

llm = CodeLLM()
swap_numbers = llm.def_function(
    "Swap two input numbers", unit_tests=[((1, 2), (2, 1))]
)
print(swap_numbers(20, 40))
# Output: (40, 20)
```

## Usage
To generate a function using PyLLM, create a CodeLLM instance and use the function method with the desired description and unit tests:

```
from pyllm import CodeLLM

# Initialize the LLM interface
llm = CodeLLM()

# Generate a function
my_function = llm.def_function(
    "Function description here",
    unit_tests=[("input_example", "expected_output")],
    use_cached=True  # Set to False to disable caching
)
```

## Contributing
Contributions are welcomed from the community, whether it's in the form of bug reports, feature requests, or code contributions. Please refer to our CONTRIBUTING.md for guidelines on how to contribute.

## License
PyLLM is released under the MIT license. See the LICENSE file for more details.

## Support
If you encounter any issues or have questions, please file an issue on our GitHub repository.

Enjoy crafting code with AI through PyLLM!

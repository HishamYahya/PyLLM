import timeout_decorator

from typing import List, Tuple, Callable
from dataclasses import dataclass
from pyllm.utils.types import Function
from pyllm.utils.io_utils import swallow_io


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


class CodeGenerator:
    def def_function(prompt: str, unit_tests: List[Tuple]) -> Function:
        pass

    @classmethod
    def unit_test(
        cls,
        function: Callable,
        unit_tests: List[Tuple],
        timeout_s: int = 5,
        use_signals: bool = True,
        quiet: bool = True
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

        function = timeout_decorator.timeout(timeout_s, use_signals=use_signals)(
            function
        )

        if quiet:
            function = swallow_io()(function)

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

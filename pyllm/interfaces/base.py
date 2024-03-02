import wrapt_timeout_decorator

from typing import List, Tuple, Callable, Union, Optional
from dataclasses import dataclass
from pyllm.utils.types import Function
from pyllm.utils.io_utils import swallow_io


@dataclass(frozen=True)
class UnitTestResult:
    id: int
    x: int
    y: any
    yhat: any
    error: any

    @property
    def failed(self) -> bool:
        return self.y != self.yhat or self.error

    @property
    def is_function(self) -> bool:
        return self.x is None and self.y is None

    @property
    def message(self) -> str:
        if self.is_function:
            if not self.failed:
                return f"Function unit test #{self.id} passed."
            else:
                return f"Function unit test #{self.id} failed. Error: {self.error}"
        else:
            if not self.failed:
                return f"Unit test #{self.id} passed: {self.x} -> {self.y}"

            if self.error:
                return f"Unit test #{self.id} failed: {self.x} -> {self.y}, got error {self.error}"
            else:
                return f"Unit test #{self.id} failed: {self.x} -> {self.y}, got {self.yhat} instead."


class CodeGenerator:
    def def_function(
        prompt: str, unit_tests: List[Union[Tuple, Callable]], namespace: dict
    ) -> Function:
        pass

    @classmethod
    def unit_test(
        cls,
        function: Callable,
        unit_tests: List[Union[Tuple, Callable]],
        timeout_s: Optional[int] = 5,
        use_signals: bool = False,
        quiet: bool = True,
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

        # Add timeout
        function = wrapt_timeout_decorator.timeout(timeout_s, use_signals=use_signals)(
            function
        )

        # Suppress IO of function
        if quiet:
            function = swallow_io()(function)

        for i, test in enumerate(unit_tests):
            if callable(test):
                try:
                    test(function)
                    results.append(
                        UnitTestResult(
                            **{
                                "id": i,
                                "error": None,
                                "x": None,
                                "y": None,
                                "yhat": None,
                            }
                        )
                    )
                except Exception as e:
                    results.append(
                        UnitTestResult(
                            **{"id": i, "error": e, "x": None, "y": None, "yhat": None}
                        )
                    )
                continue

            x, y = test
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

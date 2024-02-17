import re

from typing import List, Tuple, Iterator
from datasets import Dataset, load_dataset
from pyllm.utils.registry import DATASET_REGISTRY
from pyllm.function_datasets.base import EvaluationRow, FunctionDataset


class BaseMBPP(FunctionDataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset._data, info=dataset.info, split=dataset.split)

    def __getitem__(self, key) -> EvaluationRow:
        row = super().__getitem__(key)
        return self._to_evaluation_row(row)

    def _to_evaluation_row(self, row: dict) -> EvaluationRow:
        unit_tests = self.get_unit_tests(row)
        return EvaluationRow(row["text"], unit_tests)

    def __iter__(self) -> Iterator[EvaluationRow]:
        return map(self._to_evaluation_row, super().__iter__())

    def get_unit_tests(self, row) -> List[Tuple]:
        pattern = r"^assert .+\((.+)\)\s?==\s?(.+)$"
        unit_tests = []
        for test in row["test_list"]:
            match = re.search(pattern, test)
            if match:
                inp, out = match.groups()
                inp, out = eval(inp), eval(out)
                unit_tests.append((inp, out))
            else:
                pass
                # print(test)
        return unit_tests


@DATASET_REGISTRY.register("mbpp")
class MBPP(BaseMBPP):
    def __init__(self) -> None:
        self._dataset = load_dataset("mbpp")["test"]

        super().__init__(self._dataset)


@DATASET_REGISTRY.register("mbpp-sanitized")
class MBPPSanitized(BaseMBPP):
    def __init__(self) -> None:
        self._dataset = load_dataset("mbpp", "sanitized")["test"]

        super().__init__(self._dataset)

    def _to_evaluation_row(self, row: dict) -> EvaluationRow:
        unit_tests = self.get_unit_tests(row)
        # sanitized version renames "text" to "prompt"
        return EvaluationRow(row["prompt"], unit_tests)

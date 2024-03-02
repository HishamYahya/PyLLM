import re
import logging
from typing import Iterator, List, Tuple

from datasets import Dataset, load_dataset
from pyllm.utils.registry import DATASET_REGISTRY

from pyllm.function_datasets.base import FunctionDataset, EvaluationRow


@DATASET_REGISTRY.register("human-eval")
class HumanEval(FunctionDataset):
    def __init__(self) -> None:
        self._dataset = load_dataset("openai_humaneval")["test"]
        super().__init__(
            self._dataset._data, info=self._dataset.info, split=self._dataset.split
        )

    def __getitem__(self, key) -> EvaluationRow:
        row = super().__getitem__(key)
        return self._to_evaluation_row(row)

    def _to_evaluation_row(self, row: dict) -> EvaluationRow:
        try:
            unit_tests = self.get_unit_tests(row)
        except Exception as e:
            logging.warning(
                f"Skipping task {row['task_id']}. Got the following error when trying to parse its unit tests: {e}"
            )
            return None
        docstring_quotes = '"""' if '"""' in row['prompt'] else "'''"
        prompt = row["prompt"].split(docstring_quotes)[1]
            
        return EvaluationRow(prompt, unit_tests)

    def __iter__(self) -> Iterator[EvaluationRow]:
        return filter(
            lambda x: x is not None, map(self._to_evaluation_row, super().__iter__())
        )

    def get_unit_tests(self, row) -> List[Tuple]:
        namespace = {}
        test = exec(row["test"], namespace)
        test = namespace["check"]
        return [test]

from datasets import Dataset
from dataclasses import dataclass
from typing import List, Tuple, Iterator


@dataclass
class EvaluationRow:
    prompt: str
    unit_tests: List[Tuple]


class FunctionDataset(Dataset):
    def get_unit_tests(self, row) -> List[Tuple]:
        pass

    def __getitem__(self, idx) -> EvaluationRow:
        return super().__getitem__(idx)

    def __iter__(self) -> Iterator[EvaluationRow]:
        return super().__iter__()

from pyllm.types import SamplingParams


class Client:
    model_name: str

    def query(self, input_string: str, params: SamplingParams) -> str:
        raise NotImplementedError(
            "Client's querying capability has not been implemented yet for",
            self.__class__.__name__,
        )

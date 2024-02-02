class ParserBase:
    def parse_function(self, input_string):
        raise NotImplementedError(
            "The function parsing functionality hasn't been implemented for",
            self.__class__.__name__,
        )

    def parse_class(self, input_string):
        raise NotImplementedError(
            "The class parsing functionality hasn't been implemented for",
            self.__class__.__name__,
        )


class Function:
    function: callable
    source: str
    seed: int

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __repr__(self):
        return self.source

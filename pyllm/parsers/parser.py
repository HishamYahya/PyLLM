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

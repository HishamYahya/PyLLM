from pyllm.types import SamplingParams


class Client:
    """
    A base class for client implementations that query various models.

    This class defines a common interface for all clients, specifying that they must
    implement a query method. The class is designed to be subclassed by specific client
    implementations that interact with different models or services.

    Attributes:
        model_name (str): The name of the model this client is configured to query.
    """

    model_name: str

    def query(self, input_string: str, samplin_params: SamplingParams) -> str:
        """
        Abstract method for querying a model with a given input string and parameters.

        This method must be overridden by subclasses to implement the specific querying
        logic required to interact with a model or service. It defines the expected
        interface for all client queries.

        Args:
            input_string (str): The input string to be sent to the model for processing.
            params (SamplingParams): An instance of SamplingParams containing the parameters
                for the query, such as temperature, max tokens, etc., specific to the model's
                requirements.

        Returns:
            str: The response from the model as a string. The exact format of this string
                will depend on the model being queried and how the subclass implements this method.

        Raises:
            NotImplementedError: Indicates that the subclass has not implemented the querying
                capability, which is required for all Client subclasses.
        """
        raise NotImplementedError(
            "Client's querying capability has not been implemented yet for",
            self.__class__.__name__,
        )

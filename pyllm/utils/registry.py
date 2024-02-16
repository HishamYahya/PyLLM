class Registry:
    def __init__(self) -> None:
        self._classes_dict = {}

    def register(self, name: str = None):
        def _register(cls):
            if name is not None:
                key = name
            else:
                key = cls.__name__
            key = key.lower()
            self._classes_dict[key] = cls
            return cls

        return _register

    def build(self, name, *args, **kwargs):
        key = name.lower()
        if key not in self._classes_dict.keys():
            raise ValueError(f"Type '{name}' is not registered.")

        cls = self._classes_dict[key]
        return cls(*args, **kwargs)

    def __contains__(self, key: str):
        if key is None:
            return False
        return key.lower() in self._classes_dict

    def __getitem__(self, key: str):
        return self._classes_dict[key]


CLIENT_REGISTRY = Registry()
DATASET_REGISTRY = Registry()
METHOD_REGISTRY = Registry()

import os
import json

from filelock import FileLock
from appdirs import user_cache_dir

os.makedirs(user_cache_dir("PyLLM"), exist_ok=True)


class CacheHandler:
    """
    Manages access to a cache file with thread-safe read and write operations.
    
    This class ensures that the cache file is created if it doesn't exist and
    handles the locking mechanism to avoid concurrent write conflicts.
    
    Attributes:
        _CACHE_FILE (str): The path to the cache file used for storing function
            definitions and responses.
    
    Args:
        mode (str): The mode in which to open the cache file ('r' for read,
            'w' for write, etc.). Defaults to 'r'.
    """

    _CACHE_FILE = os.path.join(user_cache_dir("PyLLM"), "cached_functions.json")

    def __init__(self, mode: str = "r"):
        self.mode = mode

    def __enter__(self):
        """
        Manages access to a cache file with thread-safe read and write operations.
        
        This class ensures that the cache file is created if it doesn't exist and
        handles the locking mechanism to avoid concurrent write conflicts.
        
        Attributes:
            _CACHE_FILE (str): The path to the cache file used for storing function
                definitions and responses.
        
        Args:
            mode (str): The mode in which to open the cache file ('r' for read,
                'w' for write, etc.). Defaults to 'r'.
        """
        self.lock = FileLock(f"{self._CACHE_FILE}.lock")
        self.lock.acquire()
        try:
            if not os.path.exists(self._CACHE_FILE):
                with open(self._CACHE_FILE, "w") as f:
                    json.dump({}, f)
            else:
                try:
                    with open(self._CACHE_FILE, "r") as f:
                        json.load(f)
                # If the file is corrupted, empty it
                except json.JSONDecodeError:
                    with open(self._CACHE_FILE, "w") as f:
                        json.dump({}, f)
        finally:
            self.lock.release()

        self.file = open(self._CACHE_FILE, self.mode)
        return self.file

    def __exit__(self, *_):
        """
        Closes the cache file and releases the file lock on exiting the
        context manager.
        """
        self.file.close()
        self.lock.release()

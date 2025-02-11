import os
import logging

class PathUtil:
    
    _instance = None
    _root_path = None

    # -------------------------------------------------------------------------
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PathUtil, cls).__new__(cls)
            cls._init_root_path()
        return cls._instance

    # -------------------------------------------------------------------------

    @classmethod
    def _init_root_path(cls):
        """
        Initialize the root path
        """
        if cls._root_path is None:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            root_path_str = os.path.join(curr_path, '../../../../')
            cls._root_path = os.path.abspath(root_path_str).lower()

    # -------------------------------------------------------------------------

    @classmethod
    def get_root_path(cls) -> str:
        """
        Return the root path of the project
        """
        if cls._root_path is None:
            cls._init_root_path()
        return cls._root_path

    # -------------------------------------------------------------------------

    @classmethod
    def get_resources_path(cls, path_type: str='main') -> str:
        
        valid_types = ['main', 'test']
        
        if path_type not in valid_types:
            raise ValueError(f"Invalid type '{path_type}'.")
        
        path_parts = ['src', path_type, 'resources']
        resources_path = os.path.join(cls._root_path, *path_parts)
        
        if not os.path.exists(resources_path):
            raise FileNotFoundError(f"path {resources_path} does not exist.")
        
        return resources_path

    # -------------------------------------------------------------------------

    @staticmethod
    def ensure_path_exists(path: str, is_create: bool = True) -> bool | None:
        """
        Ensure the given path exists.

        Returns:
            None: if the path already exists.
            bool: 
                - True if the path was created.
                - False if the path does not exist and was not created.
        """
        if os.path.exists(path):
            return None
            
        if not is_create:
            logging.warning(f"Path '{path}' does not exist.")
            return False

        try:
            os.makedirs(path, exist_ok=True)
            logging.info(f"Path '{path}' created.")
            return True
        except OSError as ex:
            raise RuntimeError(f"Failed to create path '{path}': {ex}") from ex

    # -------------------------------------------------------------------------


if __name__=="__main__":
    
    root_path = PathUtil.get_root_path()
    print(f"Root path: {root_path}")
    
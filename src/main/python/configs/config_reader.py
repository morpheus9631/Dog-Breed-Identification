import json
import os
import yaml
from typing import Any, Dict

from utils import PathUtil

class ConfigReader:
    
    def __init__(self, config_file: str):
        """Initialize with the config file"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"'{config_file}' not found.")
        
        self.config_file = config_file
        self.config_data = self._load_config()
        
    # -------------------------------------------------------------------------
        
    def _load_config(self) -> dict:
        """Load config based on file extension"""
        file_extension = os.path.splitext(self.config_file)[1].lower()

        if file_extension in ['.yaml', '.yml']:
            return self._load_yaml()
        elif file_extension == '.json':
            return self._load_json()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    # -------------------------------------------------------------------------
    
    def _load_yaml(self) -> dict:
        """Load and parse YAML file"""
        with open(self.config_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    # -------------------------------------------------------------------------
        
    def _load_json(self) -> dict:
        """Load and parse JSON file"""
        with open(self.config_file, 'r', encoding='utf-8') as file:
            return json.load(file)

    # -------------------------------------------------------------------------
        
    def get_config(self, key: str) -> Any:
        """
        Get configuration value based on provided key
        """
        config = self.config_data
        
        for part in key.split('.'):
            config = config.get(part)
            if config is None:
                raise KeyError(f"Key '{key}' not found in configuration")
        return config

    # -------------------------------------------------------------------------
    
    def get_data(self) -> Dict[str, Any]:
        """Return the entire config data"""
        return self.config_data

    # -------------------------------------------------------------------------


if __name__ == "__main__":
    
    root_path = PathUtil.get_root_path()
    print(f"\nRoot path: {root_path}")
    
    resources_path = PathUtil.get_resources_path()
    print(f"\nResources path: {resources_path}")
    
    
    
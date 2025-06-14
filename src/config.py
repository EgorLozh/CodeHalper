import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    @property
    def LLM(self) -> str:
        return self.config['models']['llm']
    
    @property
    def EMBEDDING(self) -> str:
        return self.config['models']['embedding']
    
    @property
    def AVAILABLE_TYPES(self) -> list[str]:
        return self.config['document']['available_types']
    
    @property
    def CHUNK_SIZE(self) -> int:
        return self.config['document']['chunk_size']
    
    @property
    def CHUNK_OVERLAP(self) -> int:
        return self.config['document']['chunk_overlap']
    
    @property
    def DB_DIR(self) -> str:
        return self.config['paths']['db_dir']
    
    @property
    def TEMPLATE(self) -> str:
        return self.config['retrieval']['template']

    @property
    def DEFAULT_K(self) -> int:
        return self.config['retrieval']['default_k']

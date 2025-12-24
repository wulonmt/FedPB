import json
import os
from typing import Dict, Any


class EnvConfigLoader:
    """環境配置加載器"""
    
    def __init__(self, config_path: str = "env_config.json"):
        """
        初始化配置加載器
        
        Args:
            config_path: 配置文件路徑，默認為 "env_config.json"
        """
        self.config_path = config_path
        self.configs = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        從 JSON 文件加載配置
        
        Returns:
            配置字典
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create the file with environment configurations."
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
            print(f"✓ Loaded configuration from: {self.config_path}")
            return configs
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {self.config_path}: {e}")
    
    def get_config(self, environment: str) -> Dict[str, Any]:
        """
        獲取指定環境的配置
        
        Args:
            environment: 環境名稱
            
        Returns:
            環境配置字典
        """
        if environment in self.configs:
            config = self.configs[environment].copy()
            print(f"✓ Loaded config for environment: {environment}")
            print(f"  - Network Dim: {config['network_dim']}")
            print(f"  - Perturbation Scale: {config['perturbation_scale']}")
            print(f"  - Local Timesteps: {config['local_timesteps']}")
            print(f"  - Global Rounds: {config['global_rounds']}")
            return config
        else:
            print(f"⚠ Environment '{environment}' not found in config, using default")
            return self.configs.get('default', self._get_fallback_config())
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """
        當配置文件中沒有 default 時的後備配置
        
        Returns:
            後備配置字典
        """
        return {
            "network_dim": 64,
            "perturbation_scale": 1.0,
            "local_timesteps": 4096,
            "global_rounds": 200,
            "description": "Fallback configuration"
        }
    
    def list_environments(self) -> list:
        """
        列出所有可用的環境名稱
        
        Returns:
            環境名稱列表
        """
        return [env for env in self.configs.keys() if env != 'default']
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        驗證配置的完整性
        
        Args:
            config: 配置字典
            
        Returns:
            是否有效
        """
        required_keys = ['network_dim', 'perturbation_scale', 'local_timesteps', 'global_rounds']
        return all(key in config for key in required_keys)


# 全局配置加載器實例（單例模式）
_config_loader_instance = None

def get_config_loader(config_path: str = "env_config.json") -> EnvConfigLoader:
    """
    獲取配置加載器單例
    
    Args:
        config_path: 配置文件路徑
        
    Returns:
        EnvConfigLoader 實例
    """
    global _config_loader_instance
    if _config_loader_instance is None:
        _config_loader_instance = EnvConfigLoader(config_path)
    return _config_loader_instance
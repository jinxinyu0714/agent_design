"""
配置文件
"""
import os
from pathlib import Path


class Config:
    """系统配置类"""
    
    # 模型配置
    MODEL_NAME = "qwen3:latest"
    MODEL_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    MODEL_BASE_URL = "http://localhost:11434/v1"
    
    OPENAI_MODEL_NAME = "gpt-4o"
    OPENAI_API_KEY = ""

    # Google搜索配置
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
    
    # 输出配置
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/Users/jinxinyu/Desktop/资料/easy_agent/analy/automotive_output"))
  
    
    # 搜索配置
    MAX_SEARCH_RESULTS = 10
    MAX_IMAGES_PER_VEHICLE = 8
    
    # 网络请求配置
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 2
    
    # 团队配置
    MAX_MESSAGES = 35
    
    # 支持的图片格式
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
    
    @classmethod
    def get_output_dir(cls) -> Path:
        """获取输出目录"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return cls.OUTPUT_DIR
    
    @classmethod
    def get_images_dir(cls) -> Path:
        """获取图片输出目录"""
        images_dir = cls.get_output_dir() / "vehicle_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        return images_dir
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置是否有效"""
        required_keys = [
            cls.MODEL_API_KEY,
            cls.GOOGLE_API_KEY,
            cls.GOOGLE_CSE_ID
        ]
        
        for key in required_keys:
            if not key or key.startswith("your-") or key == "":
                return False
        
        return True
    

# 创建默认配置实例
config = Config()

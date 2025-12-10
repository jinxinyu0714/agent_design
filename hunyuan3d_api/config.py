# config.py
import os
import platform
from typing import Optional
from enum import Enum
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

class APIMode(Enum):
    """API模式枚举"""
    STANDARD = "standard"  # 标准版 - 高质量
    RAPID = "rapid"       # 极速版 - 快速生成
    PRO = "pro"           # 专业版 - 最高质量

class PolygonLevel(Enum):
    """面数档位枚举"""
    LOW = "low"      # 低面数 - 40,000
    MEDIUM = "medium" # 中面数 - 500,000
    HIGH = "high"    # 高面数 - 1,500,000

class OutputFormat(Enum):
    """输出格式枚举"""
    GLB = "glb"      # GLB格式（集成材质）
    STL = "stl"      # STL格式（白模）
    BOTH = "both"    # 同时输出GLB和STL

class Config:
    """Hunyuan3D API配置类 - 支持标准版、极速版和专业版"""
    
    # API端点配置
    API_ENDPOINT = "ai3d.tencentcloudapi.com"
    REGION = "ap-guangzhou"
    
    # 签名方法
    SIGN_METHOD = "TC3-HMAC-SHA256"
    
    # 超时设置（秒）
    REQUEST_TIMEOUT = 30
    POLLING_INTERVAL = 10
    
    # API模式设置 - 默认为专业版（最高质量）
    API_MODE = APIMode.PRO
    
    # 面数档位设置 - 默认为中面数 (500,000)
    POLYGON_LEVEL = PolygonLevel.MEDIUM
    
    # 面数映射
    POLYGON_COUNT_MAP = {
        PolygonLevel.LOW: 40000,
        PolygonLevel.MEDIUM: 500000,
        PolygonLevel.HIGH: 1500000
    }
    
    # PBR材质生成 - 默认为开启
    PBR_MATERIAL = False
    
    # 输出格式设置 - 默认为同时输出GLB和STL
    OUTPUT_FORMAT = OutputFormat.BOTH
    
    # 文件夹配置
    HISTORY_DIR = "history"
    OUTPUT_DIR = "output"
    
    # Blender路径配置
    if platform.system() == "Windows":
        # Windows系统常见的Blender安装路径
        BLENDER_PATHS = [
            "blender",  # PATH环境变量中的blender
            "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe",
            "C:\\Program Files (x86)\\Blender Foundation\\Blender\\blender.exe",
            "D:\\Program Files\\Blender Foundation\\Blender\\blender.exe"
        ]
    elif platform.system() == "Darwin":  # macOS
        BLENDER_PATHS = [
            "blender",
            "/Applications/Blender.app/Contents/MacOS/Blender"
        ]
    else:  # Linux
        BLENDER_PATHS = [
            "blender",
            "/usr/bin/blender",
            "/usr/local/bin/blender"
        ]
    
    BLENDER_PATH = None  # 将在初始化时设置
    
    # 密钥配置
    SECRET_ID = os.getenv("TENCENTCLOUD_SECRET_ID")
    SECRET_KEY = os.getenv("TENCENTCLOUD_SECRET_KEY")
    
    @classmethod
    def initialize(cls):
        """初始化配置"""
        cls.ensure_directories()
        cls.find_blender()
    
    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        os.makedirs(cls.HISTORY_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
    
    @classmethod
    def find_blender(cls):
        """自动查找Blender可执行文件"""
        for blender_path in cls.BLENDER_PATHS:
            try:
                import subprocess
                result = subprocess.run([blender_path, "--version"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    cls.BLENDER_PATH = blender_path
                    # print(f"✅ 找到Blender: {blender_path}")
                    return
            except (FileNotFoundError, PermissionError):
                continue
        
        # 如果没有找到，使用第一个路径并提示用户
        cls.BLENDER_PATH = cls.BLENDER_PATHS[0]
        print(f"⚠️  未自动找到Blender，将使用默认路径: {cls.BLENDER_PATH}")
        print("💡 如果转换失败，请在config.py中手动设置BLENDER_PATH")
    
    @classmethod
    def get_timestamp_folder_name(cls):
        """获取时间戳文件夹名称"""
        from datetime import datetime
        return datetime.now().strftime("%m-%d-%H.%M")
    
    @classmethod
    def get_credentials(cls) -> tuple:
        """获取认证凭证"""
        if not cls.SECRET_ID or not cls.SECRET_KEY:
            raise ValueError(
                "未找到有效的API凭证。请设置环境变量 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY"
            )
        return cls.SECRET_ID, cls.SECRET_KEY
    
    @classmethod
    def set_api_mode(cls, mode: APIMode):
        """
        设置API模式
        
        Args:
            mode: API模式 (STANDARD, RAPID 或 PRO)
        """
        cls.API_MODE = mode
    
    @classmethod
    def get_api_mode(cls) -> APIMode:
        """
        获取当前API模式
        
        Returns:
            APIMode: 当前API模式
        """
        return cls.API_MODE
    
    @classmethod
    def set_polygon_level(cls, level: PolygonLevel):
        """
        设置面数档位
        
        Args:
            level: 面数档位 (LOW, MEDIUM 或 HIGH)
        """
        cls.POLYGON_LEVEL = level
    
    @classmethod
    def get_polygon_level(cls) -> PolygonLevel:
        """
        获取当前面数档位
        
        Returns:
            PolygonLevel: 当前面数档位
        """
        return cls.POLYGON_LEVEL
    
    @classmethod
    def get_polygon_count(cls, level: Optional[PolygonLevel] = None) -> int:
        """
        获取面数数值
        
        Args:
            level: 面数档位，如果为None则使用当前设置
            
        Returns:
            int: 面数数值
        """
        if level is None:
            level = cls.POLYGON_LEVEL
        return cls.POLYGON_COUNT_MAP.get(level, 500000)
    
    @classmethod
    def set_pbr_material(cls, enabled: bool):
        """
        设置PBR材质生成
        
        Args:
            enabled: 是否开启PBR材质生成
        """
        cls.PBR_MATERIAL = enabled
    
    @classmethod
    def get_pbr_material(cls) -> bool:
        """
        获取PBR材质生成设置
        
        Returns:
            bool: 是否开启PBR材质生成
        """
        return cls.PBR_MATERIAL
    
    @classmethod
    def set_output_format(cls, format: OutputFormat):
        """
        设置输出格式
        
        Args:
            format: 输出格式 (GLB, STL 或 BOTH)
        """
        cls.OUTPUT_FORMAT = format
    
    @classmethod
    def get_output_format(cls) -> OutputFormat:
        """
        获取输出格式设置
        
        Returns:
            OutputFormat: 输出格式
        """
        return cls.OUTPUT_FORMAT

# 全局配置实例
config = Config()
config.initialize()  # 初始化配置
# models.py
from typing import Optional, Dict, Any
from enum import Enum
import base64

class OutputFormat(Enum):
    """支持的3D输出格式"""
    OBJ = "OBJ"
    GLB = "GLB"
    STL = "STL"
    USDZ = "USDZ"
    FBX = "FBX"
    MP4 = "MP4"

class JobStatus(Enum):
    """任务状态枚举"""
    WAIT = "WAIT"
    RUN = "RUN"
    FAIL = "FAIL"
    DONE = "DONE"

class ImageTo3DRequest:
    """图生3D请求参数 - 支持标准版、极速版和专业版"""
    
    def __init__(
        self,
        image_path: str,
        output_format: OutputFormat = OutputFormat.GLB,
        resolution: str = "1024x1024",
        style: str = "realistic",
        polygon_level: str = "medium",
        pbr_material: bool = True
    ):
        """
        初始化请求参数
        
        Args:
            image_path: 输入图片路径
            output_format: 输出3D格式
            resolution: 输出分辨率
            style: 生成风格
            polygon_level: 面数档位 (low, medium, high)
            pbr_material: 是否开启PBR材质生成
        """
        self.image_path = image_path
        self.output_format = output_format
        self.resolution = resolution
        self.style = style
        self.polygon_level = polygon_level
        self.pbr_material = pbr_material
    
    def to_rapid_api_params(self) -> Dict[str, Any]:
        """转换为极速版API参数"""
        with open(self.image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 极速版基础参数 - 确保设置ResultFormat为GLB
        params = {
            "ImageBase64": image_base64,
            "ResultFormat": self.output_format.value  # 确保设置为GLB
        }
        
        return params
    
    def to_standard_api_params(self) -> Dict[str, Any]:
        """转换为标准版API参数"""
        with open(self.image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 标准版基础参数 - 确保设置ResultFormat为GLB
        params = {
            "ImageBase64": image_base64,
            "ResultFormat": self.output_format.value,  # 确保设置为GLB
            "Resolution": self.resolution,
            "Style": self.style
        }
        
        return params
    
    def to_pro_api_params(self) -> Dict[str, Any]:
        """转换为专业版API参数 - 专业版固定返回OBJ格式的ZIP包"""
        with open(self.image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 专业版基础参数 - 专业版固定返回OBJ格式的ZIP包
        params = {
            "ImageBase64": image_base64,
            "ResultFormat": "OBJ",  # 专业版固定返回OBJ格式
            "Resolution": self.resolution,
            "Style": self.style
        }
        
        # 专业版特有参数
        # 面数档位映射
        polygon_mapping = {
            "low": "low",
            "medium": "medium", 
            "high": "high"
        }
        params["PolygonLevel"] = polygon_mapping.get(self.polygon_level, "medium")
        
        # PBR材质生成
        if self.pbr_material:
            params["EnablePBR"] = True
        
        return params

class JobResponse:
    """任务响应数据"""
    
    def __init__(self, data: Dict[str, Any]):
        self.raw_data = data
        self.job_id = data.get("JobId")
        
        status_value = data.get("Status", "WAIT")
        try:
            self.status = JobStatus(status_value)
        except ValueError:
            self.status = JobStatus.WAIT
            
        result_files = data.get("ResultFile3Ds", [])
        if result_files:
            self.download_url = result_files[0].get("Url")
        else:
            self.download_url = data.get("DownloadUrl")
            
        self.error_message = data.get("ErrorMessage")
    
    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self.status == JobStatus.DONE
    
    @property
    def is_completed(self) -> bool:
        """是否完成（成功或失败）"""
        return self.status in [JobStatus.DONE, JobStatus.FAIL]
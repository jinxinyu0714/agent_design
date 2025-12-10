# client.py
import json
import time
import os
import zipfile
import shutil
import tempfile
from typing import Dict, Any, Optional
from datetime import datetime

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ai3d.v20250513 import ai3d_client, models

from .models import ImageTo3DRequest, JobResponse
from .config import Config, APIMode

class Hunyuan3DClient:
    """Hunyuan3D API客户端 - 专业版专用"""
    
    def __init__(self, secret_id: Optional[str] = None, secret_key: Optional[str] = None):
        """
        初始化客户端
        """
        if secret_id and secret_key:
            self.secret_id, self.secret_key = secret_id, secret_key
        else:
            self.secret_id, self.secret_key = Config.get_credentials()
            
        self._init_client()
    
    def _init_client(self):
        """初始化腾讯云客户端"""
        cred = credential.Credential(self.secret_id, self.secret_key)
        
        http_profile = HttpProfile()
        http_profile.endpoint = Config.API_ENDPOINT
        http_profile.req_timeout = Config.REQUEST_TIMEOUT
        
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        client_profile.signMethod = Config.SIGN_METHOD
        
        self.client = ai3d_client.Ai3dClient(cred, Config.REGION, client_profile)

    def submit_image_to_3d_job(self, request: ImageTo3DRequest) -> Dict[str, Any]:
        """
        提交图生3D任务 - 专业版
        """
        try:
            return self._submit_pro_job(request)
        except TencentCloudSDKException as err:
            return {
                "success": False,
                "error": str(err),
                "error_code": getattr(err, "code", "Unknown")
            }
    
    def _submit_pro_job(self, request: ImageTo3DRequest) -> Dict[str, Any]:
        """提交专业版任务"""
        req = models.SubmitHunyuanTo3DProJobRequest()
        params = request.to_pro_api_params()
        req.from_json_string(json.dumps(params))
        
        resp = self.client.SubmitHunyuanTo3DProJob(req)
        resp_dict = json.loads(resp.to_json_string())
        
        return {
            "success": True,
            "job_id": resp_dict.get("JobId"),
            "response": resp_dict,
            "api_mode": "pro"
        }
    
    def get_job_status(self, job_id: str) -> Optional[JobResponse]:
        """
        查询任务状态
        """
        try:
            req = models.QueryHunyuanTo3DProJobRequest()
            params = {"JobId": job_id}
            req.from_json_string(json.dumps(params))
            
            resp = self.client.QueryHunyuanTo3DProJob(req)
            resp_dict = json.loads(resp.to_json_string())
            
            return JobResponse(resp_dict)
            
        except TencentCloudSDKException as err:
            # print(f"查询任务状态失败: {err}")
            return None
    
    def wait_for_job_completion(
        self, 
        job_id: str,
        timeout: int = 1800,
        poll_interval: int = 45
    ) -> Optional[JobResponse]:
        """
        等待任务完成（轮询）
        """
        start_time = time.time()
        
        # print(f"⏳ 等待专业版任务完成，超时时间: {timeout}秒，轮询间隔: {poll_interval}秒")
        
        while time.time() - start_time < timeout:
            job_status = self.get_job_status(job_id)
            
            if not job_status:
                return None
            
            # print(f"任务状态: {job_status.status.value}")
            
            if job_status.is_completed:
                if job_status.is_success:
                    # print("✅ 任务成功完成!")
                    if job_status.download_url:
                        # print(f"�� 下载链接: {job_status.download_url}")
                        pass
                else:
                    # print(f"❌ 任务失败: {job_status.error_message}")
                    pass
                return job_status
            
            time.sleep(poll_interval)
        
        # print(f"❌ 任务超时（{timeout}秒）")
        return None
    
    def download_pro_model(self, download_url: str, base_name: str) -> Dict[str, str]:
        """
        下载专业版模型并保存原始文件
        
        Args:
            download_url: 下载链接
            base_name: 基础文件名（不含扩展名）
            
        Returns:
            Dict: 包含原始文件路径的字典
        """
        try:
            import requests
            
            # print(f"�� 开始下载专业版ZIP包: {download_url}")
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # 创建时间戳文件夹
            timestamp_folder = Config.get_timestamp_folder_name()
            history_path = os.path.join(Config.HISTORY_DIR, timestamp_folder)
            os.makedirs(history_path, exist_ok=True)
            
            # print(f"�� 创建历史记录文件夹: {history_path}")
            
            # 创建原始文件目录
            original_dir = os.path.join(history_path, "original")
            os.makedirs(original_dir, exist_ok=True)
            
            # 下载ZIP文件
            zip_path = os.path.join(original_dir, f"{base_name}.zip")
            with open(zip_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            zip_size = os.path.getsize(zip_path)
            # print(f"✅ ZIP包已下载到: {zip_path} ({zip_size} 字节)")
            
            # 解压ZIP文件
            extract_dir = os.path.join(original_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                # print(f"�� ZIP包已解压到: {extract_dir}")
            
            # 查找并打印OBJ文件路径
            obj_files = []
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith('.obj'):
                        obj_files.append(os.path.join(root, file))
            
            if obj_files:
                obj_file = obj_files[0]
                # print(f"�� 找到OBJ文件: {obj_file}")
            else:
                # print("⚠️  在ZIP包中未找到OBJ文件")
                pass
            
            # print(f"✅ 专业版模型原始文件已保存到: {original_dir}")
            
            return {
                "history_path": history_path,
                "original_dir": original_dir,
                "extract_dir": extract_dir,
                "obj_files": obj_files
            }
            
        except Exception as e:
            # print(f"❌ 专业版模型处理失败: {e}")
            return {}
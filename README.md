## Get Started

### 方式一：本地环境安装

1. 安装 Python 3.12 并创建虚拟环境
    ```bash
    conda create -n mas python=3.12
    conda activate mas
    ```

    手动安装torch
     ```bash
    pip3 install torch torchvision
    ```

2. 安装依赖
    ```bash
    pip install -r requirements.txt
    ```

3. 配置环境变量
    ```bash
    # 复制环境变量模板文件
    cp .env.example .env
    
    # 编辑 .env 文件，填入你的 API Keys
    # 需要配置以下API密钥：
    # - DEEPSEEK_API_KEY: DeepSeek API密钥
    # - OPENAI_API_KEY: OpenAI API密钥
    # - GOOGLE_API_KEY: Google搜索API密钥
    # - GOOGLE_CSE_ID: Google自定义搜索引擎ID
    # - GEMINI_API_KEY: Google Gemini API密钥
    # - TENCENTCLOUD_SECRET_ID: 腾讯云Secret ID（用于Hunyuan3D）
    # - TENCENTCLOUD_SECRET_KEY: 腾讯云Secret Key（用于Hunyuan3D）
    ```

4. 安装 CLIPasso
    ```bash
    cd clippasso_utils
    git clone https://github.com/yael-vinker/CLIPasso.git
    ```

   加载 Docker 镜像
    ```bash
    cd /path/to/the/.tar
    docker load -i clippasso.tar
    ```

5. 安装ollama
 ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull gpt-oss
    ```


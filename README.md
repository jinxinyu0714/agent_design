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

3. 安装 CLIPasso
    ```bash
    cd clippasso_utils
    git clone https://github.com/yael-vinker/CLIPasso.git
    ```

   加载 Docker 镜像
    ```bash
    cd /path/to/the/.tar
    docker load -i clippasso.tar
    ```

4. 安装ollama
 ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull gpt-oss
    ```


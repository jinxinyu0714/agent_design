# 🚗 智能汽车设计助手 (Intelligent Vehicle Design Agent System)

本项目是一个集成了多智能体协作（Multi-Agent Collaboration）与多模态生成式 AI 的端到端汽车设计辅助系统。系统旨在通过自然语言交互，辅助设计师完成从**需求分析**、**概念草图**、**逼真渲染**、**3D建模**到**空气动力学评估**的全流程设计工作。

---

## ✨ 核心功能 (Core Features)

1.  **🧠 智能意图识别 (Intent Recognition)**
    -   基于 LLM 分析用户指令，自动判断任务类型（如竞品分析、由图生模型、由草图渲染等）。
    -   支持多模态输入（文本描述、参考图片、3D文件）。

2.  **📊 竞品分析智能体 (Auto-Analyst)**
    -   利用 `Autogen` 框架构建多智能体团队。
    -   自动联网搜索竞品车型参数、设计语言及市场反馈，生成专业的分析报告。

3.  **🎨 智能草图生成 (Sketch Generation)**
    -   集成 `CLIPasso` 算法，能够将参考图像抽象为极简线条草图，捕捉核心设计轮廓。
    -   运行于 Docker 容器中，保证环境稳定性，并通过 Python 接口无缝调用。

4.  **✨ AI 逼真渲染 (Realistic Rendering)**
    -   基于 `Stable Diffusion` + `ControlNet` 技术。
    -   精确控制线条约束，将抽象草图转化为高质量、多风格的逼真汽车渲染图。
    -   支持自定义 Prompt 控制生成风格（如“未来感”、“运动型”）。

5.  **🧊 3D 模型快速生成 (Image-to-3D)**
    -   接入腾讯 `Hunyuan3D` (混元3D) API。
    -   仅需一张渲染图即可快速生成 .stl/.obj 格式的 3D 模型文件。

6.  **🌪️ 风阻系数预测 (Aerodynamic Prediction)**
    -   内置轻量级神经网络 `phsoffNet`。
    -   无需昂贵的流体仿真软件（CFD），即可在毫秒级预测 3D 模型的风阻系数 ($C_d$)。

---

## 🛠️ 安装指南 (Installation)

### 环境要求
- **OS**: Linux (推荐 Ubuntu 22.04+) or Windows WSL2
- **Python**: 3.10 - 3.12
- **GPU**: NVIDIA GPU (支持 CUDA，推荐显存 16GB+)
- **Docker**: 用于运行 CLIPasso 模块

### 1. 克隆与环境配置

```bash
# Clone 项目 (示例)
git clone https://github.com/your-repo/agent_design.git
cd agent_design

# 创建 Conda 环境
conda create -n mas python=3.12
conda activate mas

# 安装 PyTorch (请根据您的 CUDA 版本选择对应的安装命令，以下为 CUDA 12.1 示例)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
pip install streamlit watchdog
```

### 3. 配置 API 密钥

复制 `.env.example` 为 `.env` 文件，并填入以下服务的 API Keys：

```ini
# llm keys
DEEPSEEK_API_KEY="your_key_here"
OPENAI_API_KEY="your_key_here"

# 搜索工具 keys (用于竞品分析)
GOOGLE_API_KEY="your_key_here"
GOOGLE_CSE_ID="your_key_here"

# 腾讯云 (用于 Hunyuan3D)
TENCENTCLOUD_SECRET_ID="your_id"
TENCENTCLOUD_SECRET_KEY="your_key"
```

### 4. 配置 CLIPasso (Docker)

本系统将 CLIPasso 封装在 Docker 容器中运行。

1.  确保 Docker 服务已启动。
2.  加载提供的镜像（或自行构建）：
    ```bash
    docker load -i clippasso.tar
    ```
    *(注：确保镜像名称与 `clippasso_utils/clippasso_tool.py` 中调用的一致)*

---

## 🚀 快速运行 (Quick Start)

### 方式一：Web 可视化界面 (推荐)

启动基于 Streamlit 的前端应用，提供完整的交互体验。

```bash
streamlit run app.py
```

-   启动成功后，浏览器访问终端显示的地址 (通常为 `http://localhost:8501`)。
-   **功能入口**：
    -   输入文本需求（例如：“设计一款类似Model Y的家用SUV”）。
    -   上传图片进行草图提取或 3D 重建。
    -   查看实时的任务进度与生成结果。

### 方式二：命令行 CLI 模式

如果你需要调试后端逻辑，可以直接运行工作流脚本。

```bash
python main_workflow.py
```

---

## 📂 项目结构 (Structure)

```text
agent_design/
├── app.py                      # [入口] Streamlit 前端应用程序
├── main_workflow.py            # [核心] 后端任务编排与分发逻辑
├── config.py                   # 全局配置加载
├── agents.py                   # Autogen 智能体定义 (意图识别, 分析师)
├── prompt.py                   # 系统 Prompt 模版
├── requirements.txt            # 项目依赖列表
│
├── cd_utils/                   # [模块] 风阻预测
│   ├── pipeline.py             # 预测推理管线
│   ├── model.py                # 神经网络模型定义
│   └── ...
│
├── clippasso_utils/            # [模块] 草图生成
│   ├── clippasso_tool.py       # Docker 调用接口
│   └── CLIPasso/               # CLIPasso 源码与资源
│
├── hunyuan3d_api/              # [模块] 3D 生成
│   ├── test_api.py             # API 调用封装
│   └── ...
│
├── render_utils/               # [模块] 渲染引擎
│   └── controlnet.py           # ControlNet 推理与模型缓存
│
└── utils/                      # [通用] 工具库
    ├── agent/                  # LangGraph/Autogen 辅助
    └── research_utils.py       # 搜索与网页抓取
```

## ⚠️ 常见问题与注意事项

1.  **Streamlit 启动报错 `RuntimeError: ... __path__._path ...`**
    -   这是 Streamlit 文件监控与 PyTorch 的冲突。
    -   **解决办法**: 确保项目根目录下存在 `.streamlit/config.toml` 且内容如下：
        ```toml
        [server]
        fileWatcherType = "none"
        ```
    -   *注意*: 修改此配置后，代码变动需要手动重启 Streamlit。

2.  **CUDA Out of Memory**
    -   渲染和 3D 生成非常消耗显存。如果遇到 OOM，尝试在 `render_utils/controlnet.py` 中减小生成的图像分辨率，或清理显存占用。

3.  **Docker 权限问题**
    -   如果运行 CLIPasso 报错 `Permission denied`，请确保当前用户有 Docker 执行权限，或使用 `sudo usermod -aG docker $USER` 添加权限。

---

## Credits

本项目基于以下开源项目或技术构建：
- [Autogen](https://github.com/microsoft/autogen) - 多智能体框架
- [Streamlit](https://streamlit.io/) - Web 应用框架
- [CLIPasso](https://github.com/yael-vinker/CLIPasso) - 抽象草图生成
- [ControlNet](https://github.com/lllyasviel/ControlNet) - 可控图像生成
- [Hunyuan3D](https://github.com/Tencent/Hunyuan3D) - 腾讯混元 3D 模型


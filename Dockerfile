# 使用轻量级 Python 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，防止 Python 生成 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖（构建 ChromaDB 可能需要 C++ 编译器）
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
# 使用清华源加速（因为是国内服务器）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 关键：入口点直接运行管道脚本
# 这里不使用 CMD 暴露端口，而是直接运行客户端脚本连接外部
CMD ["python", "mcp_pipe.py"]
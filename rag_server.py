# rag_server.py
from fastmcp import FastMCP
import os
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-Server")

# 初始化 MCP 服务
mcp = FastMCP("Zeabur-RAG-Service")

# --- RAG 初始化配置 ---
# 注意：Zeabur 部署时，这些 Key 会从环境变量读取，不要写死在代码里
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com") # 默认使用 DeepSeek

# 1. 初始化 Embedding 模型 (轻量级，或者使用在线 API 节省内存)
# 这里为了省内存，建议使用在线 Embedding 或极小的本地模型。
# 示例使用兼容 OpenAI 格式的 Embedding API
embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="text-embedding-3-small" # 根据你的供应商修改，如 deepseek-chat 不支持 embedding，需换用其他或本地
)

# 2. 初始化向量数据库 (持久化到本地目录)
VECTOR_STORE_PATH = "./chroma_db"
vector_store = Chroma(
    collection_name="my_knowledge",
    embedding_function=embeddings,
    persist_directory=VECTOR_STORE_PATH
)

# --- 工具定义 ---

@mcp.tool()
def query_knowledge_base(question: str) -> str:
    """
    查询内部知识库并回答问题。当用户问及特定私有文档内容时使用此工具。
    """
    logger.info(f"收到查询: {question}")

    # A. 检索 (Retrieve)
    # 检索最相关的3个片段
    results = vector_store.similarity_search(question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in results])

    if not context_text:
        return "抱歉，知识库中没有找到相关信息。"

    # B. 生成 (Generate)
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="deepseek-chat", # 或 moonshot-v1-8k
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    基于以下上下文回答用户的问题。如果上下文中没有答案，请直接说不知道。

    上下文:
    {context}

    问题:
    {question}
    """)

    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})

    return response.content

@mcp.tool()
def add_document(text: str) -> str:
    """
    向知识库添加新的文本知识。
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=x) for x in splitter.split_text(text)]

    vector_store.add_documents(docs)
    return f"成功添加 {len(docs)} 个知识片段到数据库。"

# 启动服务
if __name__ == "__main__":
    mcp.run(transport="stdio")
import re
import uuid
import time
import gradio as gr
from ollama import AsyncClient
from loguru import logger
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import OllamaEmbeddings
from concurrent.futures import ThreadPoolExecutor

from chromadb.config import Settings
from chromadb import Client
from langchain_community.vectorstores import Chroma


DEFAULT_LLM = "deepseek-r1:7b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


async def async_call_ollama_stream(model: str, messages: list[dict], host: str) -> str:
    """异步调用本地ollama模型

    Args:
        model (str): 模型名称
        messages (list[dict]): 消息记录
        host (str): ollama主机地址

    Returns:
        str: 返回模型调用后的相应文本
    """
    start = time.time()
    logger.info(f"Call model: {model}, messages: {messages}...")
    client = AsyncClient(host=host)

    full_response = ""
    async for chunk in await client.chat(model=model, messages=messages, stream=True):
        if chunk.get("message", {}).get("content"):
            content = chunk.get("message").get("content")
            print(content, end='', flush=True)
            full_response += content
    print("\n\n")
    logger.info(f"Finished calling, time elapsed: {time.time()-start}")
    return full_response


def load_pdf(file_path: str) -> List[Document]:
    """加载 PDF 文件.

    Args:
        file_path (str): PDF 文件的路径.

    Returns:
        List[Document]: 包含 PDF 文档的 Document 对象列表.
                        如果加载失败, 则返回 None.
    """
    try:
        pdf_loader = PyMuPDFLoader(file_path=file_path).load()
    except Exception as e:
        logger.exception(f"Fail  toload file {file_path}, detail: {e} .")
        return None
    return pdf_loader


def split_document(chunk_size: int = 100,
                   chunk_overlap: int = 20,
                   documents: List[Document] = None) -> List[Document]:
    """将文档分割成更小的文本块.

    Args:
        chunk_size (int, optional): 每个文本块的最大长度. 默认为 1000.
        chunk_overlap (int, optional): 相邻文本块之间的重叠长度, 用于保持上下文. 默认为 200.
        documents (List[Document], optional): 待分割的文档列表. 默认为 None.

    Returns:
        List[Document]: 分割后的文本块列表.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents=documents)
    return chunks


def get_embeddings_func(model_name: str = DEFAULT_EMBEDDING_MODEL,
                        base_url: str = DEFAULT_HOST) -> OllamaEmbeddings:
    """创建 OllamaEmbeddings 对象.

    Args:
        model_name (str): Ollama 模型的名称.

    Returns:
        OllamaEmbeddings: 使用指定模型配置的 OllamaEmbeddings 对象.
    """

    return OllamaEmbeddings(model=model_name, base_url=base_url, show_progress=True, num_gpu=1)


def get_embeddings(model_name: str, chunks: List[Document]) -> List[float]:
    """为文档块生成嵌入向量.

    Args:
        model_name (str): Ollama 模型的名称.
        chunks (List[Document]): 包含要嵌入的文本块的 Document 对象列表.

    Returns:
        List[List[float]]: 嵌入向量列表, 每个嵌入向量是对应文本块的浮点数列表.
    """
    start = time.time()
    embedding_func = get_embeddings_func(model_name=model_name)
    logger.info(f"Set embedding function with model name: {model_name} .")

    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(
            lambda x: embedding_func.embed_query(x.page_content), chunks))
    logger.info(
        f"Finish embedding {(len(chunks))} chunks! Time elapsed: {time.time()-start}")
    return embeddings


def store_vectors(chunks: List[Document],
                  collection_name: str = "llm_rag",
                  is_persistent: bool = True,
                  embeddings: Optional[list[float]] = None):
    """将文档块及其嵌入向量存储到 Chroma 向量存储中.

    Args:
        chunks (List[Document]): 包含要存储的文本块的 Document 对象列表.
        collection_name (str, optional): Chroma 集合的名称. 默认为 "llm_rag".
        is_persistent (bool, optional): Chroma 数据库是否应持久化到磁盘. 默认为 True.
        embeddings (list[float], optional): 块的预计算嵌入向量列表.
            如果为 None, 则不会存储嵌入向量. 默认为 None.
    """
    client = Client(Settings(is_persistent=is_persistent))
    collection = client.get_or_create_collection(name=collection_name)

    cnt = 0
    for idx, chunk in enumerate(chunks):
        try:
            collection.add(
                documents=[chunk.page_content],
                metadatas=[{"id": idx}],
                embeddings=[embeddings[idx]],
                ids=[str(uuid.uuid4())]
            )
            cnt += 1
        except Exception as e:
            logger.exception(
                f"Fail to add chunk {chunk.page_content} to vector store {collection_name}, detail: {e}")
            continue

    logger.info(f"Add {cnt} chunk to vector store {collection_name}")


def get_retriever(chroma_directory: str,
                  embedding_func: OllamaEmbeddings,
                  collection_name: str = "llm_rag") -> Optional[VectorStoreRetriever]:
    """获取检索器.

    Args:
        client (Client): Chroma 客户端.
        embedding_func (Callable): 嵌入函数.
        collection_name (str, optional): Chroma 集合的名称. 默认为 "llm_rag".

    Returns:
        VectorStoreRetriever | None: 检索器对象, 如果获取失败则返回 None.
    """
    try:
        vector_db = Chroma(persist_directory=chroma_directory,
                           embedding_function=embedding_func,
                           collection_name=collection_name)
    except Exception as e:
        logger.exception(f"Fail to get retriever, detail {e}")
        return None
    return vector_db.as_retriever()


def get_context(question: str, retriever: VectorStoreRetriever) -> list[Document]:
    """根据问题从检索器获取上下文.

    Args:
        question (str): 问题.
        retriever (VectorStoreRetriever): 检索器对象.

    Returns:
        str: 上下文字符串.
    """
    logger.info(f"Get question: {question}")
    try:
        results = retriever.invoke(question)
    except Exception as e:
        logger.exception(f"Fail to get context, detail: {e}")
        raise
    logger.info(f"Get results: {results}")
    # context = "\n\n".join([doc.page_content for doc in results])
    return results


async def query_deepseek(question: str,
                         context: str,
                         model_name: str = DEFAULT_LLM,
                         host: str = DEFAULT_HOST) -> str:
    """使用 DeepSeek 模型查询问题.

    Args:
        question (str): 问题.
        context (str): 上下文.
        host (str): ollama的本地api地址

    Returns:
        str: 答案.
    """
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    start = time.time()

    response = await async_call_ollama_stream(model=model_name,
                                              messages=[
                                                  {'role': 'user', 'content': formatted_prompt}],
                                              host=host)

    logger.info(f"Get repsonse with time spent: {time.time()-start}")
    final_answer = re.sub(r'<think>.*?</think>', '',
                          response, flags=re.DOTALL).strip()
    return final_answer


async def ask_question(question: str):
    """提问.

    Args:
        question (str): 问题.

    Returns:
        str: 答案.
    """
    retriever = get_retriever(
        chroma_directory="./chroma", embedding_func=get_embeddings_func())
    context = get_context(question=question, retriever=retriever)
    return await query_deepseek(question=question, context=context)


def main():
    """主函数.
    """
    interface = gr.Interface(
        fn=ask_question,
        inputs="text",
        outputs="text",
        title="RAG Chatbot: LLM RAG",
        description="Ask any question about the Foundations of LLMs book. Powered by DeepSeek-R1."
    )
    interface.launch()


if __name__ == "__main__":

    main()

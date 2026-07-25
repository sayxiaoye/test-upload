import os

os.environ["HF_HOME"] = "D:/AI_Models/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer

# 首次运行自动从hf-mirror国内镜像下载全部文件
model = SentenceTransformer("BAAI/bge-m3")

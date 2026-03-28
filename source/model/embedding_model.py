from source.core.config import Settings
import os

# Ngăn chặn HuggingFace tokenizers parallel warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

class Sentences_Transformer_Embedding:
    def __init__(self, settings: Settings):
        print(f" Đang tải mô hình nhúng: {settings.embedding_model_name}...")
        self.embeddings_bkai = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Đã khởi tạo Embedding Model thành công!")

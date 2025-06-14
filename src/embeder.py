import os
import shutil
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Embedder:
    def __init__(self, embedding_model: str, output_dir: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.output_dir = output_dir

    def clear_db(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def create_and_save_faiss_index(self, docs):
        documents = self.text_splitter.split_documents(docs)
        db = FAISS.from_documents(documents, self.embeddings)
        db.save_local(self.output_dir)

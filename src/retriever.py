from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document


class Retriever:
    def __init__(self, db_path: str, llm_model: str, embedding_model: str):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.db = FAISS.load_local(db_path, self.embeddings, allow_dangerous_deserialization=True)
        self.llm = ChatOllama(model=llm_model)

    def retrive(self, query: str, k: int) -> list[Document]:
        return self.db.similarity_search(query=query, k=k)

    def ask(self, query: str, template: str, k: int = 5) -> str:
        docs = self.retrive(query=query, k=k)
        prompt = PromptTemplate(template=template, input_variables=["docs", "query"])
        chain = prompt | self.llm
        return chain.invoke({"docs": docs, "query": query}).content

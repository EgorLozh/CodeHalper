import os
from tqdm import tqdm
from typing import Generator
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document


class Loader:
    def __init__(self, directory_path: str, available_types: list[str]):
        self.directory_path = directory_path
        self.available_types = available_types

    def load_code_from_directory(self) -> Generator[Document, None, None]:
        for root, _, files in tqdm(os.walk(self.directory_path), desc="Loading code from directory"):
            for file in files:
                if "." + file.split('.')[-1] in self.available_types:
                    file_path = os.path.join(root, file)
                    loader = TextLoader(file_path, encoding="utf-8")
                    yield from loader.load()

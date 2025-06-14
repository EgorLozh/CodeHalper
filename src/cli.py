import argparse
import os
from typing import Optional

from config import Config
from loader import Loader
from retriever import Retriever
from embeder import Embedder


def load_code(code_path: str, available_types: list[str], embedder: Embedder):
    print(f"Загрузка кода из {code_path}...")
    loader = Loader(code_path, available_types)
    docs = loader.load_code_from_directory()
    if not docs:
        print("Не найдено файлов для обработки")
        return
    embedder.create_and_save_faiss_index(docs)
    print("✅ Код успешно проиндексирован")


def clear_db(embedder: Embedder):
    embedder.clear_db()
    print("✅ База очищена")


def ask_query(query: str, retriever, template: str, k: int):
    print(f"Вопрос: {query}")
    answer = retriever.ask(query, template, k)
    print("\nОтвет:")
    print(answer)


def main():
    config = Config()
    embedder = Embedder(config.EMBEDDING, config.DB_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    print("CodeHelper запущен. Доступные команды: load, query, clear, exit")

    parser = argparse.ArgumentParser(description="CodeHelperBot - AI-powered code assistant")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    subparsers.add_parser("clear", help="Очистить векторную БД")

    load_parser = subparsers.add_parser("load", help="Загрузить код из директории в БД")
    load_parser.add_argument("path", type=str, help="Путь до директории с исходным кодом")

    query_parser = subparsers.add_parser("query", help="Задать вопрос по вашему коду")
    query_parser.add_argument("text", nargs=argparse.REMAINDER, help="Текст вопроса")

    subparsers.add_parser("exit", help="Выйти из программы")

    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue

            args = parser.parse_args(user_input.split())

            if args.command == "exit":
                print("Выход из CodeHelperBot")
                break

            elif args.command == "clear":
                if input("Вы уверены, что хотите очистить БД? [y/N] ").lower() in ("y", "yes"):
                    clear_db(embedder)
                else:
                    print("❌ Операция отменена")

            elif args.command == "load":
                if not os.path.isdir(args.path):
                    print(f"❌ Путь '{args.path}' не существует или не является директорией")
                    continue
                load_code(args.path, config.AVAILABLE_TYPES, embedder)

            elif args.command == "query":
                if not os.path.exists(config.DB_DIR):
                    print(f"❌ База знаний не инициализирована. Сначала выполните 'load'")
                    continue
                prompt_template = config.TEMPLATE
                retriever = Retriever(config.DB_DIR, config.LLM, config.EMBEDDING)
                ask_query(" ".join(args.text), retriever, prompt_template, config.DEFAULT_K)

        except SystemExit:
            print("❌ Неверная команда. Используйте: load, query, clear, exit")
        except Exception as e:
            print(f"⚠️ Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
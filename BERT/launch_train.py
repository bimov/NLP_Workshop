import torch
import os
from model import SiameseNetwork
from training import train


def main():
    # Пути к файлам с данными
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    triples_filepath = os.path.join(data_dir, "run.bm25_russian-msmarco.txt")
    collection_path = os.path.join(data_dir, "russian_collection.tsv")
    queries_path = os.path.join(data_dir, "russian_queries.dev.tsv")
    queries_path2 = os.path.join(data_dir, "russian_queries.dev.tsv")

    # Инициализация модели с выбранной стратегией агрегации (например, 'mean', 'cls' или 'max')
    model = SiameseNetwork(aggregation='mean')

    # Запуск обучения
    train(model, triples_filepath, collection_path, queries_path, queries_path2,
          start_line=0, max_examples=10000, epochs=3, lr=1e-5)

    # Сохранение обученной модели
    model_save_path = os.path.join(data_dir, "saved_siamese_bert4.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()

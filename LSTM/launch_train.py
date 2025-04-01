import torch
import torch.optim as optim
from LSTM.data_processing import load_fasttext_embeddings
from model import SiameseLSTM
from training import train_model
import os
import pickle


def main():
    # Пути к данным
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    data_path = os.path.join(data_dir, "run.bm25_russian-msmarco.txt")
    collection_path = os.path.join(data_dir, "russian_collection.tsv")
    queries_path = os.path.join(data_dir, "russian_queries.dev.tsv")
    queries_path2 = os.path.join(data_dir, "russian_queries.dev.tsv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка эмбеддингов
    with open(os.path.join(data_dir, "embedding_data.pkl"), "rb") as f:
        loaded_data = pickle.load(f)

    embedding_matrix = loaded_data["embedding_matrix"]
    word2idx = loaded_data["word2idx"]

    # Инициализация модели, оптимизатора и планировщика lr
    model = SiameseLSTM(embedding_matrix, bidirectional=False, hidden_size=128, num_layers=1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Обучение модели
    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_path=data_path,
        collection_path=collection_path,
        queries_path=queries_path,
        queries_path2=queries_path2,
        word2idx=word2idx,
        device=device,
        num_epochs=3,
        max_samples_per_epoch=10000
    )

    # Сохранение модели
    save_path = os.path.join(data_dir, "saved_siamese_lstm4.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Модель успешно обучена и сохранена по пути: {save_path}")


if __name__ == "__main__":
    main()

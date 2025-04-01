import torch
import os
import pickle
from LSTM.model import SiameseLSTM
from training import predict_relevant_snippets


def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    # Пути к данным и модели
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    model_path = os.path.join(data_dir, "saved_siamese_lstm.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка эмбеддингов (нужно для инициализации модели и word2idx)
    with open(os.path.join(data_dir, "embedding_data.pkl"), "rb") as f:
        loaded_data = pickle.load(f)

    embedding_matrix = loaded_data["embedding_matrix"]
    word2idx = loaded_data["word2idx"]

    # Инициализация модели
    model = SiameseLSTM(embedding_matrix, bidirectional=False, hidden_size=128, num_layers=1)
    model = load_model(model, model_path, device)

    # Пример запроса и сниппетов для предсказания
    query = "Какие книги для изучения машинного обучения?"
    snippets = [
        "Введение в машинное обучение: классическая книга для начинающих, охватывающая основы теории.",
        "Глубокое обучение с использованием Python: практическое руководство по созданию нейронных сетей.",
        "Статистическое обучение: современные методы и алгоритмы в машинном обучении.",
        "Машинное обучение на практике: пошаговые инструкции по реализации моделей.",
        "Полное руководство по нейронным сетям: от базовых понятий до современных архитектур.",
        "Современные подходы к машинному обучению: обзор последних исследований и методик."
    ]

    results = predict_relevant_snippets(model, query, snippets, word2idx, device, top_k=6)
    for snippet, score in results:
        print(f"Сниппет: {snippet}\nОценка: {score:.4f}\n")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from LSTM.data_processing import generate_examples_triples


def margin_ranking_loss(query_emb, pos_emb, neg_emb, margin=0.5):
    """
    Вычисляет Margin Ranking Loss для тройки эмбеддингов (запрос, позитивный, негативный).
    """
    s_pos = F.cosine_similarity(query_emb, pos_emb)
    s_neg = F.cosine_similarity(query_emb, neg_emb)
    target = torch.ones_like(s_pos)
    return nn.MarginRankingLoss(margin=margin)(s_pos, s_neg, target)


def train_model(model, optimizer, scheduler, data_path, collection_path, queries_path, queries_path2, word2idx, device,
                num_epochs=5, max_samples_per_epoch=10000):
    """
    Обучает модель на тройках примеров с использованием Margin Ranking Loss.
    Функция проходит по данным, генерируемым с помощью generate_examples_triples, и обновляет параметры модели.
    Для каждой эпохи данные считываются с указанной строки и обрабатываются порциями по max_samples_per_epoch примеров.
    """
    model.train()
    model.to(device)
    start_line = 0
    for epoch in range(num_epochs):
        total_loss = 0
        samples_processed = 0
        for idx, batch in itertools.islice(
                generate_examples_triples(data_path, collection_path, queries_path, queries_path2,
                                          start_line=start_line), max_samples_per_epoch):
            query, positive, negative = batch.values()
            query_emb = model([query], word2idx)
            pos_emb = model([positive], word2idx)
            neg_emb = model([negative], word2idx)
            loss = margin_ranking_loss(query_emb, pos_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            samples_processed += 1
            if samples_processed % 100 == 0:
                print(
                    f"Эпоха {epoch + 1}, Шаг {samples_processed}, Средняя потеря: {total_loss / samples_processed:.4f}")
        scheduler.step()
        print(f"Эпоха {epoch + 1} завершена. Средняя потеря: {total_loss / samples_processed:.4f}")
        print(f"Новый lr: {scheduler.get_last_lr()[0]:.6f}")
        start_line += max_samples_per_epoch


def predict_relevant_snippets(model, query, snippets, word2idx, device, top_k=5):
    """
    Выполняет предсказание, возвращая top_k наиболее релевантных сниппетов для заданного запроса.
    Функция вычисляет эмбеддинг запроса и эмбеддинги всех сниппетов, затем определяет косинусное сходство
    между запросом и каждым сниппетом и возвращает сниппеты с наивысшими оценками.
    """
    model.eval()
    with torch.no_grad():
        query_emb = model([query], word2idx).to(device)
        snippets_emb = model(snippets, word2idx).to(device)
        similarities = F.cosine_similarity(query_emb, snippets_emb, dim=1)
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(snippets)))
        results = [(snippets[idx], top_scores[i].item()) for i, idx in enumerate(top_indices)]
    return results

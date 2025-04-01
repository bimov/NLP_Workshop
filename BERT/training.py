import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from data_processing import generate_examples_triples


def margin_ranking_loss(query_emb, pos_emb, neg_emb, margin=0.5):
    """
    Вычисляет Margin Ranking Loss (triplet loss) для заданных эмбеддингов.
    """
    s_pos = F.cosine_similarity(query_emb, pos_emb)
    s_neg = F.cosine_similarity(query_emb, neg_emb)
    target = torch.ones_like(s_pos)
    loss_fn = nn.MarginRankingLoss(margin=margin)
    loss = loss_fn(s_pos, s_neg, target)
    return loss


def train(model, triples_filepath, collection_path, queries_path, queries_path2, start_line=10000, max_examples=None,
          epochs=3, lr=1e-5):
    """
    Обучает модель на тройках примеров с использованием Margin Ranking Loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        data_generator = generate_examples_triples(triples_filepath, collection_path, queries_path, queries_path2,
                                                   start_line=start_line * epoch)
        data_iterator = itertools.islice(data_generator, max_examples) if max_examples is not None else data_generator

        for idx, features in data_iterator:
            optimizer.zero_grad()
            query = features["query"]
            positive = features["positive"]
            negative = features["negative"]

            query_emb = model([query])
            pos_emb = model([positive])
            neg_emb = model([negative])

            loss = margin_ranking_loss(query_emb, pos_emb, neg_emb, margin=0.5)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            count += 1
            if count % 10 == 0:
                print(f"Epoch {epoch + 1}, Step {count}, Loss: {loss.item():.4f}")
        if count != 0:
            print(f"Epoch {epoch + 1} finished. Average Loss: {total_loss / count:.4f}")


def predict(model, query, candidate_texts):
    """
    Выполняет предсказание, возвращая ранжированный список кандидатов по релевантности заданному запросу.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        query_emb = model([query])
        cand_emb = model(candidate_texts)
        query_emb_exp = query_emb.expand(cand_emb.size(0), -1)
        scores = F.cosine_similarity(query_emb_exp, cand_emb)
    ranked = sorted(zip(candidate_texts, scores.cpu().numpy()), key=lambda x: x[1], reverse=True)
    return ranked

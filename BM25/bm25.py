import math
import re
from collections import Counter
import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def tokenize(text):
    tokens = re.findall(r'\w+', text.lower())
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    return lemmas


def compute_bm25(documents, query, k1=1.5, b=0.75):
    tokenized_docs = [tokenize(doc) for doc in documents]
    doc_lens = [len(doc) for doc in tokenized_docs]
    avgdl = sum(doc_lens) / len(documents)
    N = len(documents)

    df = {}
    for doc in tokenized_docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    query_terms = tokenize(query)

    scores = []
    for i, doc in enumerate(tokenized_docs):
        score = 0.0
        freq = Counter(doc)
        for term in query_terms:
            if term not in df:
                continue
            n_t = df[term]
            idf = math.log10(((N - n_t + 0.5) / (n_t + 0.5)) + 1)
            f_td = freq.get(term, 0)
            denominator = f_td + k1 * (1 - b + b * (doc_lens[i] / avgdl))
            score += idf * (f_td * (k1 + 1)) / denominator if denominator != 0 else 0
        scores.append(score)
    return scores


documents = [
    "Проращивание косточки авокадо в воде: пошаговая инструкция с фото и советами по уходу за ростком.",
    "Авокадо из косточки: как выбрать спелый плод и создать оптимальные условия для роста тропического растения.",
    "Частые ошибки при выращивании авокадо дома: почему желтеют листья и как спасти растение.",
    "Декоративное авокадо в горшке: особенности формирования кроны и пересадки в грунт.",
    "Лайфхаки для ускорения роста: использование стимуляторов корнеобразования и правильный полив.",
    "Можно ли получить плоды авокадо в квартире? Реальные сроки плодоношения и необходимые условия."
]
query = "Как вырастить авокадо из косточки в домашних условиях?"

bm25_scores = compute_bm25(documents, query)
for i, score in enumerate(bm25_scores, start=1):
    print(f"Документ {i}: BM25 = {score:.4f}")

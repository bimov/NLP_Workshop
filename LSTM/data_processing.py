import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk



def load_fasttext_embeddings(ft_path, max_vocab=200000):
    """
    Загружает предобученные FastText эмбеддинги из файла.
    """
    print("Загрузка FastText модели...")
    fasttext_model = KeyedVectors.load_word2vec_format(ft_path, binary=False)
    embedding_dim = fasttext_model.vector_size
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    embeddings = [np.zeros(embedding_dim), np.random.normal(scale=0.6, size=(embedding_dim,))]
    for i, word in enumerate(fasttext_model.index_to_key):
        if i >= max_vocab - 2:
            break
        word2idx[word] = len(word2idx)
        embeddings.append(fasttext_model[word])
    embedding_matrix = np.array(embeddings)
    print(f"Размер словаря: {len(word2idx)}, размер эмбеддингов: {embedding_matrix.shape}")
    return embedding_matrix, word2idx


def tokenize(text):
    """
    Токенизирует входной текст, приводя его к нижнему регистру.
    """
    return word_tokenize(text.lower(), language='russian')


def text_to_indices(text, word2idx):
    """
    Преобразует текст в последовательность индексов на основе словаря.
    """
    tokens = tokenize(text)
    return torch.tensor([word2idx.get(token, word2idx["<UNK>"]) for token in tokens], dtype=torch.long)


def generate_examples_triples(filepath, collection_path, queries_path, queries_path2, start_line=0):
    """
    Генерирует тройки примеров (запрос, позитивный и негативный) для обучения модели.

    Функция загружает коллекцию документов и запросы из заданных файлов,
    а затем для каждой строки файла с тройками выдаёт словарь с ключами:
    'query', 'positive' и 'negative'.
    """
    collection, queries = {}, {}
    # Загрузка коллекции документов
    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            doc_id, doc = line.strip().split("\t")
            collection[doc_id] = doc
    # Загрузка запросов из первого файла
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            query_id, query = line.strip().split("\t")
            queries[query_id] = query
    # Загрузка запросов из второго файла
    with open(queries_path2, encoding="utf-8") as f:
        for line in f:
            query_id, query = line.strip().split("\t")
            queries[query_id] = query
    # Генерация примеров по тройкам идентификаторов
    with open(filepath, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_line:
                continue
            query_id, pos_id, neg_id = line.strip().split("\t")
            yield idx, {
                "query": queries.get(query_id, ""),
                "positive": collection.get(pos_id, ""),
                "negative": collection.get(neg_id, "")
            }

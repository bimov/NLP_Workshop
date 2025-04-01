def generate_examples_triples(filepath, collection_path, queries_path, queries_path2, start_line=0):
    """
    Генерирует тройки примеров из файлов с данными.
    Загружает коллекцию документов и запросы из указанных файлов, затем для каждой строки
    файла с тройками (query_id, pos_id, neg_id) возвращает словарь с ключами:
    'query', 'positive' и 'negative'.
    """
    collection = {}
    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            doc_id, doc = line.rstrip().split("\t")
            collection[doc_id] = doc

    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            query_id, query = line.rstrip().split("\t")
            queries[query_id] = query
    with open(queries_path2, encoding="utf-8") as f:
        for line in f:
            query_id, query = line.rstrip().split("\t")
            queries[query_id] = query

    with open(filepath, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_line:
                continue
            query_id, pos_id, neg_id = line.rstrip().split("\t")
            features = {
                "query": queries[query_id],
                "positive": collection[pos_id],
                "negative": collection[neg_id],
            }
            yield idx, features

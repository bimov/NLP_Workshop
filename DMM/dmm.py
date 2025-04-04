from collections import defaultdict

documents = [
    "apple banana",
    "apple apple orange",
    "banana orange"
]

# разбиение каждого документа на список слов
documents_tokens = [document.split() for document in documents]

# формирование словаря всех уникальных слов
vocabulary = list({word for tokens in documents_tokens for word in tokens})
vocabulary_size = len(vocabulary)
print("Словарь:", vocabulary)

# количество тем
num_topics = 2

# гиперпараметры
alpha = 0.1
beta = 0.1

topic_assignment = [0, 0, 1]

total_words_per_topic = [0 for _ in range(num_topics)]
word_count_per_topic = [defaultdict(int) for _ in range(num_topics)]

for doc_index, tokens in enumerate(documents_tokens):
    assigned_topic = topic_assignment[doc_index]
    total_words_per_topic[assigned_topic] += len(tokens)
    for token in tokens:
        word_count_per_topic[assigned_topic][token] += 1

# Функция вычисления вероятности назначения темы для документа
def compute_topic_probability(document_tokens, topic_index):
    prior_probability = total_words_per_topic[topic_index] + alpha
    likelihood_probability = 1.0
    for token in document_tokens:
        token_frequency_in_topic = word_count_per_topic[topic_index][token]
        likelihood_probability *= (token_frequency_in_topic + beta) / (total_words_per_topic[topic_index] + vocabulary_size * beta)
    return prior_probability * likelihood_probability

num_iterations = 50

for iteration in range(num_iterations):
    for doc_index, tokens in enumerate(documents_tokens):
        # Удаляем вклад текущего документа из счетчиков
        current_topic = topic_assignment[doc_index]
        total_words_per_topic[current_topic] -= len(tokens)
        for token in tokens:
            word_count_per_topic[current_topic][token] -= 1

        # вероятность для каждой темы
        topic_probabilities = [compute_topic_probability(tokens, topic_index) for topic_index in range(num_topics)]

        # тема с максимальной вероятностью
        new_topic = topic_probabilities.index(max(topic_probabilities))

        topic_assignment[doc_index] = new_topic
        total_words_per_topic[new_topic] += len(tokens)
        for token in tokens:
            word_count_per_topic[new_topic][token] += 1

    print(f"Итерация {iteration + 1}: назначения тем -> {topic_assignment}")

print("\nРаспределение тем по документам:")
for i, topic in enumerate(topic_assignment, start=1):
    print(f"Документ {i} принадлежит теме {topic + 1}")

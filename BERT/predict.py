import torch
import os
from model import SiameseNetwork
from training import predict


def load_model(model, model_path, device):
    """
    Загружает сохранённое состояние модели.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    # Путь к сохранённой модели
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    model_path = os.path.join(data_dir, "saved_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork(aggregation='mean')
    model = load_model(model, model_path, device)
    test_examples = [
        {
            "query": "Какие преимущества у регулярных занятий йогой?",
            "candidates": [
                "Йога улучшает гибкость и осанку: как всего 20 минут в день меняют тело за месяц занятий.",
                "Медитативные практики в йоге: снижение стресса и нормализация сна через дыхательные упражнения.",
                "Йога для офисных работников: комплексы асан для снятия напряжения в шее и пояснице.",
                "Исследования ученых: влияние йоги на сердечно-сосудистую систему и когнитивные функции.",
                "Йога vs пилатес: сравниваем эффективность для похудения и развития мышечного корсета.",
                "Как выбрать направление йоги новичку: хатха, аштанга, кундалини и другие стили."
    ]
        },
        {
            "query": "Как приготовить борщ?",
            "candidates": [
                "Традиционный рецепт украинского борща.",
                "Советы по приготовлению вкусных супов.",
                "Обзор новинок кино.",
                "Рецепты здорового питания."
            ]
        },
        {
            "query": "Что нового в мире технологий?",
            "candidates": [
                "Последние новости в IT-сфере.",
                "Обзор новых гаджетов и инноваций.",
                "Советы по уходу за растениями.",
                "Новости моды и стиля."
            ]
        },
        {
            "query": "Как начать изучать программирование?",
            "candidates": [
                "Основы программирования для начинающих.",
                "Лучшие онлайн-курсы по Python и Java.",
                "Новости музыкальной индустрии.",
                "Советы по здоровому образу жизни."
            ]
        }
    ]

    for example in test_examples:
        query = example["query"]
        candidates = example["candidates"]
        ranked = predict(model, query, candidates)
        print(f"\nЗапрос: {query}")
        for candidate, score in ranked:
            print(f"Score: {score:.4f} - {candidate}")


if __name__ == "__main__":
    main()

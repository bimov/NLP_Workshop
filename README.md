[Презентация](https://docs.google.com/presentation/d/1iIThP7jm1k16UAtk-hseoq7D0USnQUDqEvxrkedwWkw/edit?usp=sharing)

# Описание проекта
Были реализованы 3 модели для опеределения релевантности сниппетов к запросу.
1. В папке BERT лежит модель, русскоязычная версия модели BERT, обученную на разговорных текстах (социальные сети, Pikabu), которую мы доубучали на 100 000 запросах и сниппетах из датасета mmarco (русскоязычная версия MS Macro).
2. В папке LSTM лежит модель, которую мы обучали с 0, используя только готовые ембендинги из FastText.
3. В папке DiTy лежит готовая модель, которая была обучена по такому же принципу, как мы, только на всех 8 800 000 запросах.
# Использование моделей
1. `git clone https://github.com/bimov/NLP_Workshop.git`
2. Добавить папку `data` и загрузить в нее файлы с [гугл диска](https://drive.google.com/drive/folders/1xue6ryiruPfQDtMpuPbzx2bqTD4j7GyD?usp=sharing)
3. Запустить `PyQt/main.py`
4. Перед вами будет приложение, в котором при вводе запроса и сниппетов, 3 модели выдадут свой результат определения релевантности.

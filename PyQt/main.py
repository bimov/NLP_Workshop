import sys
import os
import torch
import pickle
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QDialog, QDialogButtonBox, QSplitter
)
from PyQt5.QtCore import Qt
from BERT.model import SiameseNetwork  # Для BERT
from LSTM.model import SiameseLSTM  # Для LSTM
from sentence_transformers import CrossEncoder

# Подкласс для текстового поля с возможностью открытия увеличенного вида по двойному клику.
class EnlargeableTextEdit(QTextEdit):
    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.setReadOnly(True)
        self.title = title

    def mouseDoubleClickEvent(self, event):
        self.open_enlarged_view()

    def open_enlarged_view(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Увеличенный просмотр - {self.title}")
        dialog.resize(600, 400)
        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(self.toPlainText())
        font = text_edit.font()
        font.setPointSize(font.pointSize() + 2)
        text_edit.setFont(font)
        layout.addWidget(text_edit)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        buttonBox.accepted.connect(dialog.accept)
        layout.addWidget(buttonBox)
        dialog.exec_()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NLP Модель - Сравнение результатов")
        self.load_models()
        self.setup_ui()
        self.apply_styles()
        self.original_sys_path = sys.path.copy()

    def load_models(self):
        """Загружаем модели и эмбеддинги при старте"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем BERT-модель
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
        model_path_bert = os.path.join(base_dir, "../data", "saved_model.pth")
        self.bert_model = SiameseNetwork(aggregation='mean')
        self.bert_model.load_state_dict(torch.load(model_path_bert, map_location=device))
        self.bert_model.to(device)
        self.bert_model.eval()

        # Загружаем LSTM-модель и эмбеддинги
        model_path_lstm = os.path.join(base_dir, "../data", "saved_siamese_lstm.pth")
        embedding_data_path = os.path.join(base_dir, "../data", "embedding_data.pkl")

        with open(embedding_data_path, "rb") as f:
            loaded_data = pickle.load(f)

        self.embedding_matrix = loaded_data["embedding_matrix"]
        self.word2idx = loaded_data["word2idx"]

        self.lstm_model = SiameseLSTM(self.embedding_matrix, bidirectional=False, hidden_size=128, num_layers=1)
        self.lstm_model.load_state_dict(torch.load(model_path_lstm, map_location=device))
        self.lstm_model.to(device)
        self.lstm_model.eval()

        # Загружаем DiTy CrossEncoder
        self.dity_model = CrossEncoder('DiTy/cross-encoder-russian-msmarco', max_length=512, device=str(device))

    def setup_ui(self):
        main_layout = QVBoxLayout()

        # Восстанавливаем расположение полей ввода запроса и сниппетов как было изначально.
        input_layout = QVBoxLayout()

        # Ввод запроса
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Введите запрос...")
        input_layout.addWidget(QLabel("Запрос:"))
        input_layout.addWidget(self.query_input)

        # Ввод сниппетов
        self.snippets_input = QTextEdit()
        self.snippets_input.setPlaceholderText("Введите сниппеты, по одному на строке...")
        # Для примера оставим стандартный размер, его можно регулировать при необходимости
        input_layout.addWidget(QLabel("Сниппеты:"))
        input_layout.addWidget(self.snippets_input)

        main_layout.addLayout(input_layout)

        # Кнопки: предсказание и расширенный просмотр всех результатов
        buttons_layout = QHBoxLayout()
        self.predict_button = QPushButton("Предсказать")
        self.predict_button.clicked.connect(self.on_predict)
        buttons_layout.addWidget(self.predict_button)

        self.expand_all_button = QPushButton("Expand All")
        self.expand_all_button.setToolTip("Открыть увеличенные окна для всех результатов одновременно")
        self.expand_all_button.clicked.connect(self.expand_all_views)
        buttons_layout.addWidget(self.expand_all_button)
        main_layout.addLayout(buttons_layout)

        # QSplitter для вывода результатов с тремя областями
        self.splitter = QSplitter(Qt.Horizontal)

        self.bert_result = EnlargeableTextEdit(title="BERT")
        bert_container = self.create_result_group("BERT", self.bert_result)
        self.splitter.addWidget(bert_container)

        self.lstm_result = EnlargeableTextEdit(title="LSTM")
        lstm_container = self.create_result_group("LSTM", self.lstm_result)
        self.splitter.addWidget(lstm_container)

        self.dity_result = EnlargeableTextEdit(title="DiTy")
        dity_container = self.create_result_group("DiTy", self.dity_result)
        self.splitter.addWidget(dity_container)

        # Задаём стартовые размеры для областей вывода
        self.splitter.setSizes([350, 350, 350])
        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)

    def create_result_group(self, title, widget):
        container = QWidget()
        vbox = QVBoxLayout()
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        vbox.addWidget(label)
        vbox.addWidget(widget)
        container.setLayout(vbox)
        return container

    def bert_predict(self, query, candidates):
        """Функция предсказания для BERT"""
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'BERT'))
        from BERT.training import predict
        return predict(self.bert_model, query, candidates)

    def lstm_predict(self, query, snippets):
        """Функция предсказания для LSTM"""
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LSTM'))
        from LSTM.training import predict_relevant_snippets
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return predict_relevant_snippets(self.lstm_model, query, snippets, self.word2idx, device, top_k=6)

    def dity_predict(self, query, snippets):
        """Функция предсказания для DiTy"""
        return self.dity_model.rank(query, snippets)

    def on_predict(self):
        query = self.query_input.text().strip()
        snippets_text = self.snippets_input.toPlainText().strip()
        snippets = [s.strip() for s in snippets_text.splitlines() if s.strip()]
        if not query or not snippets:
            error_text = "Введите и запрос, и сниппеты."
            self.bert_result.setPlainText(error_text)
            self.lstm_result.setPlainText(error_text)
            self.dity_result.setPlainText(error_text)
            return
        # Предсказания с использованием загруженных моделей
        bert_output = self.bert_predict(query, snippets)
        sys.path = self.original_sys_path
        lstm_output = self.lstm_predict(query, snippets)
        sys.path = self.original_sys_path
        dity_output = self.dity_predict(query, snippets)
        dity_output_good = []
        for element in dity_output:
            id = element['corpus_id']
            dity_output_good.append((snippets[id], element['score']))

        print(bert_output, lstm_output, dity_output_good)
        self.bert_result.setPlainText(self.format_results(bert_output))
        self.lstm_result.setPlainText(self.format_results(lstm_output))
        self.dity_result.setPlainText(self.format_results(dity_output_good))

    def format_results(self, results):
        output = ""
        for candidate, score in results:
            output += f"Score: {score:.4f} - {candidate}\n"
        return output

    def expand_all_views(self):
        # Создаем единый диалог, где увеличенные окна результатов расположены рядом.
        dialog = QDialog(self)
        dialog.setWindowTitle("Увеличенный просмотр всех результатов")
        dialog.resize(1200, 600)
        layout = QVBoxLayout(dialog)

        splitter = QSplitter(Qt.Horizontal)

        # Для каждого результата создаем отдельное текстовое поле с увеличенным шрифтом.
        for title, result_text in [("BERT", self.bert_result.toPlainText()),
                                   ("LSTM", self.lstm_result.toPlainText()),
                                   ("DiTy", self.dity_result.toPlainText())]:
            container = QWidget()
            vbox = QVBoxLayout()
            label = QLabel(title)
            label.setAlignment(Qt.AlignCenter)
            vbox.addWidget(label)
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText(result_text)
            font = text_edit.font()
            font.setPointSize(font.pointSize() + 2)
            text_edit.setFont(font)
            vbox.addWidget(text_edit)
            container.setLayout(vbox)
            splitter.addWidget(container)

        splitter.setSizes([400, 400, 400])
        layout.addWidget(splitter)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        buttonBox.accepted.connect(dialog.accept)
        layout.addWidget(buttonBox)

        dialog.exec_()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: Arial;
                font-size: 18px;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005F99;
            }
            QLabel {
                font-weight: bold;
            }
        """)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec_())

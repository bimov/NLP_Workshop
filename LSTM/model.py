import torch
import torch.nn as nn
from LSTM.data_processing import text_to_indices
from torch.nn.utils.rnn import pad_sequence


class SiameseLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128, num_layers=1, dropout=0.3, bidirectional=False):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=False)
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, texts, word2idx):
        """
        Преобразует список текстов в последовательность индексов, добавляет паддинг и вычисляет
        эмбеддинги с последующим прохождением через LSTM.
        """
        sequences = [text_to_indices(text, word2idx) for text in texts]
        padded = pad_sequence(sequences, batch_first=True, padding_value=0).to(self.embedding.weight.device)
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long).to(self.embedding.weight.device)
        embeds = self.embedding(padded)
        packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hn, _) = self.lstm(packed)
        h_final = torch.cat((hn[-2], hn[-1]), dim=1) if self.bidirectional else hn[-1]
        return self.dropout(h_final)

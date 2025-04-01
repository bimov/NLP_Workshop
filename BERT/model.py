import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class SiameseNetwork(nn.Module):
    """
    Siamese Network на основе DistilRuBERT для обработки текстов на русском языке.
    """

    def __init__(self, model_name='DeepPavlov/distilrubert-base-cased-conversational', aggregation='mean'):
        super(SiameseNetwork, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.aggregation = aggregation

    def forward(self, texts):
        """
        Вычисляет эмбеддинги для списка текстов.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(next(self.encoder.parameters()).device) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        if self.aggregation == 'cls':
            embeddings = hidden_states[:, 0, :]
        elif self.aggregation == 'max':
            embeddings, _ = hidden_states.max(dim=1)
        else:
            embeddings = hidden_states.mean(dim=1)
        return embeddings

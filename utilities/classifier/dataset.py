import torch
import pandas as pd


# noinspection PyUnresolvedReferences
class Dataset(torch.utils.data.Dataset):
    """
    Custom datasets adapted from NLP coursework. This handles iteration and collation of batches.
    """

    def __init__(self, tokenizer, input_set: pd.DataFrame):
        self.tokenizer = tokenizer
        self.texts = input_set['text'].tolist()
        try:
            self.labels = [int(x) for x in input_set['label'].tolist()]
        except KeyError:
            self.labels = [-1 for _ in range(len(input_set))]

    def collate_fn(self, batch):
        texts = []
        labels = []

        for b in batch:
            texts.append(b['text'])
            labels.append(b['labels'])

        encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        encodings['labels'] = torch.tensor(labels)

        return encodings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'labels': self.labels[idx]
        }

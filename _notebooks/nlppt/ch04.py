from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch

import pandas as pd
import numpy as np
from io import StringIO
import requests


class GdriveHelper(object):
    SURNAME_DATASET = "1T1la2tYO1O7XkMRawG8VcFcvtjbxDqU-"

    @staticmethod
    def gdrive_csv_to_df(doc_id: str) -> pd.DataFrame:
        url = 'https://drive.google.com/uc?export=download&id={}'.format(doc_id)
        return pd.read_csv(StringIO(requests.get(url).text))

    @staticmethod
    def surname_df() -> pd.DataFrame:
        return GdriveHelper.gdrive_csv_to_df(GdriveHelper.SURNAME_DATASET)


def noneOrElse(o1, o2):
    if o1 is None:
        return o2
    return o1


def generate_batches(dataset: Dataset, batch_size: int, device: str='cpu', shuffle: bool=True, drop_last: bool=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        yield {k: tensor_val.to(device) for k, tensor_val in data_dict.items()}


class Tokenizer(object):
    def tokenize(self, sentence: str) -> list:
        return sentence.split(' ')


class Vocabulary(object):
    def __init__(self, token_to_idx: dict = None, unk_token: str = '<UNK>') -> None:
        self._token_to_idx = noneOrElse(token_to_idx, dict())
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        self._unk_token = unk_token
        self._unk_token_idx = self.add_token(self._unk_token)

    def add_token(self, token: str) -> int:
        if token in self._token_to_idx:
            return self._token_to_idx[token]

        last_index = len(self._token_to_idx)
        self._idx_to_token[last_index] = token
        self._token_to_idx[token] = last_index

    def token_for(self, index: int):
        return self._idx_to_token.get(index, self._unk_token_idx)

    def idx_for(self, token: str):
        # will force to raise KeyError
        return self._token_to_idx[token]

    def __len__(self):
        return len(self._token_to_idx)

    @staticmethod
    def from_df_series(df: pd.DataFrame, series_name: str, tokenizer: Tokenizer = Tokenizer()):
        def maybe_tokenize(value):
            if tokenizer is not None:
                return tokenizer.tokenize(value)

            return value

        def add_value(vocab, value):
            if type(value) == list:
                for val in value:
                    vocab.add_token(val)
            else:
                vocab.add_token(value)

        vocab = Vocabulary()
        for value in df[series_name].tolist():
            value = maybe_tokenize(value)
            add_value(vocab, value)
        return vocab


class Vectorizer(object):
    def __init__(self, vocabulary: Vocabulary) -> None:
        self._vocabulary = vocabulary

    def vectorize(self, token: str) -> np.array:
        pass


class OneHotVectorizer(Vectorizer):
    def vectorize(self, token: str) -> np.array:
        one_hot = np.zeros(len(self._vocabulary), dtype=np.float)
        one_hot[self._vocabulary.idx_for(token)] = 1
        return one_hot


class SurnameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, X_column: str, y_column: str, X_vectorizer: Vectorizer, y_vectorizer: Vectorizer) -> None:
        self.df = df
        self._vectorizer = X_vectorizer
        self._y_vectorizer = y_vectorizer

        self._init_df()

        self._target_df, self._target_size = None, None
        self._target_split = None
        self.set_split('train')

        self._X_column = X_column
        self._y_column = y_column

    def _init_df(self):
        self.train_df = self.df.query("split == 'train'")
        self.train_size = len(self.train_df)

        self.val_df = self.df.query("split == 'val'")
        self.val_size = len(self.val_df)

        self.test_df = self.df.query("split == 'test'")
        self.test_size = len(self.test_df)

        self._lookup_dict = dict(
            train=(self.train_df, self.train_size),
            val=(self.val_df, self.val_size),
            test=(self.test_df, self.test_size)
        )

    def get_vectorizer(self) -> Vectorizer:
        return self._vectorizer

    def set_split(self, split='train') -> None:
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __getitem__(self, index) -> dict:
        row = self._target_df.iloc[index]
        return dict(
            x=torch.from_numpy(self._vectorizer.vectorize(row[self._X_column])),
            y=torch.tensor(self._y_vectorizer._vocabulary.idx_for(row[self._y_column])).long()
        )

    def __len__(self) -> int:
        return self._target_size

    def n_batches(self, batch_size) -> int:
        return len(self) // batch_size


class SurnameClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, predict_proba=False):
        z = F.relu(self.fc1(X))
        z = self.fc2(z)

        if predict_proba:
            return F.softmax(z, dim=1)

        return z


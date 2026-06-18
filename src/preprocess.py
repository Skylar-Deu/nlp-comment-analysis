"""文本预处理模块 - 数据加载、清洗、Dataset实现"""

import re
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def clean_text(text):
    """清洗文本：去HTML标签、特殊字符，转小写"""
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)  # 去HTML标签
    text = re.sub(r'http\S+|www\S+', '', text)  # 去URL
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # 只保留字母数字
    text = re.sub(r'\s+', ' ', text).strip()  # 合并空格
    return text.lower()


class ToxicDataset(Dataset):
    """有害评论数据集"""

    LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.FloatTensor(self.labels[idx])
        }


class SentimentDataset(Dataset):
    """情感分析数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        label = 1 if self.labels[idx] == 'positive' else 0
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_toxic_data(filepath, test_size=0.2, sample_size=None):
    """加载有害评论数据"""
    df = pd.read_csv(filepath)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    texts = df['comment_text'].tolist()
    labels = df[ToxicDataset.LABEL_COLS].values.tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    return train_texts, val_texts, train_labels, val_labels


def load_sentiment_data(filepath, test_size=0.2, sample_size=None):
    """加载情感分析数据"""
    df = pd.read_csv(filepath)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    return train_texts, val_texts, train_labels, val_labels


def get_tokenizer(model_name='bert-base-uncased'):
    """获取BERT tokenizer"""
    return BertTokenizer.from_pretrained(model_name)

"""模型定义 - 传统ML、LSTM、BERT"""

import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
import numpy as np


# ==================== 传统ML模型 ====================

class TraditionalMLModel:
    """传统机器学习模型（TF-IDF + 分类器）"""

    def __init__(self, model_type='logistic', task='toxicity'):
        self.task = task
        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))

        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, C=1.0)
        elif model_type == 'svm':
            self.model = LinearSVC(max_iter=2000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if task == 'toxicity':
            self.model = OneVsRestClassifier(self.model)

    def fit(self, train_texts, train_labels):
        """训练模型"""
        X = self.vectorizer.fit_transform(train_texts)
        self.model.fit(X, train_labels)

    def predict(self, texts):
        """预测"""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts):
        """预测概率（如果支持）"""
        X = self.vectorizer.transform(texts)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return self.model.decision_function(X)


# ==================== LSTM模型 ====================

class LSTMModel(nn.Module):
    """双向LSTM + Attention模型"""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=6, dropout=0.3, task='toxicity'):
        super().__init__()
        self.task = task
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)

        # Attention机制
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        output = self.dropout(context)
        output = self.fc(output)

        if self.task == 'sentiment':
            return output
        return torch.sigmoid(output)


# ==================== BERT模型 ====================

class BertForToxicity(nn.Module):
    """BERT有害评论检测模型（多标签分类）"""

    def __init__(self, model_name='bert-base-uncased', num_labels=6, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}


class BertForSentiment(nn.Module):
    """BERT情感分析模型（二分类）"""

    def __init__(self, model_name='bert-base-uncased', freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}

"""训练脚本 - 统一训练接口"""

import os
import sys
import yaml
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    load_toxic_data, load_sentiment_data,
    ToxicDataset, SentimentDataset, get_tokenizer
)
from models import TraditionalMLModel, LSTMModel, BertForToxicity, BertForSentiment
from evaluate import evaluate_toxicity, evaluate_sentiment


def load_config():
    """加载配置"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(results, filepath):
    """保存训练结果为pkl文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """加载训练结果"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def train_bert_model(model, train_loader, val_loader, task, epochs=5, lr=2e-5, device='cpu'):
    """训练BERT模型"""
    model.to(device)
    lr = float(lr)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_f1 = 0
    history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []}

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        # 验证
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                outputs = model(input_ids, attention_mask)
                logits = outputs['logits'].cpu()

                if task == 'toxicity':
                    probs = torch.sigmoid(logits)
                    all_probs.extend(probs.numpy())
                    all_preds.extend((probs >= 0.5).int().numpy())
                else:
                    probs = torch.softmax(logits, dim=1)
                    all_probs.extend(probs.numpy())
                    all_preds.extend(torch.argmax(logits, dim=1).numpy())

                all_labels.extend(labels.numpy())

        # 计算指标
        if task == 'toxicity':
            metrics = evaluate_toxicity(all_labels, all_probs)
        else:
            metrics = evaluate_sentiment(all_labels, all_preds, np.array(all_probs))

        avg_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['f1'])
        history['val_auc'].append(metrics.get('auc', 0))

        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, "
              f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), f'models/{task}/best_model.pt')
            print(f"Saved best model (F1={best_f1:.4f})")

    # 保存训练结果
    results = {
        'model_type': 'bert',
        'task': task,
        'best_f1': best_f1,
        'final_metrics': metrics,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    save_results(results, f'models/{task}/bert_results.pkl')

    return model, results


def train_lstm_model(train_texts, train_labels, val_texts, val_labels,
                     task, epochs=10, batch_size=64, device='cpu'):
    """训练LSTM模型"""
    from collections import Counter

    # 构建词表
    words = ' '.join(train_texts).lower().split()
    word_counts = Counter(words)
    vocab = {w: i+2 for i, (w, _) in enumerate(word_counts.most_common(20000))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    def text_to_indices(texts, max_len=256):
        indices = []
        for text in texts:
            tokens = text.lower().split()[:max_len]
            idx = [vocab.get(t, 1) for t in tokens]
            idx += [0] * (max_len - len(idx))
            indices.append(idx)
        return np.array(indices)

    X_train = text_to_indices(train_texts)
    X_val = text_to_indices(val_texts)

    if task == 'toxicity':
        y_train = np.array(train_labels)
        y_val = np.array(val_labels)
        num_classes = 6
    else:
        y_train = np.array([1 if l == 'positive' else 0 for l in train_labels])
        y_val = np.array([1 if l == 'positive' else 0 for l in val_labels])
        num_classes = 2

    train_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(X_train), torch.FloatTensor(y_train) if task == 'toxicity' else torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(X_val), torch.FloatTensor(y_val) if task == 'toxicity' else torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = LSTMModel(len(vocab), num_classes=num_classes, task=task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if task == 'toxicity':
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    best_f1 = 0
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch)

                if task == 'toxicity':
                    all_preds.extend((pred.cpu() >= 0.5).int().numpy())
                else:
                    all_preds.extend(torch.argmax(pred, dim=1).cpu().numpy())

                all_labels.extend(y_batch.numpy())

        if task == 'toxicity':
            metrics = evaluate_toxicity(all_labels, all_preds)
        else:
            metrics = evaluate_sentiment(all_labels, all_preds)

        avg_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(metrics['accuracy'])
        history['val_f1'].append(metrics['f1'])

        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, "
              f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), f'models/{task}/best_lstm.pt')

    # 保存训练结果
    results = {
        'model_type': 'lstm',
        'task': task,
        'best_f1': best_f1,
        'final_metrics': metrics,
        'history': history,
        'vocab_size': len(vocab),
        'timestamp': datetime.now().isoformat()
    }
    save_results(results, f'models/{task}/lstm_results.pkl')

    return model, vocab, results


def train_traditional_model(train_texts, train_labels, val_texts, val_labels,
                            model_type='logistic', task='toxicity'):
    """训练传统ML模型"""
    model = TraditionalMLModel(model_type=model_type, task=task)
    print(f"Training {model_type} for {task}...")
    model.fit(train_texts, train_labels)

    preds = model.predict(val_texts)

    if task == 'toxicity':
        metrics = evaluate_toxicity(val_labels, preds)
    else:
        metrics = evaluate_sentiment(val_labels, preds)

    print(f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # 保存训练结果
    results = {
        'model_type': model_type,
        'task': task,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    save_results(results, f'models/{task}/{model_type}_results.pkl')

    return model, metrics, results


if __name__ == '__main__':
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_results = {}

    # ========== 情感分析 ==========
    print("\n" + "="*50)
    print("=== Sentiment Analysis ===")
    print("="*50)

    train_texts, val_texts, train_labels, val_labels = load_sentiment_data(
        config['data']['sentiment'], sample_size=5000
    )

    # 1. 传统ML模型
    print("\n--- Logistic Regression ---")
    _, _, lr_results = train_traditional_model(
        train_texts, train_labels, val_texts, val_labels,
        model_type='logistic', task='sentiment'
    )
    all_results['sentiment_lr'] = lr_results

    print("\n--- SVM ---")
    _, _, svm_results = train_traditional_model(
        train_texts, train_labels, val_texts, val_labels,
        model_type='svm', task='sentiment'
    )
    all_results['sentiment_svm'] = svm_results

    # 2. LSTM模型
    print("\n--- LSTM ---")
    _, _, lstm_results = train_lstm_model(
        train_texts, train_labels, val_texts, val_labels,
        task='sentiment', epochs=5, device=device
    )
    all_results['sentiment_lstm'] = lstm_results

    # 3. BERT模型
    print("\n--- BERT ---")
    tokenizer = get_tokenizer(config['bert']['model_name'])
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])

    model = BertForSentiment(config['bert']['model_name'])
    _, bert_results = train_bert_model(
        model, train_loader, val_loader,
        task='sentiment', epochs=config['training']['epochs'],
        lr=config['training']['learning_rate'], device=device
    )
    all_results['sentiment_bert'] = bert_results

    # ========== 有害评论检测 ==========
    print("\n" + "="*50)
    print("=== Toxicity Detection ===")
    print("="*50)

    train_texts, val_texts, train_labels, val_labels = load_toxic_data(
        config['data']['toxicity'], sample_size=5000
    )

    # 1. 传统ML模型
    print("\n--- Logistic Regression ---")
    _, _, lr_results = train_traditional_model(
        train_texts, train_labels, val_texts, val_labels,
        model_type='logistic', task='toxicity'
    )
    all_results['toxicity_lr'] = lr_results

    print("\n--- SVM ---")
    _, _, svm_results = train_traditional_model(
        train_texts, train_labels, val_texts, val_labels,
        model_type='svm', task='toxicity'
    )
    all_results['toxicity_svm'] = svm_results

    # 2. LSTM模型
    print("\n--- LSTM ---")
    _, _, lstm_results = train_lstm_model(
        train_texts, train_labels, val_texts, val_labels,
        task='toxicity', epochs=5, device=device
    )
    all_results['toxicity_lstm'] = lstm_results

    # 3. BERT模型
    print("\n--- BERT ---")
    tokenizer = get_tokenizer(config['bert']['model_name'])
    train_dataset = ToxicDataset(train_texts, train_labels, tokenizer)
    val_dataset = ToxicDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])

    model = BertForToxicity(config['bert']['model_name'])
    _, bert_results = train_bert_model(
        model, train_loader, val_loader,
        task='toxicity', epochs=config['training']['epochs'],
        lr=config['training']['learning_rate'], device=device
    )
    all_results['toxicity_bert'] = bert_results

    # 保存所有结果汇总
    save_results(all_results, 'models/all_results.pkl')
    print("\n" + "="*50)
    print("Training complete! All results saved.")
    print("="*50)

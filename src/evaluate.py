"""评估模块 - 指标计算、可视化"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)


def evaluate_toxicity(y_true, y_pred, threshold=0.5):
    """评估有害评论检测（多标签）"""
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
    y_true = np.array(y_true)

    report = classification_report(
        y_true, y_pred_binary,
        target_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
        output_dict=True
    )

    try:
        auc = roc_auc_score(y_true, y_pred, average='macro')
    except:
        auc = 0.0

    return {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'auc': auc,
        'report': report
    }


def evaluate_sentiment(y_true, y_pred, y_proba=None):
    """评估情感分析（二分类）"""
    report = classification_report(
        y_true, y_pred,
        target_names=['negative', 'positive'],
        output_dict=True
    )

    auc = 0.0
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        except:
            pass

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'auc': auc,
        'report': report
    }


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    return plt


def plot_roc_curve(y_true, y_score, title='ROC Curve'):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt


def plot_comparison(results, metric='f1'):
    """绘制模型对比图"""
    models = list(results.keys())
    values = [results[m][metric] for m in models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=['#2196F3', '#4CAF50', '#FF9800'])
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.ylabel(metric.upper())
    plt.ylim(0, 1)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    return plt

# NLP Comment Analysis

有害评论识别 + 情感分析系统

基于传统机器学习、LSTM 和 BERT 的文本分类对比实验，包含 Streamlit Web 界面。

## 功能

- **有害评论检测**：识别文本中的有害内容（辱骂、歧视、威胁等），多标签分类
- **情感分析**：判断文本的情感倾向（正面/负面），二分类
- **Web 界面**：基于 Streamlit 的交互式分析界面，支持单条分析和批量分析

## 模型对比

| 模型 | 情感分析 F1 | 有害检测 F1 | 说明 |
|------|------------|------------|------|
| Logistic Regression | 0.856 | 0.097 | 基线模型 |
| SVM | 0.878 | 0.503 | 传统ML最佳 |
| LSTM | 0.810 | 0.295 | 深度学习基线 |
| **BERT** | **0.908** | **0.702** | 最佳模型 |

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/nlp-comment-analysis.git
cd nlp-comment-analysis
```

### 2. 安装依赖

```bash
conda create -n nlp python=3.11
conda activate nlp
pip install -r requirements.txt
```

### 3. 下载数据集

从 Kaggle 下载数据集，放入 `data/raw/` 目录：

| 数据集 | 下载链接 | 保存文件名 |
|--------|----------|-----------|
| Jigsaw Toxic Comment | [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) | `data/raw/toxic_en.csv` |
| IMDB Movie Reviews | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | `data/raw/imdb.csv` |

**注意**：
- Jigsaw 数据集下载后是压缩包，解压后找到 `train.csv`，重命名为 `toxic_en.csv`
- IMDB 数据集下载后找到 `IMDB Dataset.csv`，重命名为 `imdb.csv`

### 4. 训练模型

```bash
python src/train.py
```

训练完成后，模型权重和结果保存在 `models/` 目录。

### 5. 启动 Web 界面

```bash
streamlit run app.py
```

访问 http://localhost:8501

## 项目结构

```
├── app.py                    # Streamlit Web界面
├── config.yaml               # 项目配置
├── requirements.txt          # Python依赖
├── data/
│   ├── raw/                  # 原始数据集（需自行下载）
│   └── processed/            # 处理后数据
├── models/                   # 训练好的模型（需自行训练）
│   ├── toxicity/
│   └── sentiment/
└── src/
    ├── preprocess.py         # 文本预处理
    ├── models.py             # 模型定义（传统ML + LSTM + BERT）
    ├── train.py              # 训练脚本
    └── evaluate.py           # 评估模块
```

## 技术方案

### 文本预处理

- 去除 HTML 标签、URL、特殊字符
- 转小写、合并空格

### 传统机器学习

- 特征提取：TF-IDF（max_features=50000, ngram_range=(1,2)）
- 分类器：Logistic Regression、SVM
- 多标签处理：OneVsRestClassifier

### LSTM

- 词嵌入 + 双向 LSTM + Attention 机制
- 词表大小：20,000
- 隐藏层维度：256

### BERT

- 预训练模型：`bert-base-uncased`
- 微调策略：全参数微调
- 学习率：2e-5
- Epochs：5

## 依赖

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.30
- Streamlit >= 1.28
- scikit-learn >= 1.2
- pandas, numpy, matplotlib, tqdm

## 许可证

MIT License

## 参考

- [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

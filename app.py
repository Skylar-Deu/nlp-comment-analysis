"""Streamlit界面 - 有害评论识别 + 情感分析"""

import os
import sys
import yaml
import torch
import streamlit as st
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import clean_text, get_tokenizer
from models import BertForToxicity, BertForSentiment, TraditionalMLModel


@st.cache_resource
def load_config():
    """加载配置"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_tokenizer():
    """加载tokenizer"""
    config = load_config()
    return get_tokenizer(config['bert']['model_name'])


@st.cache_resource
def load_toxicity_model():
    """预加载有害评论检测模型"""
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForToxicity(config['bert']['model_name'])
    model_path = 'models/toxicity/best_model.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model.to(device)
    return None


@st.cache_resource
def load_sentiment_model():
    """预加载情感分析模型"""
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSentiment(config['bert']['model_name'])
    model_path = 'models/sentiment/best_model.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model.to(device)
    return None


def predict_toxicity(text, model, tokenizer, device):
    """预测有害评论"""
    model.eval()
    clean = clean_text(text)
    encoding = tokenizer(clean, max_length=256, padding='max_length',
                        truncation=True, return_tensors='pt')

    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs['logits']).cpu().numpy()[0]

    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return dict(zip(labels, probs.tolist()))


def predict_sentiment(text, model, tokenizer, device):
    """预测情感"""
    model.eval()
    clean = clean_text(text)
    encoding = tokenizer(clean, max_length=256, padding='max_length',
                        truncation=True, return_tensors='pt')

    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs['logits'], dim=1).cpu().numpy()[0]

    return {'negative': float(probs[0]), 'positive': float(probs[1])}


def main():
    st.set_page_config(
        page_title="NLP Comment Analysis",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 NLP Comment Analysis")
    st.markdown("**有害评论识别 + 情感分析**")

    # 侧边栏
    st.sidebar.title("Settings")
    task = st.sidebar.selectbox("选择任务", ["有害评论检测", "情感分析", "批量分析"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = load_tokenizer()

    # 预加载模型（首次加载会慢，之后秒开）
    toxicity_model = load_toxicity_model()
    sentiment_model = load_sentiment_model()

    # 有害评论检测
    if task == "有害评论检测":
        st.header("有害评论检测")

        if toxicity_model is None:
            st.error("未找到训练好的模型，请先运行训练脚本")
        else:
            text = st.text_area("输入评论文本", height=150,
                               placeholder="Enter a comment to analyze...")

            if st.button("分析") and text:
                results = predict_toxicity(text, toxicity_model, tokenizer, device)

                st.subheader("分析结果")
                col1, col2 = st.columns(2)

                with col1:
                    for label, score in results.items():
                        color = "red" if score > 0.5 else "green"
                        st.markdown(f"**{label}**: :{color}[{score:.2%}]")
                        st.progress(score)

                with col2:
                    is_toxic = any(v > 0.5 for v in results.values())
                    if is_toxic:
                        st.error("⚠️ 检测到有害内容！")
                    else:
                        st.success("✅ 未检测到有害内容")

    # 情感分析
    elif task == "情感分析":
        st.header("情感分析")

        if sentiment_model is None:
            st.error("未找到训练好的模型，请先运行训练脚本")
        else:
            text = st.text_area("输入评论文本", height=150,
                               placeholder="Enter a review to analyze...")

            if st.button("分析") and text:
                results = predict_sentiment(text, sentiment_model, tokenizer, device)

                st.subheader("分析结果")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Positive**: :green[{results['positive']:.2%}]")
                    st.progress(results['positive'])
                    st.markdown(f"**Negative**: :red[{results['negative']:.2%}]")
                    st.progress(results['negative'])

                with col2:
                    sentiment = "Positive 😊" if results['positive'] > 0.5 else "Negative 😞"
                    st.markdown(f"### 情感倾向: {sentiment}")

    # 批量分析
    elif task == "批量分析":
        st.header("批量分析")
        st.info("上传CSV文件进行批量分析（需要包含text列）")

        uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])

        if uploaded_file:
            import pandas as pd
            df = pd.read_csv(uploaded_file)

            if 'text' in df.columns:
                st.dataframe(df.head())

                if st.button("开始分析"):
                    if toxicity_model and sentiment_model:
                        results = []
                        progress = st.progress(0)

                        for i, row in df.iterrows():
                            text = row['text']
                            toxic_result = predict_toxicity(text, toxicity_model, tokenizer, device)
                            sentiment_result = predict_sentiment(text, sentiment_model, tokenizer, device)

                            results.append({
                                'text': text[:100] + '...',
                                'is_toxic': any(v > 0.5 for v in toxic_result.values()),
                                'sentiment': 'positive' if sentiment_result['positive'] > 0.5 else 'negative',
                                'toxic_score': max(toxic_result.values()),
                                'positive_score': sentiment_result['positive']
                            })

                            progress.progress((i + 1) / len(df))

                        result_df = pd.DataFrame(results)
                        st.dataframe(result_df)

                        # 统计
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("有害评论数", result_df['is_toxic'].sum())
                        with col2:
                            st.metric("正面评论数", (result_df['sentiment'] == 'positive').sum())
                    else:
                        st.warning("请先加载所有模型")
            else:
                st.error("CSV文件需要包含'text'列")

    # 底部信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 关于")
    st.sidebar.info("NLP项目：有害评论识别 + 情感分析\n\n使用BERT模型进行分析")


if __name__ == '__main__':
    main()

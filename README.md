
# 🎬 IMDB Sentiment Analysis with BERT (PyTorch)

基于 BERT 的 IMDB 情感分析模型，使用 PyTorch 与 Hugging Face Transformers 实现。

Sentiment analysis model using BERT, built with PyTorch and Hugging Face Transformers on the IMDB movie reviews dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-ee4c2c?style=flat-square&logo=pytorch)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=flat-square&logo=huggingface)

---

## 📌 项目简介 | Project Overview

本项目通过微调预训练的 `bert-base-uncased` 模型，对 IMDB 电影评论进行二分类（正面 / 负面情感），最终模型在测试集上达到 **93.8% 准确率** 和 **0.938 F1-score**。

This project fine-tunes the `bert-base-uncased` model for binary sentiment classification on the IMDB movie review dataset. The final model achieves **93.8% accuracy** and **0.938 F1-score** on the test set.

---

## 💡 特点亮点 | Highlights

- 🔍 **完整流程**：数据加载 → 预处理 → 模型微调 → 推理 → 可视化
- 🧠 **高性能模型**：基于 BERT 的 Transformer 架构，效果显著
- 🔁 **交互式预测**：支持输入评论并预测情感
- 📈 **系统优化**：通过调整 batch size、max_len、warmup 等超参提升效果

- 🔍 **End-to-End Pipeline**: Data loading → Preprocessing → Fine-tuning → Inference → Visualization  
- 🧠 **High Performance**: Based on Transformer (BERT), excellent real-world results  
- 🔁 **Interactive Prediction**: Real-time prediction for new reviews  
- 📈 **Optimization**: Tuned hyperparameters (batch size, max_len, warmup ratio, etc.)  

---

## 🗂️ 项目结构 | Project Structure

```text
pytorch_imdb_sentiment_analysis/
├── data/                   # 数据集目录 / Dataset
├── models/                 # 模型定义 / Model wrapper
├── utils/                  # 工具函数 / Preprocessing & training utils
├── notebooks/              # Jupyter Notebook (主实验入口)
│   └── imdb_sentiment_analysis_bert.ipynb
├── main.py                 # 可选：脚本入口
├── requirements.txt        # 依赖列表 / Dependencies
└── README.md


⸻

🚀 快速开始 | Quick Start
	1.	克隆项目 / Clone this repo

git clone https://github.com/DWGqaz123/pytorch_imdb_sentiment_analysis.git
cd pytorch_imdb_sentiment_analysis

	2.	创建环境 / Create virtual env

conda create -n imdb_sentiment python=3.10
conda activate imdb_sentiment
pip install -r requirements.txt

	3.	下载数据集 / Download dataset
👉 http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
解压至 ./data/aclImdb/ 并保留原结构。
Extract to ./data/aclImdb/ with subfolders train/ and test/.
	4.	运行 Notebook / Run Notebook

jupyter lab

打开并运行 notebooks/imdb_sentiment_analysis_bert.ipynb。

Open and run notebooks/imdb_sentiment_analysis_bert.ipynb.

⸻

🤖 示例预测 | Example Prediction

# 示例代码 / Sample usage
review = "This movie was incredibly moving and thought-provoking."
sentiment, prob = predict_sentiment(review, model, tokenizer, device)
print(f"Sentiment: {sentiment}, Confidence: {prob:.4f}")

📎 License

MIT License © 2025 Dong Wenguang

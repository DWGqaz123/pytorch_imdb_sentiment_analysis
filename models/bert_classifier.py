# models/bert_classifier.py
import torch.nn as nn
from transformers import BertForSequenceClassification

class BERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super(BERTClassifier, self).__init__()
        # 加载预训练的 BERT 模型
        # BertForSequenceClassification 已经包含了 BERT 主体和一个分类头
        # num_labels 对应你的分类任务的类别数
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Hugging Face 模型的前向传播
        # 返回一个字典，包含 loss (如果提供了 labels) 和 logits
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
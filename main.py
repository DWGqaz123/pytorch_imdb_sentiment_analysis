# main.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler

# 导入我们自己定义的模块
import sys
# 确保sys.path包含项目根目录，以便导入models和utils
# 这个路径根据main.py相对于项目根目录的位置来调整
# 如果main.py在根目录下，append('.')
# 如果main.py在某个子目录下，则需要相应调整
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

from utils.text_preprocessing import load_imdb_data, clean_text, IMDBDatasetWithBERT
from models.bert_classifier import BERTClassifier
from utils.train_eval_utils import set_seed # 仅设置种子，训练评估循环直接在main中实现

# --- 配置参数 ---
# 建议将这些参数作为命令行参数，以便灵活调整
def get_args():
    parser = argparse.ArgumentParser(description="BERT Sentiment Analysis on IMDB Dataset")

    # General
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, mps, auto)')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for IMDB dataset')
    parser.add_argument('--model_checkpoint_dir', type=str, default='./model_checkpoints/bert_imdb_sentiment', help='Directory to save/load model checkpoints')
    
    # Training
    parser.add_argument('--train', action='store_true', help='Flag to perform training')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length for tokenizer')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Initial learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer') # BERT微调通常不设高权重衰减
    parser.add_argument('--warmup_ratio', type=float, default=0.06, help='Warmup steps ratio for LR scheduler')
    parser.add_argument('--scheduler_type', type=str, default='linear', help='LR scheduler type (linear, cosine)')

    # Inference
    parser.add_argument('--predict', action='store_true', help='Flag to perform inference on a given text')
    parser.add_argument('--text', type=str, default="This movie was absolutely brilliant! A true masterpiece.", help='Text to perform sentiment prediction on')

    args = parser.parse_args()
    return args

# --- Main Functions ---

def train(args):
    # Device setup
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device for training.")
        else:
            device = torch.device("cpu")
            print("MPS device not available, using CPU for training.")
    else:
        device = torch.device(args.device)
        print(f"Using specified device: {device} for training.")

    set_seed(args.seed)

    # 1. Data Loading and Preprocessing
    IMDB_DATA_ROOT = os.path.join(args.data_root, 'aclImdb')
    
    print("Loading real IMDB training data...")
    raw_train_data = load_imdb_data(os.path.join(IMDB_DATA_ROOT, 'train'))
    print("Cleaning training data...")
    cleaned_train_data = [(label, clean_text(text)) for label, text in raw_train_data]

    print("Loading real IMDB testing data...")
    raw_test_data = load_imdb_data(os.path.join(IMDB_DATA_ROOT, 'test'))
    print("Cleaning testing data...")
    cleaned_test_data = [(label, clean_text(text)) for label, text in raw_test_data]

    print(f"Loaded {len(cleaned_train_data)} training samples.")
    print(f"Loaded {len(cleaned_test_data)} testing samples.")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = IMDBDatasetWithBERT(cleaned_train_data, tokenizer, args.max_len)
    test_dataset = IMDBDatasetWithBERT(cleaned_test_data, tokenizer, args.max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Model, Optimizer, Scheduler
    model = BERTClassifier(num_labels=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_scheduler(
        args.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 3. Training Loop
    print(f"\n--- Starting BERT Fine-tuning for {args.num_epochs} epochs ---")
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0
        correct_train_predictions = 0
        total_train_samples = 0

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(logits, dim=1)
            correct_train_predictions += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

            if batch_idx % 500 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = correct_train_predictions / total_train_samples * 100
        print(f"Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    print("--- BERT Fine-tuning Finished ---")

    # 4. Evaluation
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    print("\n--- Starting BERT Evaluation ---")
    model.eval() 
    all_labels = []
    all_preds = []
    total_eval_loss = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()

            _, predicted = torch.max(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_eval_loss = total_eval_loss / len(test_dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1)

    print(f"Evaluation completed. Avg Test Loss: {avg_eval_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("--- BERT Evaluation Finished ---")

    # 5. Save Model
    os.makedirs(args.model_checkpoint_dir, exist_ok=True)
    print(f"\nSaving model and tokenizer to {args.model_checkpoint_dir}")
    model.bert.save_pretrained(args.model_checkpoint_dir) # 保存的是BertForSequenceClassification的父类
    tokenizer.save_pretrained(args.model_checkpoint_dir)
    print("Model saved.")

def predict(args):
    # Device setup
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device for prediction.")
        else:
            device = torch.device("cpu")
            print("MPS device not available, using CPU for prediction.")
    else:
        device = torch.device(args.device)
        print(f"Using specified device: {device} for prediction.")

    # Load model and tokenizer
    print(f"\nLoading model from {args.model_checkpoint_dir}")
    model = BertForSequenceClassification.from_pretrained(args.model_checkpoint_dir)
    tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint_dir)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    def predict_sentiment_func(text, model, tokenizer, device, max_len):
        model.eval()
        cleaned_text = clean_text(text)
        encoding = tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=1)
        positive_probability = probs[:, 1].item()
        predicted_class_id = torch.argmax(probs, dim=1).item()
        sentiment = "Positive" if predicted_class_id == 1 else "Negative"
        return sentiment, positive_probability

    sentiment, prob = predict_sentiment_func(args.text, model, tokenizer, device, args.max_len)
    print(f"\nReview: '{args.text}'")
    print(f"Predicted Sentiment: {sentiment} (Positive Probability: {prob:.4f})")

# --- Main execution block ---
if __name__ == "__main__":
    args = get_args()

    # Determine device for main.py's own use, not for model's
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            print("MPS is available. Model will use MPS.")
        else:
            print("MPS not available. Model will use CPU.")
    
    if args.train:
        train(args)
    elif args.predict:
        predict(args)
    else:
        print("Please specify --train to start training or --predict to make a prediction.")
        print("Example: python main.py --train")
        print("Example: python main.py --predict --text 'This movie was amazing and I loved every minute of it.'")
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import re
from tqdm import tqdm
import os

# ==========================================
# 1. 配置
# ==========================================
CONFIG = {
    'model_name': 'bert-base-uncased', # 或者 'distilbert-base-uncased' (速度更快)
    'max_len': 128,          # 食材文本的最大长度
    'batch_size': 32,        # 显存够大(5070Ti)可以设为 32 或 64
    'epochs': 10,             # BERT 微调通常 3-5 轮即可
    'lr': 3e-5,              # BERT 常用学习率
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(CONFIG['seed'])
print(f"Using device: {CONFIG['device']}")

# ==========================================
# 2. 数据清洗 (关键步骤)
# ==========================================
def clean_ingredient(text):
    # 转小写
    text = text.lower()
    
    # 去除特殊符号
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 常见的无用修饰词 (可以根据数据分析继续扩充)
    stop_words = [
        'fresh', 'ground', 'chopped', 'sliced', 'diced', 'crushed', 'minced', 'grated', 
        'large', 'medium', 'small', 'cloves', 'lb', 'oz', 'drained', 'pitted', 'beaten', 
        'unsalted', 'all-purpose', 'chunks', 'dried', 'leaves', 'powder', 'frozen', 'warm'
    ]
    
    # 移除这些词
    for word in stop_words:
        text = re.sub(r'\b' + word + r'\b', '', text)
        
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(file_path, is_train=True):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # 将 list 里的食材清洗后，用逗号连接成字符串
    # 例如: ["fresh garlic", "soy sauce"] -> "garlic, soy sauce"
    # BERT 能够理解这种序列
    df['text'] = df['ingredients'].apply(
        lambda x: ', '.join([clean_ingredient(ing) for ing in x])
    )
    
    return df

# ==========================================
# 3. Dataset 定义
# ==========================================
class CuisineBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

# ==========================================
# 4. 训练函数
# ==========================================
def train_epoch(model, data_loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0
    correct_predictions = 0
    n_examples = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for d in progress_bar:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # HuggingFace 的分类模型输出是 logits
        loss = loss_fn(outputs.logits, targets)
        
        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == targets)
        n_examples += targets.size(0)
        
        total_loss += loss.item()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': loss.item()})
        
    return correct_predictions.double() / n_examples, total_loss / len(data_loader)

def eval_model(model, data_loader, device, loss_fn):
    model.eval()
    correct_predictions = 0
    n_examples = 0
    total_loss = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = loss_fn(outputs.logits, targets)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == targets)
            n_examples += targets.size(0)
            
    return correct_predictions.double() / n_examples, total_loss / len(data_loader)

# ==========================================
# 5. 主程序
# ==========================================
def main():
    # --- 1. 数据加载与处理 ---
    print("Loading and preprocessing data...")
    train_df = preprocess_data('train.json', is_train=True)
    test_df = preprocess_data('test.json', is_train=False)
    
    # 标签编码
    label_encoder = LabelEncoder()
    train_df['label_id'] = label_encoder.fit_transform(train_df['cuisine'])
    num_classes = len(label_encoder.classes_)
    
    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['text'].values, 
        train_df['label_id'].values, 
        test_size=0.1, 
        random_state=CONFIG['seed'],
        stratify=train_df['label_id'].values
    )
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # --- 2. Tokenizer & DataLoader ---
    print(f"Loading BERT tokenizer ({CONFIG['model_name']})...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    
    train_dataset = CuisineBertDataset(X_train, y_train, tokenizer, CONFIG['max_len'])
    val_dataset = CuisineBertDataset(X_val, y_val, tokenizer, CONFIG['max_len'])
    test_dataset = CuisineBertDataset(test_df['text'].values, None, tokenizer, CONFIG['max_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # --- 3. 模型初始化 ---
    print("Initializing BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False
    )
    model = model.to(CONFIG['device'])
    
    # 优化器设置：使用 AdamW，并带有 Warmup
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'])
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1), # 10% warmup
        num_training_steps=total_steps
    )
    
    # 损失函数 (Label Smoothing 有助于防止过拟合)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(CONFIG['device'])
    
    # --- 4. 训练循环 ---
    best_acc = 0
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        
        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, CONFIG['device'], loss_fn
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        val_acc, val_loss = eval_model(
            model, val_loader, CONFIG['device'], loss_fn
        )
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_bert_model.pth')
            print("=> Model saved!")
            
    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    
    # --- 5. 预测与提交 ---
    print("Generating predictions on Test set...")
    model.load_state_dict(torch.load('best_bert_model.pth'))
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for d in tqdm(test_loader, desc="Predicting"):
            input_ids = d["input_ids"].to(CONFIG['device'])
            attention_mask = d["attention_mask"].to(CONFIG['device'])
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            
    # 转换回文本标签
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'cuisine': predicted_labels
    })
    
    submission.to_csv('submission_bert.csv', index=False)
    print("Done! Saved to submission_bert.csv")

if __name__ == '__main__':
    main()
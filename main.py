import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import time

# ==========================================
# 1. 配置与环境设置
# ==========================================
CONFIG = {
    'min_freq': 2,          # 过滤掉出现次数少于2次的稀有食材，减少噪音
    'batch_size': 128,      # 批次大小
    'lr': 0.001,            # 学习率
    'epochs': 15,           # 训练轮数
    'hidden_dim': 512,      # 隐藏层维度
    'dropout': 0.5,         # Dropout比例
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Running on device: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. 数据预处理工具类
# ==========================================
class IngredientTokenizer:
    """
    负责将食材列表转换为 Multi-Hot 向量 (0/1 向量)
    """
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.vocab = {}
        self.idx2word = {}
        
    def fit(self, ingredients_list):
        # 统计词频
        counter = Counter()
        for ingredients in ingredients_list:
            counter.update(ingredients)
        
        # 构建词表 (过滤低频词)
        valid_ingredients = [word for word, count in counter.items() if count >= self.min_freq]
        self.vocab = {word: idx for idx, word in enumerate(valid_ingredients)}
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)} (Original unique ingredients: {len(counter)})")
        
    def transform(self, ingredients_list):
        # 转换为 Multi-Hot 向量
        dim = len(self.vocab)
        vectors = np.zeros((len(ingredients_list), dim), dtype=np.float32)
        
        for i, ingredients in enumerate(ingredients_list):
            for ing in ingredients:
                if ing in self.vocab:
                    vectors[i, self.vocab[ing]] = 1.0
        return vectors

    def __len__(self):
        return len(self.vocab)

# ==========================================
# 3. PyTorch Dataset 定义
# ==========================================
class CuisineDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

# ==========================================
# 4. 模型定义 (MLP)
# ==========================================
class CuisineClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.5):
        super(CuisineClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # BN层加速收敛
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 5. 主程序逻辑
# ==========================================
def load_data():
    print("Loading data...")
    with open('train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 提取数据
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    return train_df, test_df

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    best_acc = 0.0
    
    for epoch in range(CONFIG['epochs']):
        # --- Training ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
    print(f"Training finished. Best Validation Accuracy: {best_acc:.2f}%")

def main():
    # 1. 读取数据
    train_df, test_df = load_data()
    
    # 2. 标签编码 (String -> Int)
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['cuisine'])
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")

    # 3. 特征工程 (Ingredients -> Multi-Hot Vectors)
    tokenizer = IngredientTokenizer(min_freq=CONFIG['min_freq'])
    tokenizer.fit(train_df['ingredients']) # 仅基于训练集构建词表
    
    X_all = tokenizer.transform(train_df['ingredients'])
    X_test = tokenizer.transform(test_df['ingredients'])
    
    # 4. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
    
    # 5. 构建 DataLoader
    # 注意: Windows下 num_workers 建议设为 0，否则可能报错
    train_dataset = CuisineDataset(X_train, y_train)
    val_dataset = CuisineDataset(X_val, y_val)
    test_dataset = CuisineDataset(X_test)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # 6. 初始化模型、损失函数、优化器
    model = CuisineClassifier(input_dim=len(tokenizer), output_dim=num_classes, hidden_dim=CONFIG['hidden_dim'], dropout=CONFIG['dropout'])
    model = model.to(CONFIG['device'])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5) # weight_decay用于L2正则化
    
    # 7. 训练
    train_model(model, train_loader, val_loader, criterion, optimizer, CONFIG['device'])
    
    # 8. 预测 Test 集
    print("Generating predictions on test set...")
    # 加载最佳模型参数
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(CONFIG['device'])
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            
    # 9. 将数字标签转换回字符串
    predicted_cuisines = label_encoder.inverse_transform(predictions)
    
    # 10. 保存结果
    submission = pd.DataFrame({
        'id': test_df['id'],
        'cuisine': predicted_cuisines
    })
    submission.to_csv('submission.csv', index=False)
    print("Done! Result saved to submission.csv")

if __name__ == '__main__':
    main()
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import re
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer

# ==========================================
# 0. 环境与配置
# ==========================================
# 第一次运行需要下载 nltk 数据
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

CONFIG = {
    'seed': 42,
    'roberta_model': 'roberta-base',
    'max_len': 128,
    'batch_size': 32,      # 5070Ti 显存充足，可设为 32 或 64
    'epochs': 12,           # RoBERTa 训练轮数
    'lr': 2e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'ensemble_weight': 0.5 # 0.5 表示 SVC 和 RoBERTa 各占 50% 权重
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['seed'])
print(f"Running on: {CONFIG['device']}")

# ==========================================
# 1. 强力数据清洗
# ==========================================
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # 1. 转小写
    text = text.lower()
    # 2. 去除非字母字符 (保留空格和连字符)
    text = re.sub(r'[^a-z\s-]', '', text)
    # 3. 去除无意义的单位和修饰词
    stop_words = set([
        'fresh', 'ground', 'chopped', 'sliced', 'diced', 'crushed', 'minced', 'grated', 
        'large', 'medium', 'small', 'cloves', 'lb', 'oz', 'drained', 'pitted', 'beaten', 
        'unsalted', 'all-purpose', 'chunks', 'dried', 'leaves', 'powder', 'frozen', 'warm',
        'melted', 'boneless', 'skinless', 'halves'
    ])
    words = [w for w in text.split() if w not in stop_words]
    # 4. 词形还原 (olives -> olive)
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

def load_data():
    print("Loading data...")
    with open('train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # 清洗 ingredients
    print("Cleaning ingredients...")
    # 保存原始列表用于 Deep Learning 的 Shuffle
    train_df['clean_list'] = train_df['ingredients'].apply(lambda x: [clean_text(i) for i in x])
    test_df['clean_list'] = test_df['ingredients'].apply(lambda x: [clean_text(i) for i in x])
    
    # 生成用于 TF-IDF 的字符串
    train_df['text_str'] = train_df['clean_list'].apply(lambda x: ' '.join(x))
    test_df['text_str'] = test_df['clean_list'].apply(lambda x: ' '.join(x))
    
    return train_df, test_df

# ==========================================
# 2. 模型 A: TF-IDF + LinearSVC (传统强项)
# ==========================================
def train_svc(train_df, test_df, y_train_enc):
    print("\n[Model A] Training LinearSVC...")
    
    # TF-IDF 特征提取 (包含 1-gram 和 2-gram)
    tfidf = TfidfVectorizer(binary=True, ngram_range=(1, 2), min_df=3, max_df=0.9)
    X_train_tfidf = tfidf.fit_transform(train_df['text_str'])
    X_test_tfidf = tfidf.transform(test_df['text_str'])
    
    # LinearSVC 本身不支持 predict_proba，需配合 CalibratedClassifierCV
    svc = LinearSVC(C=0.8, penalty='l2', dual=False, max_iter=3000, random_state=CONFIG['seed'])
    clf = CalibratedClassifierCV(svc, method='sigmoid', cv=5)
    
    clf.fit(X_train_tfidf, y_train_enc)
    
    # 获取概率预测
    probs = clf.predict_proba(X_test_tfidf)
    print("SVC training done.")
    return probs

# ==========================================
# 3. 模型 B: RoBERTa + Shuffle Augmentation
# ==========================================
class CuisineRobertaDataset(Dataset):
    def __init__(self, ingredients_lists, labels, tokenizer, max_len, is_train=False):
        self.ingredients_lists = ingredients_lists
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train # 如果是训练模式，开启 Shuffle
        
    def __len__(self):
        return len(self.ingredients_lists)
    
    def __getitem__(self, idx):
        ingredients = self.ingredients_lists[idx]
        
        # --- 关键策略: 训练时随机打乱食材顺序 ---
        if self.is_train:
            ingredients = list(ingredients) # 复制副本
            np.random.shuffle(ingredients)
            
        text = ", ".join(ingredients) # 使用逗号分隔对 RoBERTa 更友好
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
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

def train_roberta(train_df, test_df, y_train_enc, num_classes):
    print("\n[Model B] Training RoBERTa with Shuffle Augmentation...")
    
    tokenizer = RobertaTokenizer.from_pretrained(CONFIG['roberta_model'])
    
    # 全量数据训练 (不再划分验证集，为了最大化利用数据冲榜)
    # 如果你想看验证集分数，可以自己 split，但提交时建议用全量
    train_dataset = CuisineRobertaDataset(
        train_df['clean_list'].values, 
        y_train_enc, 
        tokenizer, 
        CONFIG['max_len'], 
        is_train=True # 开启打乱
    )
    
    test_dataset = CuisineRobertaDataset(
        test_df['clean_list'].values, 
        None, 
        tokenizer, 
        CONFIG['max_len'], 
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    model = RobertaForSequenceClassification.from_pretrained(
        CONFIG['roberta_model'],
        num_labels=num_classes
    )
    model.to(CONFIG['device'])
    
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * CONFIG['epochs']
    )
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) # 标签平滑
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['labels'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    print("RoBERTa training finished. Generating predictions...")
    
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            
            outputs = model(input_ids, attention_mask=attention_mask)
            # 使用 Softmax 获取概率
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_probs, axis=0)

# ==========================================
# 4. 主流程：集成 (Ensemble)
# ==========================================
def main():
    train_df, test_df = load_data()
    
    # 标签编码
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_df['cuisine'])
    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}")
    
    # 1. 获取 SVC 的概率预测
    svc_probs = train_svc(train_df, test_df, y_train_enc)
    
    # 2. 获取 RoBERTa 的概率预测
    roberta_probs = train_roberta(train_df, test_df, y_train_enc, num_classes)
    
    # 3. 加权融合 (Soft Voting)
    print("\nBlending models...")
    # 可以根据验证集结果调整权重，通常 SVC 在此任务很强，5:5 开或者 SVC 占 0.4 都可以
    final_probs = (svc_probs * 0.5) + (roberta_probs * 0.5)
    
    # 4. 取最大概率对应的标签
    final_preds = np.argmax(final_probs, axis=1)
    final_labels = le.inverse_transform(final_preds)
    
    # 5. 保存
    submission = pd.DataFrame({
        'id': test_df['id'],
        'cuisine': final_labels
    })
    submission.to_csv('submission_ensemble.csv', index=False)
    print("Done! Result saved to submission_ensemble.csv")

if __name__ == '__main__':
    main()
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
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import re
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
import gc
import warnings

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 高端配置 (稳健版)
# ==========================================
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

CONFIG = {
    'seed': 2024,
    'model_name': 'microsoft/deberta-v3-large', 
    'max_len': 128,
    
    # 显存/稳定性优化配置 (你的修改版)
    'batch_size': 2,        
    'accum_steps': 16,        
    'use_checkpointing': False, 
    
    'epochs': 10, # 有了验证集，我们可以少跑几轮，观察收敛即可
    'lr': 1e-5,              
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'pseudo_threshold': 0.95 
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])
print(f"🚀 Running IMPROVED solution on: {CONFIG['device']} with {CONFIG['model_name']}")

# ==========================================
# 1. 增强版数据清洗 (Data Cleaning)
# ==========================================
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    
    # [修改1] 将连字符替换为空格 (low-fat -> low fat)，方便去除 split 后的单词
    text = text.replace('-', ' ')
    
    # [修改2] 仅保留字母和空格
    text = re.sub(r'[^a-z\s]', '', text)
    
    # [修改3] 大幅扩充的停用词库 (基于对食材数据的深入分析)
    stop_words = set([
        # --- 原始列表 ---
        'fresh', 'ground', 'chopped', 'sliced', 'diced', 'crushed', 'minced', 'grated', 
        'large', 'medium', 'small', 'cloves', 'lb', 'oz', 'drained', 'pitted', 'beaten', 
        'unsalted', 'all-purpose', 'chunks', 'dried', 'leaves', 'powder', 'frozen', 'warm',
        'melted', 'boneless', 'skinless', 'halves', 'raw', 'extra', 'virgin',
        
        # --- 新增：加工状态 ---
        'canned', 'jarred', 'stewed', 'condensed', 'evaporated', 'thawed', 'smoked',
        'cured', 'pickled', 'harden', 'softened', 'puree', 'paste',
        
        # --- 新增：形状/切割 ---
        'cubed', 'wedges', 'strips', 'rings', 'lengthwise', 'pieces', 'segments', 
        'florets', 'spears', 'hearts', 'whole', 'fillet', 'filet', 'loins',
        
        # --- 新增：健康/成分标签 ---
        'low', 'fat', 'nonfat', 'free', 'reduced', 'sodium', 'gluten', 'skim', 'part',
        'light', 'lite', 'organic',
        
        # --- 新增：温度/物理 ---
        'room', 'temperature', 'lukewarm', 'cold', 'hot', 'boiling',
        
        # --- 新增：通用量词/容器 ---
        'cup', 'cups', 'teaspoon', 'tablespoon', 'tbsp', 'tsp', 'pinch', 'dash',
        'quart', 'pint', 'gallon', 'bottle', 'can', 'stick', 'pack', 'package'
    ])
    
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    
    # 重新组合
    cleaned = ' '.join(words)
    
    # [兜底策略] 如果清洗后把词都洗没了 (比如 "fresh large"), 就返回原词，防止空字符串
    if not cleaned.strip():
        return text
        
    return cleaned

def load_data():
    print("Loading data...")
    with open('train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    print("Cleaning ingredients with ENHANCED rules...")
    train_df['clean_list'] = train_df['ingredients'].apply(lambda x: [clean_text(i) for i in x])
    test_df['clean_list'] = test_df['ingredients'].apply(lambda x: [clean_text(i) for i in x])
    
    train_df['text_str'] = train_df['clean_list'].apply(lambda x: ', '.join(x)) 
    test_df['text_str'] = test_df['clean_list'].apply(lambda x: ', '.join(x))
    
    return train_df, test_df

# ==========================================
# 2. 模型 A: LinearSVC
# ==========================================
def train_svc(X_train, y_train, X_val, y_val, X_test):
    print("\n[LinearSVC] Training...")
    # TF-IDF 设置
    tfidf = TfidfVectorizer(binary=True, ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    
    # 拟合训练集
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)
    X_test_vec = tfidf.transform(X_test)
    
    svc = LinearSVC(C=0.6, penalty='l2', dual=False, max_iter=3000, random_state=CONFIG['seed'])
    clf = CalibratedClassifierCV(svc, method='sigmoid', cv=5)
    
    clf.fit(X_train_vec, y_train)
    
    # 打印验证集分数
    val_score = clf.score(X_val_vec, y_val)
    print(f"[LinearSVC] Validation Accuracy: {val_score:.4f}")
    
    probs = clf.predict_proba(X_test_vec)
    return probs, clf, tfidf

# ==========================================
# 3. 模型 B: DeBERTa-v3-Large (含验证循环)
# ==========================================
class CuisineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augment=False):
        self.texts = texts 
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        ingredients = self.texts[idx]
        if self.augment:
            ingredients = list(ingredients)
            np.random.shuffle(ingredients)
            
        text = ", ".join(ingredients)
        
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

def train_transformer(X_train, y_train, X_val, y_val, X_test, num_classes, pseudo_texts=None, pseudo_labels=None):
    print(f"\n[Transformer] Training {CONFIG['model_name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # 合并伪标签数据（如果有）到训练集
    all_train_texts = list(X_train)
    all_train_labels = list(y_train)
    
    if pseudo_texts is not None:
        print(f"🔥 Adding {len(pseudo_texts)} Pseudo-Labeled samples to training!")
        all_train_texts.extend(pseudo_texts)
        all_train_labels.extend(pseudo_labels)
        # 注意：这里我们不把伪标签加到验证集，验证集必须保持纯净
    
    # 构建 Dataset
    train_dataset = CuisineDataset(all_train_texts, all_train_labels, tokenizer, CONFIG['max_len'], augment=True)
    val_dataset = CuisineDataset(X_val, y_val, tokenizer, CONFIG['max_len'], augment=False)
    test_dataset = CuisineDataset(X_test, None, tokenizer, CONFIG['max_len'], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size']*2, shuffle=False, num_workers=0) # 验证集BS可以大一点
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size']*2, shuffle=False, num_workers=0)
    
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG['model_name'], num_labels=num_classes)
    model.to(CONFIG['device'])
    
    # 显存优化配置
    if CONFIG['use_checkpointing']:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    total_steps = len(train_loader) * CONFIG['epochs'] // CONFIG['accum_steps']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    
    scaler = torch.amp.GradScaler('cuda') 
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Training Loop ---
    best_val_acc = 0.0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['labels'].to(CONFIG['device'])
            
            with torch.amp.autocast('cuda'): 
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss = loss / CONFIG['accum_steps'] 
            
            scaler.scale(loss).backward()
            
            if (step + 1) % CONFIG['accum_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * CONFIG['accum_steps']
            pbar.set_postfix({'loss': total_train_loss / (step + 1)})
        
        # --- Validation Loop (每个 Epoch 结束后运行) ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        print(f"Validating Epoch {epoch+1}...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(CONFIG['device'])
                attention_mask = batch['attention_mask'].to(CONFIG['device'])
                labels = batch['labels'].to(CONFIG['device'])
                
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        print(f"👉 [Epoch {epoch+1}] Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 在这里可以保存模型，或者直接继续跑
            
    print("Generating predictions...")
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
            
            probs = torch.softmax(outputs.logits, dim=1).float()
            all_probs.append(probs.cpu().numpy())
            
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()
    
    return np.concatenate(all_probs, axis=0)

# ==========================================
# 4. 主流程 (含伪标签)
# ==========================================
def main():
    train_df, test_df = load_data()
    
    le = LabelEncoder()
    # 全量标签编码
    y_full = le.fit_transform(train_df['cuisine'])
    num_classes = len(le.classes_)
    
    # [新增] 划分训练集和验证集 (90% Train, 10% Val)
    # 使用 stratify 保证每一类菜系的比例在验证集中一致
    print("Splitting data into Train (90%) and Validation (10%)...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        train_df['clean_list'].values, 
        y_full, 
        test_size=0.1, 
        random_state=CONFIG['seed'], 
        stratify=y_full
    )
    
    # 为了 SVC 需要 string 格式
    X_train_str = [", ".join(x) for x in X_train_raw]
    X_val_str = [", ".join(x) for x in X_val_raw]
    X_test_str = test_df['text_str'].values
    
    # --- Round 1: 初始训练 ---
    print("\n" + "="*30 + "\n ROUND 1: Initial Training \n" + "="*30)
    
    # 训练 SVC (带验证)
    svc_probs, svc_model, tfidf_model = train_svc(X_train_str, y_train, X_val_str, y_val, X_test_str)
    
    # 训练 DeBERTa (带验证)
    deb_probs = train_transformer(X_train_raw, y_train, X_val_raw, y_val, test_df['clean_list'].values, num_classes)
    
    ensemble_probs_r1 = (svc_probs * 0.4) + (deb_probs * 0.6) 
    
    # --- Round 2: 伪标签 (Pseudo Labeling) ---
    print("\n" + "="*30 + "\n ROUND 2: Pseudo Labeling \n" + "="*30)
    
    max_probs = np.max(ensemble_probs_r1, axis=1)
    pseudo_indices = np.where(max_probs >= CONFIG['pseudo_threshold'])[0]
    pseudo_labels = np.argmax(ensemble_probs_r1[pseudo_indices], axis=1)
    
    print(f"Found {len(pseudo_indices)} samples with confidence >= {CONFIG['pseudo_threshold']}")
    
    if len(pseudo_indices) > 0:
        pseudo_texts_list = test_df['clean_list'].iloc[pseudo_indices].values
        pseudo_texts_str = test_df['text_str'].iloc[pseudo_indices].values
        
        # 重新训练 SVC (Train + Pseudo)
        # 注意：这里我们依然在原始 Val 上验证，看看加入伪标签后模型是否变强
        X_train_full_str = X_train_str + list(pseudo_texts_str)
        y_train_full = np.concatenate([y_train, pseudo_labels])
        
        svc_probs_r2, _, _ = train_svc(X_train_full_str, y_train_full, X_val_str, y_val, X_test_str)
        
        # 重新训练 DeBERTa
        deb_probs_r2 = train_transformer(
            X_train_raw, y_train, 
            X_val_raw, y_val, # 验证集不变
            test_df['clean_list'].values, num_classes,
            pseudo_texts=pseudo_texts_list, pseudo_labels=pseudo_labels
        )
        
        final_probs = (svc_probs_r2 * 0.4) + (deb_probs_r2 * 0.6)
    else:
        print("Not enough confident samples for pseudo labeling. Using Round 1 results.")
        final_probs = ensemble_probs_r1

    final_preds = np.argmax(final_probs, axis=1)
    final_labels = le.inverse_transform(final_preds)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'cuisine': final_labels
    })
    submission.to_csv('submission_sota.csv', index=False)
    print("Done! Check submission_sota.csv")

if __name__ == '__main__':
    main()
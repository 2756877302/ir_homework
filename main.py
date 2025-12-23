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
    
    # --- [关键修改] 显存/稳定性优化 ---
    # 之前的 Gradient Checkpointing 导致了报错。
    # 我们关闭它，并将 Batch Size 调小，同时增加累积步数。
    # 显存占用会很低，且绝对不会报错。
    # 实际效能：Batch 2 * Accum 16 = 32 (与原方案 8*4=32 一致)
    'batch_size': 2,         
    'accum_steps': 16,        
    'use_checkpointing': False, # <--- 彻底关闭以修复报错
    # --------------------------------
    
    'epochs': 10,
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
print(f"🚀 Running STABLE solution on: {CONFIG['device']} with {CONFIG['model_name']}")
print(f"⚙️ Config: Batch={CONFIG['batch_size']} | Accum={CONFIG['accum_steps']} | Checkpointing={CONFIG['use_checkpointing']}")

# ==========================================
# 1. 数据清洗
# ==========================================
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s-]', '', text)
    stop_words = set([
        'fresh', 'ground', 'chopped', 'sliced', 'diced', 'crushed', 'minced', 'grated', 
        'large', 'medium', 'small', 'cloves', 'lb', 'oz', 'drained', 'pitted', 'beaten', 
        'unsalted', 'all-purpose', 'chunks', 'dried', 'leaves', 'powder', 'frozen', 'warm',
        'melted', 'boneless', 'skinless', 'halves', 'raw', 'extra', 'virgin'
    ])
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

def load_data():
    with open('train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    print("Cleaning ingredients...")
    train_df['clean_list'] = train_df['ingredients'].apply(lambda x: [clean_text(i) for i in x])
    test_df['clean_list'] = test_df['ingredients'].apply(lambda x: [clean_text(i) for i in x])
    
    train_df['text_str'] = train_df['clean_list'].apply(lambda x: ', '.join(x)) 
    test_df['text_str'] = test_df['clean_list'].apply(lambda x: ', '.join(x))
    
    return train_df, test_df

# ==========================================
# 2. 模型 A: LinearSVC
# ==========================================
def train_svc(X_train, y_train, X_test):
    print("\n[LinearSVC] Training...")
    tfidf = TfidfVectorizer(binary=True, ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    svc = LinearSVC(C=0.6, penalty='l2', dual=False, max_iter=3000, random_state=CONFIG['seed'])
    clf = CalibratedClassifierCV(svc, method='sigmoid', cv=5)
    clf.fit(X_train_vec, y_train)
    
    probs = clf.predict_proba(X_test_vec)
    return probs, clf, tfidf

# ==========================================
# 3. 模型 B: DeBERTa-v3-Large
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

def train_transformer(train_texts, train_labels, test_texts, num_classes, pseudo_texts=None, pseudo_labels=None):
    print(f"\n[Transformer] Training {CONFIG['model_name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    all_train_texts = list(train_texts)
    all_train_labels = list(train_labels)
    
    if pseudo_texts is not None:
        print(f"🔥 Adding {len(pseudo_texts)} Pseudo-Labeled samples to training!")
        all_train_texts.extend(pseudo_texts)
        all_train_labels.extend(pseudo_labels)
    
    train_dataset = CuisineDataset(all_train_texts, all_train_labels, tokenizer, CONFIG['max_len'], augment=True)
    test_dataset = CuisineDataset(test_texts, None, tokenizer, CONFIG['max_len'], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG['model_name'], num_labels=num_classes)
    model.to(CONFIG['device'])
    
    # ---------------------------------------------------------
    # [稳定修复] 
    # 如果开启 Checkpointing 报错，直接关闭。
    # 只要 Batch Size 足够小 (设为2)，显存通常足够。
    # ---------------------------------------------------------
    if CONFIG['use_checkpointing']:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("Gradient Checkpointing enabled.")
    else:
        print("Gradient Checkpointing DISABLED for stability.")

    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    total_steps = len(train_loader) * CONFIG['epochs'] // CONFIG['accum_steps']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    
    # 使用新版 API
    scaler = torch.amp.GradScaler('cuda') 
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    optimizer.zero_grad()
    
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['labels'].to(CONFIG['device'])
            
            # 使用新版 API
            with torch.amp.autocast('cuda'): 
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss = loss / CONFIG['accum_steps'] 
            
            # 这里不会再报错了，因为没有 checkpointing 导致的图冲突
            scaler.scale(loss).backward()
            
            if (step + 1) % CONFIG['accum_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * CONFIG['accum_steps']
            pbar.set_postfix({'loss': total_loss / (step + 1)})
            
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
    y_train_enc = le.fit_transform(train_df['cuisine'])
    num_classes = len(le.classes_)
    
    # --- Round 1: 初始训练 ---
    print("\n" + "="*30 + "\n ROUND 1: Initial Training \n" + "="*30)
    
    svc_probs, svc_model, tfidf_model = train_svc(train_df['text_str'], y_train_enc, test_df['text_str'])
    
    deb_probs = train_transformer(train_df['clean_list'].values, y_train_enc, test_df['clean_list'].values, num_classes)
    
    ensemble_probs_r1 = (svc_probs * 0.4) + (deb_probs * 0.6) 
    
    # --- Round 2: 伪标签 (Pseudo Labeling) ---
    print("\n" + "="*30 + "\n ROUND 2: Pseudo Labeling \n" + "="*30)
    
    max_probs = np.max(ensemble_probs_r1, axis=1)
    pseudo_indices = np.where(max_probs >= CONFIG['pseudo_threshold'])[0]
    pseudo_labels = np.argmax(ensemble_probs_r1[pseudo_indices], axis=1)
    
    print(f"Found {len(pseudo_indices)} samples with confidence >= {CONFIG['pseudo_threshold']}")
    
    if len(pseudo_indices) > 0:
        pseudo_texts_list = test_df['clean_list'].iloc[pseudo_indices].values
        
        X_train_full = pd.concat([train_df['text_str'], test_df['text_str'].iloc[pseudo_indices]])
        y_train_full = np.concatenate([y_train_enc, pseudo_labels])
        
        svc_probs_r2, _, _ = train_svc(X_train_full, y_train_full, test_df['text_str'])
        
        deb_probs_r2 = train_transformer(
            train_df['clean_list'].values, y_train_enc, 
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
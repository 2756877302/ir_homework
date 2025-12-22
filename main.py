import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import torch

# ==========================================
# 1. 配置
# ==========================================
# 你的 5070Ti 显存充足，presets='best_quality' 会做多层 stacking，效果最好但训练较慢
# 如果想快点看到结果，可以改用 presets='high_quality' 或 'medium_quality'
CONFIG = {
    'presets': 'best_quality', 
    'time_limit': 600,  # 训练时间限制 (秒)，设为 None 则不限制，跑完为止
    'eval_metric': 'accuracy'
}

# 检查 GPU
num_gpus = 1 if torch.cuda.is_available() else 0
print(f"GPUs available: {num_gpus}")

# ==========================================
# 2. 数据处理
# ==========================================
def load_and_process(file_path, is_train=True):
    # 读取 JSON
    df = pd.read_json(file_path)
    
    # 核心步骤：将 list ['salt', 'pepper'] 转换为 string "salt pepper"
    # 这样 AutoGluon 就能把它当作 NLP 文本特征自动处理
    df['ingredients_text'] = df['ingredients'].apply(lambda x: ' '.join(x))
    
    # 只要文本特征和标签，去掉原始 list 列
    if is_train:
        return df[['ingredients_text', 'cuisine']]
    else:
        # 测试集保留 id 用于提交
        return df[['id', 'ingredients_text']]

print("Loading data...")
train_data = load_and_process('train.json', is_train=True)
test_data = load_and_process('test.json', is_train=False)

print(f"Train shape: {train_data.shape}")
print(train_data.head())

# ==========================================
# 3. AutoGluon 训练
# ==========================================
print("Starting AutoGluon training...")

# 转换为 TabularDataset 对象
train_data = TabularDataset(train_data)

predictor = TabularPredictor(
    label='cuisine', 
    eval_metric=CONFIG['eval_metric'],
    path='ag_models_cuisine'  # 模型保存路径
).fit(
    train_data,
    presets=CONFIG['presets'],
    time_limit=CONFIG['time_limit'],
    ag_args_fit={'num_gpus': num_gpus}  # 显式开启 GPU
)

# ==========================================
# 4. 评估与预测
# ==========================================
# 查看训练集上的各个模型表现
leaderboard = predictor.leaderboard(train_data, silent=True)
print("\nModel Leaderboard (on training data subset):")
print(leaderboard[['model', 'score_val']].head())

print("\nPredicting on Test set...")
# 预测
test_features = TabularDataset(test_data[['ingredients_text']])
predictions = predictor.predict(test_features)

# ==========================================
# 5. 生成提交文件
# ==========================================
submission = pd.DataFrame({
    'id': test_data['id'],
    'cuisine': predictions
})

submission.to_csv('submission_autogluon.csv', index=False)
print("Done! Saved to submission_autogluon.csv")
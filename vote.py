import pandas as pd
import numpy as np
import os
import glob


def ensemble_submissions(input_dir, power=10):
    """
    input_dir: 存放 csv 的文件夹路径
    power: 幂次系数。该值越大，高分模型的话语权越呈几何倍数增长。
           如果模型之间差距极小，建议设置在 10-20 之间。
    """
    # 1. 获取所有 csv 文件
    file_paths = glob.glob(os.path.join(input_dir, "*.csv"))

    if not file_paths:
        print("未找到 CSV 文件，请检查路径。")
        return

    all_preds = []
    weights = []

    print(f"正在读取文件并计算权重 (Power={power})...")

    for path in file_paths:
        # 获取文件名（不含扩展名）作为得分
        filename = os.path.basename(path).replace(".csv", "")
        try:
            score = float(filename) / 100000.0  # 还原为准确率
        except ValueError:
            continue

        # 读取数据
        df = pd.read_csv(path)
        # 确保按 ID 排序，保证行对应一致
        df = df.sort_values("id").reset_index(drop=True)

        all_preds.append(df["cuisine"].values)

        # 非线性权重计算：Weight = Score^p
        # 这种方式能让 0.81521 和 0.81357 的微小差距在权重上产生显著差异
        weights.append(np.power(score, power))
        print(f"文件: {filename}.csv | 原始分: {score:.5f} | 权重: {weights[-1]:.4f}")

    # 2. 进行加权投票
    # 转换为矩阵 [模型数, 样本数]
    preds_matrix = np.array(all_preds)
    weights = np.array(weights)

    # 获取类别列表（假设标签是整数或可分类的）
    unique_labels = np.unique(preds_matrix)

    final_predictions = []

    print("正在进行加权修正...")
    # 遍历每个样本（每一行）
    for i in range(preds_matrix.shape[1]):
        sample_preds = preds_matrix[:, i]

        # 计算每个类别的加权得分
        label_scores = {}
        for label in unique_labels:
            # 找到预测为该 label 的模型索引，并累加它们的权重
            label_weights = weights[sample_preds == label]
            label_scores[label] = np.sum(label_weights)

        # 选择加权得分最高的类别
        best_label = max(label_scores, key=label_scores.get)
        final_predictions.append(best_label)

    # 3. 保存结果
    submission = pd.read_csv(file_paths[0]).sort_values("id").reset_index(drop=True)
    submission["cuisine"] = final_predictions

    output_name = "vote.csv"
    submission.to_csv(output_name, index=False)
    print(f"\n融合完成！结果已保存至: {output_name}")


if __name__ == "__main__":
    # 假设你的文件夹叫 submission_history
    ensemble_submissions("submission_history", power=15)

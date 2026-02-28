"""
基于线性回归（Linear Regression）的 Redmond 市房价预测

数据集来源：Kaggle House Data
https://www.kaggle.com/datasets/shree1992/housedata/data

任务：回归 - 预测房屋价格
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath='data/data.csv'):
    """加载房价数据集"""
    try:
        df = pd.read_csv(filepath)
        print(f"数据加载成功，共 {len(df)} 条记录。")
        print("原始数据前5行：")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"错误：请确保 '{filepath}' 文件存在。")
        print("你可以从以下地址下载: "
              "https://www.kaggle.com/datasets/shree1992/housedata/data")
        exit()

def preprocess_data(df, city='Redmond'):
    """数据预处理：筛选城市 + 特征选择 + 对数变换"""
    # 筛选指定城市
    df_city = df[df['city'] == city].copy()
    print(f"\n筛选 '{city}' 后的数据量: {len(df_city)}")

    # 选择特征列
    feature_cols = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
        'floors', 'waterfront', 'view', 'condition', 'yr_built'
    ]

    X = df_city[feature_cols]
    y = df_city['price']

    # 对目标变量进行对数变换（减少偏态分布影响）
    y_log = np.log1p(y)

    print(f"\n特征数量: {len(feature_cols)}")
    print(f"目标变量（price）统计：")
    print(f"  原始 - 均值: {y.mean():.2f}, 标准差: {y.std():.2f}")
    print(f"  对数变换后 - 均值: {y_log.mean():.4f}, 标准差: {y_log.std():.4f}")

    return X, y, y_log, feature_cols

def train_and_evaluate(X, y_log, feature_cols, test_size=0.2, random_state=42):
    """模型训练与评估"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=test_size, random_state=random_state
    )
    print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 模型构建与训练
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\n模型训练完成！")

    # 模型预测
    y_pred = model.predict(X_test)

    # 评估指标（在对数空间下）
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("           模型评估结果（对数空间）")
    print("=" * 50)
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²):  {r2:.4f}")

    # 特征重要性（按系数绝对值排序）
    print("\n各特征权重（按绝对值排序）：")
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_
    })
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

    for _, row in coef_df.iterrows():
        direction = "↑" if row['Coefficient'] > 0 else "↓"
        print(f"  {row['Feature']:15s}: {row['Coefficient']:+.6f} {direction}")

    # 还原预测值到原始价格空间
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)

    mse_original = mean_squared_error(y_test_original, y_pred_original)
    r2_original = r2_score(y_test_original, y_pred_original)

    print(f"\n{'=' * 50}")
    print("           模型评估结果（原始价格空间）")
    print("=" * 50)
    print(f"均方误差 (MSE): {mse_original:,.2f}")
    print(f"决定系数 (R²):  {r2_original:.4f}")

    return model, y_test, y_pred

def plot_results(y_test, y_pred, output_dir='results'):
    """绘制预测值 vs 真实值散点图"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)

    plt.figure(figsize=(10, 7))
    plt.scatter(y_test_original, y_pred_original, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.plot(
        [y_test_original.min(), y_test_original.max()],
        [y_test_original.min(), y_test_original.max()],
        'r--', linewidth=2, label='Perfect Prediction'
    )
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title('House Price Prediction: Actual vs Predicted', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'prediction_scatter.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n预测散点图已保存至：{save_path}")

if __name__ == '__main__':
    # 1. 加载数据
    df = load_data()

    # 2. 数据预处理
    X, y, y_log, feature_cols = preprocess_data(df, city='Redmond')

    # 3. 训练与评估
    model, y_test, y_pred = train_and_evaluate(X, y_log, feature_cols)

    # 4. 可视化
    plot_results(y_test, y_pred)

    print("\n✅ 房价预测实验完成！")

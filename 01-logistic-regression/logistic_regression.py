"""
基于对数几率回归（Logistic Regression）的药物分类预测

数据集来源：Kaggle Drug Classification
https://www.kaggle.com/datasets/prathamtripathi/drug-classification/data

任务：二分类 - 预测患者应使用 DrugX 还是 DrugY
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(filepath='data/drug200.csv'):
    """加载药物分类数据集"""
    try:
        df = pd.read_csv(filepath)
        print("数据加载成功！")
        print("原始数据前5行：")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"错误：请确保 '{filepath}' 文件存在。")
        print("你可以从以下地址下载: "
              "https://www.kaggle.com/datasets/prathamtripathi/drug-classification/data")
        exit()


def preprocess_data(df):
    """数据预处理：筛选目标类别 + 特征编码"""
    # 筛选 DrugY 和 DrugX（二分类任务）
    df_filtered = df[df['Drug'].isin(['DrugY', 'DrugX'])].copy()
    print(f"\n筛选 'DrugY' 和 'DrugX' 后的数据量: {df_filtered.shape}")

    # 类别特征数值化
    df_filtered['Sex'] = df_filtered['Sex'].map({'F': 0, 'M': 1})
    df_filtered['BP'] = df_filtered['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
    df_filtered['Cholesterol'] = df_filtered['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})

    # 目标变量数值化
    df_filtered['Drug'] = df_filtered['Drug'].map({'DrugX': 0, 'DrugY': 1})

    print("\n数据预处理完成后的数据前5行：")
    print(df_filtered.head())
    return df_filtered


def train_and_evaluate(df_filtered, test_size=0.25, random_state=42):
    """模型训练与评估"""
    # 定义特征和标签
    feature_cols = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
    X = df_filtered[feature_cols]
    y = df_filtered['Drug']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 模型构建与训练
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    print("\n模型训练完成！")

    # 打印模型参数
    print(f"模型截距 (Intercept): {model.intercept_}")
    print(f"模型系数 (Coefficients): {model.coef_}")
    print("\n各特征系���对应关系：")
    for feat, coef in zip(feature_cols, model.coef_[0]):
        print(f"  {feat}: {coef:.4f}")

    # 模型预测与评估
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\n" + "=" * 50)
    print("           模型评估结果")
    print("=" * 50)
    print(f"准确率 (Accuracy): {accuracy:.2f}")
    print(f"\n混淆矩阵 (Confusion Matrix):\n{conf_matrix}")
    print(f"\n分类报告 (Classification Report):\n{class_report}")

    return model, accuracy


if __name__ == '__main__':
    # 1. 加载数据
    df = load_data()

    # 2. 数据预处理
    df_processed = preprocess_data(df)

    # 3. 训练与评估
    model, accuracy = train_and_evaluate(df_processed)
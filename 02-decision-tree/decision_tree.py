基于决策树的药物分类及多模型对比分析

数据集来源：Kaggle Drug Classification
https://www.kaggle.com/datasets/prathamtripathi/drug-classification/data

任务：
  - 实验一：DrugX vs DrugC 二分类（20% 测试集）
  - 实验二：DrugX vs DrugC 二分类（30% 测试集）
  - 实验三：DrugY vs DrugA 二分类（20% 测试集）
  - 每个实验对比决策树与逻辑回归的性能

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def run_experiment(df, class1, class2, test_size=0.2, random_state=42,
                   save_tree=False, output_dir='results'):
    """
    运行一个完整的分类实验流程：数据准备、模型训练、评估与对比。

    Parameters
    ----------
    df : pd.DataFrame - 原始数据集
    class1 : str - 分类目标类别1（编码为0）
    class2 : str - 分类目标类别2（编码为1）
    test_size : float - 测试集比例
    random_state : int - 随机种子
    save_tree : bool - 是否保存决策树可视化图
    output_dir : str - 结果保存目录

    Returns
    -------
    dict : 两个模型的性能指标
    """
    print(f"\n{'=' * 60}")
    print(f"  实验：{class1} vs {class2}（测试集比例={test_size}）")
    print(f"{'=' * 60}")

    # 1. 数据筛选与准备
    df_binary = df[df['Drug'].isin([class1, class2])].copy()
    target_map = {class1: 0, class2: 1}
    df_binary['Drug'] = df_binary['Drug'].map(target_map)
    print(f"\n筛选后样本数量：{len(df_binary)}")

    X = df_binary.drop('Drug', axis=1)
    y = df_binary['Drug']

    # 2. 特征工程：独热编码 + 数值归一化
    X = pd.get_dummies(X, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)

    scaler = MinMaxScaler()
    numerical_cols = ['Age', 'Na_to_K']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # 3. 划分数据集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"训练集：{len(X_train)} 样本，测试集：{len(X_test)} 样本")

    # --- 决策树模型 ---
    print("\n【决策树模型】")
    dt_clf = DecisionTreeClassifier(
        criterion='gini', max_depth=5, random_state=random_state
    )
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)

    acc_dt = accuracy_score(y_test, y_pred_dt)
    report_dt = classification_report(
        y_test, y_pred_dt, target_names=[class1, class2], output_dict=True
    )
    cm_dt = confusion_matrix(y_test, y_pred_dt)

    print(f"准确率 (Accuracy): {acc_dt:.2f}")
    print("分类报告:")
    print(classification_report(y_test, y_pred_dt, target_names=[class1, class2]))
    print(f"混淆矩阵:\n{cm_dt}")

    # 决策树可视化
    if save_tree:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(20, 10))
        plot_tree(
            dt_clf, filled=True, feature_names=X.columns,
            class_names=[class1, class2], rounded=True
        )
        plt.title(f"Decision Tree Visualization ({class1} vs {class2})")
        save_path = os.path.join(output_dir, "decision_tree.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"决策树可视化已保存至：{save_path}")

    # --- 逻辑回归模型 ---
    print("\n【逻辑回归模型】")
    lr_clf = LogisticRegression(random_state=random_state)
    lr_clf.fit(X_train, y_train)
    y_pred_lr = lr_clf.predict(X_test)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(
        y_test, y_pred_lr, target_names=[class1, class2],
        output_dict=True, zero_division=0
    )
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    print(f"准确率 (Accuracy): {acc_lr:.2f}")
    print("分类报告:")
    print(classification_report(
        y_test, y_pred_lr, target_names=[class1, class2], zero_division=0
    ))
    print(f"混淆矩阵:\n{cm_lr}")

    # 返回性能指标
    results = {
        'Decision Tree': {
            'accuracy': acc_dt,
            f'{class1}_f1': report_dt[class1]['f1-score'],
            f'{class2}_f1': report_dt[class2]['f1-score'],
        },
        'Logistic Regression': {
            'accuracy': acc_lr,
            f'{class1}_f1': report_lr[class1]['f1-score'],
            f'{class2}_f1': report_lr[class2]['f1-score'],
        }
    }
    return results


def plot_comparison(results, title, output_dir='results'):
    """绘制模型性能对比的条形图并保存。"""
    os.makedirs(output_dir, exist_ok=True)

    df_results = pd.DataFrame(results).T
    ax = df_results.plot(kind='bar', figsize=(12, 7), rot=0)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Metrics')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存至：{save_path}")


if __name__ == '__main__':
    # 加载数据
    DATA_PATH = 'data/drug200.csv'
    try:
        df_main = pd.read_csv(DATA_PATH)
        print(f"数据加载成功，共 {len(df_main)} 条记录。")
    except FileNotFoundError:
        print(f"错误：请确保 '{DATA_PATH}' 文件存在。")
        print("下载地址: https://www.kaggle.com/datasets/prathamtripathi/drug-classification/data")
        exit()

    # 实验一：DrugX vs DrugC, 20% 测试集（保存决策树可视化）
    results_1 = run_experiment(
        df_main, 'drugX', 'drugC', test_size=0.2, save_tree=True
    )

    # 实验二：DrugX vs DrugC, 30% 测试集（探究测试集比例影响）
    results_2 = run_experiment(
        df_main, 'drugX', 'drugC', test_size=0.3
    )

    # 实验三：DrugY vs DrugA, 20% 测试集（探究不同分类任务的影响）
    results_3 = run_experiment(
        df_main, 'drugY', 'drugA', test_size=0.2
    )

    # 绘制实验三的对比图（任务较复杂，对比更有意义）
    plot_comparison(
        results_3,
        "Model Performance Comparison (DrugY vs DrugA)"
    )

    print("\n✅ 所有实验完成！")

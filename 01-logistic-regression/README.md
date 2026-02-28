# 💊 基于对数几率回归的药物分类预测

## 📖 项目简介
使用 Logistic Regression（对数几率回归）模型对 Kaggle Drug Classification 数据集进行二分类预测，
判断患者应使用 DrugX 还是 DrugY。

## 🗂️ 数据集
- **来源**：[Kaggle Drug Classification](https://www.kaggle.com/datasets/prathamtripathi/drug-classification/data)
- **样本量**：200 条（筛选后保留 DrugX 和 DrugY）
- **特征**：Age, Sex, BP, Cholesterol, Na_to_K
- **任务**：二分类（DrugX vs DrugY）

## 🛠️ 技术栈
- Python 3.x
- Scikit-learn（Logistic Regression）
- Pandas / NumPy

## 🔧 数据预处理
| 特征 | 原始值 | 编码后 |
|------|--------|--------|
| Sex | F / M | 0 / 1 |
| BP | LOW / NORMAL / HIGH | 0 / 1 / 2 |
| Cholesterol | NORMAL / HIGH | 0 / 1 |

## 📊 实验结果
- **准确率**：100%（在 DrugX vs DrugY 二分类任务上）
- 模型输出各特征的系数，可解释每个特征对分类结果的影响程度

## 🚀 如何运行
```bash
pip install scikit-learn pandas
python logistic_regression.py
```
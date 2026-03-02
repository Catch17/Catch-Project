# 🔐 Security ML Mini: Phishing Website Detection（二分类）

## 项目简介
使用 Python + scikit-learn 构建钓鱼/恶意网站检测模型。数据集标签 `Result` 可能取值为 `-1/0/1`，本项目将其转换为二分类：
- 0：Benign（Result == -1）
- 1：Malicious/Suspicious（Result != -1）

项目输出 `metrics.json`、混淆矩阵与 ROC 曲线，便于复现实验与对比模型。

## 数据集
- 名称：Website Phishing Data Set
- 来源：https://www.kaggle.com/datasets/ahmednour/website-phishing-data-set?resource=download
- 将 CSV 放入：`data/phishing.csv`

## 方法
- 数据划分：train/test = 8/2，分层抽样（stratify），random_state=42
- 模型（可选）：
  - Logistic Regression（class_weight=balanced）
  - SVM-RBF（class_weight=balanced）
- 指标：Accuracy、Precision、Recall、F1、ROC-AUC；并输出 Confusion Matrix、ROC Curve

## 如何运行
```bash
pip install -r requirements.txt

# LR
python src/train.py --data data/phishing.csv --label-col Result --model lr --outdir outputs

# SVM
python src/train.py --data data/phishing.csv --label-col Result --model svm --outdir outputs
```

## 输出
- `outputs/metrics.json`
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`

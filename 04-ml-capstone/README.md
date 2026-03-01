# 🎓 机器学习最终课设：多数据集分类模型对比与调参（.mat 可配置）

## 📌 项目简介（中文）
本项目整理并工程化了课程最终课设的核心代码，围绕 **多个真实数据集** 的分类任务，完成了：
- **对数几率回归（Logistic Regression）**：基于 Hessian 的 Newton / 拟牛顿式更新（`pinv` 求逆）  
- **决策树（Decision Tree）**：基于 5-fold 交叉验证的网格搜索调参  
- **支持向量机（SVM, RBF）**：基于 StratifiedKFold 的参数搜索（`C`、`gamma`），并修复“训练集上评估”的数据泄漏问题（最终在 held-out test set 评估）
- 输出统一的评估指标与可视化结果（混淆矩阵热力图、决策树结构图等）

### 为什么数据集缺失也能运行？
考虑到 `.mat` 数据文件可能无法随仓库一并提供，本项目采用“**方案A：数据加载可配置**”：
- 运行时通过参数指定 `.mat` 路径、变量 key、标签列位置
- 提供 `inspect_mat.py` 帮你快速查看 `.mat` 文件包含哪些 key、shape，从而确定参数怎么填

---

## 🗂️ 项目结构
- `src/inspect_mat.py`：查看 `.mat` 文件 keys / shape，建议默认 `--mat-key` 与 `--label-col`
- `src/data.py`：通用 `.mat` 读取与特征/标签切分（label 支持 `first|last|index`）
- `src/metrics.py`：统一指标（accuracy、precision/recall/F1 的 macro/micro/weighted；二分类额外提供“第一类指标”口径）
- `src/plots.py`：保存混淆矩阵热力图；保存决策树可视化（使用非交互后端，适合无 GUI 环境）
- `src/logreg_newton_binary.py`：二分类 Logistic Regression（Newton/拟牛顿更新）
- `src/logreg_ovo_multiclass.py`：多分类 Logistic Regression（OvO + 众数投票）
- `src/decision_tree_tuning.py`：决策树 CV 调参 + 测试集评估
- `src/svm_tuning.py`：SVM CV 调参 + **测试集评估**（避免数据泄漏）
- `src/run.py`：统一 CLI 入口（选择模型/是否调参/输出 reports）
- `tests/test_smoke.py`：合成数据 smoke tests（不依赖 `.mat`）

---

## 🔍 第一步：检查你的 .mat 文件（推荐）
```bash
python 04-ml-capstone/src/inspect_mat.py --mat-path your_dataset.mat
```

---

## 🚀 运行示例
> 说明：以下命令中的 `--mat-key`、`--label-col` 需要根据你的 `.mat` 内容调整。  
> 如果不确定，请先运行 `inspect_mat.py`。

### 1) 运行 SVM（可选调参）并输出报告
```bash
python 04-ml-capstone/src/run.py \
  --mat-path your_dataset.mat \
  --mat-key data \
  --label-col last \
  --model svm \
  --tune \
  --test-size 0.2 \
  --random-state 42
```

### 2) 运行决策树并调参
```bash
python 04-ml-capstone/src/run.py \
  --mat-path your_dataset.mat \
  --mat-key data \
  --label-col last \
  --model decision_tree \
  --tune \
  --test-size 0.2 \
  --random-state 42
```

### 3) 运行二分类 Logistic Regression（Newton）
```bash
python 04-ml-capstone/src/run.py \
  --mat-path your_dataset.mat \
  --mat-key data \
  --label-col first \
  --model logreg_newton_binary \
  --test-size 0.3 \
  --random-state 42
```

---

## 📈 输出结果（reports）
每次运行都会在 `reports/<run-id>/` 生成：
- `metrics.json`：指标与最佳参数（如有调参）
- `confusion_matrix.png`：混淆矩阵热力图
- `tree.png`：决策树结构图（仅 decision tree）
并在 `reports/summary.csv` 追加一行汇总。

> 注：`reports/` 已加入 `.gitignore`，默认不会提交运行产物（避免仓库膨胀）。

---

## 🧰 环境依赖
建议 Python 3.x，主要依赖：
- numpy, scipy
- scikit-learn
- matplotlib, seaborn

可直接使用仓库根目录的 `requirements.txt` 安装（如其中已包含上述依赖）。

---

# English Summary
This folder is a cleaned-up, reproducible version of my ML course capstone project.  
It provides a unified CLI to run and tune multiple classifiers (LogReg with Newton-style updates, Decision Tree with CV grid search, SVM with StratifiedKFold tuning) on `.mat` datasets **without bundling data files**. Use `inspect_mat.py` to discover keys/shapes and configure `--mat-key` / `--label-col`. Each run exports metrics and plots under `reports/<run-id>/`.

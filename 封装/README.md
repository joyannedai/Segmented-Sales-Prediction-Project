# Capstone Sales Forecast

产品销售序列可预测性量化与分群建模项目。支持数据预处理、STL智能分群、多模型训练（树模型/深度学习/传统时序）、Optuna超参数调优、模型融合与短时序处理。

---

## 目录结构

```
.
├── config.yaml              # 全局配置文件（路径、超参数、随机种子）
├── requirements.txt         # Python 依赖
├── main.py                  # 统一入口脚本
├── logs/                    # 运行日志输出目录
├── output/                  # 结果与图表输出目录
└── src/
    ├── utils.py             # 配置加载、日志初始化、随机种子
    ├── data_processing.py   # 数据清洗、月度聚合、缺失月填充
    ├── clustering.py        # STL分解 + 综合加权分群
    ├── features.py          # 特征工程（时间/滞后/趋势/编码）
    ├── evaluation.py        # Baseline（均值法/季节性朴素法）
    ├── tuning.py            # Optuna 超参数调参
    ├── short_term.py        # 短时序（<24个月）建模
    ├── visualization.py     # 可视化图表
    └── models/
        ├── base.py          # 统一评估指标（MAPE/WAPE/RMSE/R2）
        ├── dl_models.py     # LSTM + Transformer（PyTorch）
        ├── tree_models.py   # RandomForest / GBDT / LightGBM / XGBoost
        ├── traditional.py   # Ridge / Prophet
        └── ensemble.py      # 简单平均/加权/中位数/截尾平均融合
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 修改配置（如需）

编辑 `config.yaml`：
- `paths.input_parquet`：输入数据路径
- `paths.output_dir`：结果输出路径
- `project.random_seed`：全局随机种子（默认 42）

### 3. 运行完整流程

```bash
python main.py
```

### 4. 分阶段运行

```bash
# 仅数据预处理
python main.py --stage data

# 仅分群
python main.py --stage cluster

# 仅长时序建模
python main.py --stage train

# 仅短时序建模
python main.py --stage short
```

### 5. Optuna 调参开关

调优默认**启用**（`config.yaml` 中 `modeling.enable_tuning: true`）。树模型会在验证集上搜索最优超参数，然后用最优参数在完整训练集上重新训练，并用于加权融合。

**关闭调优（快速验证）：**

```bash
# 方式一：命令行覆盖（推荐）
python main.py --skip-tuning

# 方式二：修改配置文件
# config.yaml -> modeling.enable_tuning: false
```

---

## 主要流程

1. **数据预处理** (`data_processing.py`)
   - 加载 `capstone_project_1000_data.parquet`
   - 剔除日期异常行（`ref_branch_end_date == 1900-01-01` 视为无限期）
   - 负 RSV 截断为 0
   - 按日去重并月度聚合（sum RSV / mean price）
   - 缺失率 >= 80% 的组合剔除
   - 缺失月份用前3月均值 + 去年同期值填充

2. **智能分群** (`clustering.py`)
   - 按时间跨度区分长/短时序（阈值 24 个月）
   - 长时序做 STL 分解，提取 CV、季节强度、残差 CV
   - 加权打分（CV 0.4 + 季节强度 0.4 + 残差 CV 0.2）
   - 按分位数切分为 high / medium / low 三组

3. **特征工程** (`features.py`)
   - 时间特征：year、month_num、quarter、month_sin/cos
   - 滞后特征：lag 1/2/3/6/12
   - 趋势特征：离散趋势（环比涨跌）、rolling_mean_3、time_idx
   - 类别编码：LabelEncoder（合并训练集+测试集统一 fit）
   - 验证集从训练集尾部切 15%，保持时序顺序

4. **模型训练**
   - **树模型**：RandomForest / GBDT / LightGBM / XGBoost
     - 若 `enable_tuning=true`：先用 Optuna 在严格训练集（train 尾部切出 val）上搜索最优超参数，记录验证集 WAPE；再用最优参数在**完整训练集**上重新训练；最后在测试集上预测
     - 若 `enable_tuning=false`：直接使用 `tree_models.py` 中的默认参数训练
   - **传统模型**：Ridge / Prophet（抽样运行）
   - **深度学习**：LSTM（BiLSTM + MultiheadAttention）/ Transformer（PositionalEncoding）

5. **模型融合** (`ensemble.py`)
   - **简单平均**：所有模型预测值取平均
   - **加权平均**：按验证集 WAPE 倒数加权（WAPE 越低权重越高）。**仅当启用调优时生效**，否则退化为简单平均
   - **中位数**：取各模型预测的中位数，抗异常值
   - **截尾平均**：排序后去掉最高和最低预测，再取平均
   - 对齐方式：取最小长度，尾部对齐（不同模型可能因数据边界差 1~2 个样本）

6. **短时序处理** (`short_term.py`)
   - 对 <24 个月的序列单独建模
   - 同样支持树模型 + Ridge + Prophet + DL + 融合

---

## 输出文件

运行后在 `output/` 目录下生成：

| 文件 | 说明 |
|---|---|
| `monthly_aggregated_filled.parquet` | 清洗填充后的月度数据 |
| `weighted_score_clusters.csv` | 分群结果标签 |
| `results_{high,medium,low}.csv` | 各组模型评估指标 |
| `results_short.csv` | 短时序模型评估指标 |
| `model_comparison_{group}.png` | 各组建模效果对比图 |
| `feature_importance_{group}_{model}.png` | 特征重要性图 |
| `all_models.pkl` | 训练好的模型缓存 |
| `logs/run.log` | 完整运行日志 |

---

## 关键设计

- **可复现**：`set_random_seed()` 统一设置 numpy / pytorch / python hash 种子
- **简洁**：每个函数只做一件事，不引入过度抽象
- **配置驱动**：所有路径和超参集中在 `config.yaml`，换机器只改配置即可；调参开关也在配置中统一管理
- **依赖完整**：`requirements.txt` 列出全部依赖，确保可迁移
- **调优与融合闭环**：调参得到的验证集 WAPE 会传递给融合模块，用于计算加权平均权重，不是拍脑袋的固定参数

---

## 原始代码

整合来源（保留未改动）：
- `capstone_week9_parquet_processing.ipynb` — 数据预处理
- `dl_models_improved.py` — 深度学习模型
- `新数据分群+建模+调参+融合+短时序.ipynb` — 主流程 notebook

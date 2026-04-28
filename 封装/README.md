# Capstone Sales Forecast

产品销售序列可预测性量化与分群建模项目。支持数据预处理、STL智能分群、多模型训练（树模型/深度学习/传统时序）、Optuna超参数调优、模型融合。

---

## 目录结构

```
.
├── config.yaml              # 全局配置文件（路径、超参数、随机种子）
├── requirements.txt         # Python 依赖
├── main.py                  # 统一入口脚本（仅调用 5 个 Stage）
├── logs/                    # 运行日志输出目录
├── output/                  # 结果与图表输出目录
└── src/
    ├── utils.py               # 配置加载、日志初始化、随机种子
    ├── data_processing.py     # Stage 1: 数据清洗、月度聚合、缺失月填充、外部特征补充、节日特征
    ├── clustering.py          # Stage 2: STL分解 + 综合加权分群
    ├── group_preparation.py   # Stage 3: 按群分割训练/测试集 + 特征工程
    ├── modeling.py            # Stage 4: 调参、训练模型、测试集预测、融合
    ├── result_analysis.py     # Stage 5: 结果对比、可视化、保存最佳模型
    ├── features.py            # 特征工程函数（时间/滞后/趋势/编码）
    ├── evaluation.py          # 评估指标（MAPE/WAPE/RMSE/MAE/R2）
    ├── tuning.py              # Optuna 超参数调参
    ├── visualization.py       # 可视化图表
    └── models/
        ├── base.py            # 基线计算（均值法/季节性朴素法）
        ├── dl_models.py       # LSTM + Transformer（PyTorch）
        ├── tree_models.py     # RandomForest / GBDT / LightGBM / XGBoost
        ├── traditional.py     # Ridge / Prophet
        └── ensemble.py        # 简单平均/加权/中位数/截尾平均融合
```

---

## 5-Stage 主流程

| Stage | 模块 | 职责 |
|-------|------|------|
| 1 | `data_processing.py` | 数据清洗、聚合、缺失值处理、外部特征补充、节日特征 |
| 2 | `clustering.py` | 可预测性打分与分群（high / medium / low） |
| 3 | `group_preparation.py` | 每个群按时间切割训练/测试集，并构建特征 |
| 4 | `modeling.py` | 调参、训练所有模型、测试集预测、模型融合 |
| 5 | `result_analysis.py` | 选出最佳模型，与 baseline 和未分群结果对比，输出可视化 |

`main.py` 仅作为调度入口，不包含具体算法逻辑。

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

## 各 Stage 详细说明

### Stage 1: 数据处理 (`data_processing.py`)

- 加载 `capstone_project_1000_data.parquet`
- 剔除日期异常行（`ref_branch_end_date == 1900-01-01` 视为无限期）
- 负 RSV 截断为 0
- 按日去重并月度聚合（sum RSV / mean price）
- 缺失率 >= 80% 的组合剔除
- 缺失月份用前3月均值 + 去年同期值填充
- 补充原始外部特征（门店属性等类别变量）
- 添加节日特征（新年、春节、劳动节、国庆）
- **只输出一个最终文件**，不再保留中间步骤的多个 parquet

### Stage 2: 智能分群 (`clustering.py`)

- 按时间跨度区分长/短时序（阈值 24 个月），短时序仅筛出、不参与后续建模
- 长时序做 STL 分解，提取 CV、季节强度、残差 CV
- 加权打分（CV 0.4 + 季节强度 0.4 + 残差 CV 0.2）
- 按分位数切分为 high / medium / low 三组
- **只保存带有 `predictability_level` 标签的完整数据**，删除高中低分开保存的冗余

### Stage 3: 群内准备 (`group_preparation.py`)

- 按群过滤数据
- 按时间顺序切割训练/测试集（测试占20%）
- 构建特征：时间特征、滞后特征、趋势特征、类别编码
- 从训练集尾部切出 15% 作为验证集，保持时序顺序
- `return` 处理后的数据字典，直接传递到下一步

### Stage 4: 模型建模 (`modeling.py`)

- **树模型**：RandomForest / GBDT / LightGBM / XGBoost
  - 若 `enable_tuning=true`：先用 Optuna 在严格训练集（切出 val）上搜索最优超参数，记录验证集 WAPE；再用最优参数在完整训练集上重新训练；最后在测试集上预测
  - 若 `enable_tuning=false`：直接使用默认参数训练
- **传统模型**：Ridge / Prophet（抽样运行）
- **深度学习**：LSTM（BiLSTM + MultiheadAttention） / Transformer（PositionalEncoding）
- **融合**：简单平均 / 加权平均（按验证集 WAPE 倒数加权） / 中位数 / 截尾平均

### Stage 5: 结果分析 (`result_analysis.py`)

- 将各模型指标汇总为 DataFrame
- 计算相对于 Baseline 的提升率
- 选出每组最佳模型
- 生成对比图（`model_comparison_{group}.png`）
- 生成树模型特征重要性图（`feature_importance_{group}_{model}.png`）
- 保存结果 CSV（`results_{high,medium,low}.csv`）

## 输出文件

运行后在 `output/` 目录下生成：

| 文件 | 说明 |
|---|---|
| `monthly_aggregated_sales_predict.parquet` | 完整处理后带有可预测性标签的数据 |
| `weighted_score_clusters.csv` | 分群结果标签 |
| `results_{high,medium,low}.csv` | 各组模型评估指标 |
| `model_comparison_{group}.png` | 各组建模效果对比图 |
| `feature_importance_{group}_{model}.png` | 特征重要性图 |
| `all_models.pkl` | 训练好的模型缓存 |
| `logs/run.log` | 完整运行日志 |

---


# Capstone Sales Forecast

产品销售序列可预测性量化与分群建模项目。项目以 `main.py` 为统一入口，按 `config.yaml` 驱动，完成数据清洗、可预测性分群、分组建模、短时序建模、融合评估和结果输出。

## 目录结构

```text
.
├── config.yaml                # 路径、阈值、特征、模型、调参开关
├── tuned_params.json          # high / medium / low / short 树模型预调参数
├── requirements.txt           # 项目依赖
├── main.py                    # 统一入口，只负责调度
├── logs/                      # 运行日志
├── output/                    # 结果、图表、模型缓存
└── src/
    ├── data_processing.py     # Stage 1: 数据清洗与月度聚合
    ├── clustering.py          # Stage 2: STL/CV 指标与可预测性分群
    ├── group_preparation.py   # Stage 3: high/medium/low 训练测试划分
    ├── features.py            # 长时序特征工程
    ├── modeling.py            # high/medium/low 建模、融合
    ├── short_term_modeling.py # short 短时序建模、短时序调参
    ├── result_analysis.py     # 结果保存、可视化、最佳模型选择
    ├── evaluation.py          # MAPE/WAPE/RMSE/MAE/R2
    ├── tuning.py              # Optuna 搜索空间
    ├── visualization.py       # 模型对比图、特征重要性图
    └── models/
        ├── base.py            # Baseline
        ├── tree_models.py     # RandomForest / GBDT / LightGBM / XGBoost
        ├── traditional.py     # Ridge
        ├── dl_models.py       # LSTM / Transformer
        └── ensemble.py        # Avg / Weighted / Median / Trimmed 融合
```

## 运行命令

```bash
# 完整流水线：数据处理 -> 分群 -> 建模 -> 结果分析
python main.py

# 快速建模：关闭长时序实时调参，优先读取 tuned_params.json
python main.py --skip-tuning

# 只跑数据处理
python main.py --stage data

# 只跑分群
python main.py --stage cluster

# 只跑建模和结果分析
python main.py --stage train

# 只调短时序树模型参数，并写回 tuned_params.json 的 short 分组
python main.py --stage train --tune-short-params
```

## Pipeline 说明

| Stage | 模块 | 输出 |
|---|---|---|
| Stage 1 | `data_processing.py` | `output/monthly_aggregated_sales_predict.parquet` |
| Stage 2 | `clustering.py` | `output/weighted_score_clusters.csv` |
| Stage 3 | `group_preparation.py` + `features.py` | high/medium/low 的训练、验证、测试特征 |
| Stage 4 | `modeling.py` | high/medium/low 模型结果与模型对象 |
| Stage 4-5 | `short_term_modeling.py` | short 组模型结果与模型对象 |
| Stage 5 | `result_analysis.py` | CSV 结果、图表、最佳模型日志 |

## 数据处理逻辑

Stage 1 做以下处理：

- 读取 `paths.input_parquet`。
- 删除不在门店有效营业日期范围内的记录。
- 将负 RSV 截断为 0。
- 按 `ref_branch_code + material_nature_sum_desc + stock_out_date` 做日粒度去重。
- 聚合到月度销量 `monthly_sales` 和月均价格 `price`。
- 统计并剔除缺失率过高的门店-商品组合。
- 对缺失月份补齐，并用近 3 月均值和去年同期值填补。
- 补充门店、城市、区域、品类等类别特征。
- 补充节假日特征。

## 分群逻辑

Stage 2 按 `ref_branch_code + material_nature_sum_desc` 计算每个序列的月份跨度。

- 跨度 `< clustering.long_term_threshold`：标记为 `short`。
- 跨度 `>= clustering.long_term_threshold`：进入 STL 可预测性评分。

长时序评分指标：

- `CV`
- `seasonal_strength`
- `residual_cv`

综合分数：

```text
score = 0.4 * (1 - CV_pct)
      + 0.4 * seasonal_strength_pct
      + 0.2 * (1 - residual_cv_pct)
```

再按分位数切分：

- `high`
- `medium`
- `low`

## 长时序建模逻辑

长时序包括 `high`、`medium`、`low` 三组。

### 特征

长时序特征包括：

- 时间特征：year、month_num、quarter、month_sin、month_cos。
- 价格和节假日特征。
- 类别编码特征。
- 门店编码 `store_code`、商品编码 `prod_code`。
- 滞后特征：lag_1、lag_2、lag_3、lag_6、lag_12。
- 趋势特征：discrete_trend、time_idx、rolling_mean_3。

训练集内部再切出验证集：

```text
X_train_strict = 训练集前 85%
X_val          = 训练集后 15%
X_test         = 最后 20% 测试集
```

### 模型

当前主流程训练：

- RandomForest
- GBDT
- LightGBM
- XGBoost
- Ridge
- LSTM
- Transformer

### 树模型训练策略

项目默认 `modeling.enable_tuning: false`，因此树模型优先读取 `tuned_params.json`。

不同组使用不同训练策略：

- `high`、`medium`：读取参数后，用完整 `X_train` 训练，再预测 `X_test`。
- `low`：读取参数后，只用 `X_train_strict` 训练，再预测 `X_test`。这是 low 组当前实验表现更好的口径。

如果打开 `modeling.enable_tuning: true`：

- 先用 `X_train_strict` 训练并在 `X_val` 上调参。
- high/medium 使用最优参数在完整 `X_train` 上训练。
- low 使用最优参数仍在 `X_train_strict` 上训练。

### 融合逻辑

参与融合的预测包括：

- 四个树模型预测。
- Ridge 预测。

融合方法：

- `Ensemble_Avg`
- `Ensemble_Weighted`
- `Ensemble_Median`
- `Ensemble_Trimmed`

当存在验证集 WAPE 时，Weighted 使用验证 WAPE 的倒数作为权重；否则 Weighted 退化为普通平均。

## 短时序建模逻辑

短时序组为：

```text
predictability_level = short
```

短时序由于历史长度不足，不使用测试期真实销量生成 lag 特征。短时序特征包括：

- 时间特征。
- 价格和节假日特征。
- 类别编码特征。
- 门店、商品编码。
- 训练历史统计特征：
  - hist_len
  - hist_mean
  - hist_median
  - hist_std
  - hist_min
  - hist_max
  - hist_last
  - hist_last2_mean
  - hist_last3_mean
  - hist_nonzero_mean
  - hist_zero_rate
  - hist_trend_last_first

短时序模型包括：

- Baseline_Mean
- Baseline_LastValue
- Baseline_MovingAvg3
- Baseline_SeasonalNaive
- RandomForest
- GBDT
- LightGBM
- XGBoost
- Ridge
- LSTM / Transformer（由 `short_term.enable_dl` 控制）
- Ensemble

短时序树模型参数读取：

- 默认 `short_term.enable_tuning: false`。
- 若 `tuned_params.json` 中存在 `short` 分组，直接读取。
- 若不存在，回退到代码默认参数。

短时序调参：

```bash
python main.py --stage train --tune-short-params
```

该命令只把短时序树模型最优参数写入：

```text
tuned_params.json -> short
```

不额外输出短时序调参明细 CSV。

## 配置重点

```yaml
modeling:
  enable_tuning: false
  tree_models:
    - RandomForest
    - GBDT
    - LightGBM
    - XGBoost
  non_tree_models:
    - Ridge
    - LSTM
    - Transformer

short_term:
  enable_modeling: true
  enable_tuning: false
  enable_dl: true
  min_months: 6
  test_ratio: 0.2
```

说明：

- `modeling.enable_tuning` 控制 high/medium/low 是否实时 Optuna 调参。
- `short_term.enable_tuning` 控制 short 是否在普通建模中实时调参。
- `--skip-tuning` 只覆盖长时序实时调参。
- `--tune-short-params` 是短时序参数写入入口。

## 输出文件

| 文件 | 说明 |
|---|---|
| `output/monthly_aggregated_sales_predict.parquet` | 处理后并带分群标签的数据 |
| `output/weighted_score_clusters.csv` | high/medium/low/short 标签与评分指标 |
| `output/results_high.csv` | high 组结果 |
| `output/results_medium.csv` | medium 组结果 |
| `output/results_low.csv` | low 组结果 |
| `output/results_short.csv` | short 组结果 |
| `output/model_comparison_{group}.png` | 各组模型 WAPE 对比图 |
| `output/feature_importance_{group}_{model}.png` | 模型特征重要性图 |
| `output/all_models.pkl` | 所有已训练模型 |
| `logs/run.log` | 运行日志 |

## 复现建议

推荐顺序：

```bash
python main.py --skip-tuning
```

如果需要重新生成短时序参数：

```bash
python main.py --stage train --tune-short-params
python main.py --skip-tuning
```

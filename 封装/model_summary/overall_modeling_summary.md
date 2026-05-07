# 整体建模流程与结果汇总

## 整体建模流程

1. 数据处理：读取原始 parquet，完成月度聚合、缺失月份补齐、价格与节假日等外部特征补充。
2. 可预测性分群：先按序列跨度识别 short 组；长时序再基于 CV、seasonal_strength、residual_cv 计算综合得分，划分 high / medium / low。
3. 特征准备：对 high / medium / low 构造时间、类别、价格、节假日、lag、趋势和滚动统计特征，并按时间顺序切分训练、验证和测试集。
4. 分组建模：对 high / medium / low 训练树模型、Ridge 和深度模型；short 组使用独立短时序特征和模型流程。
5. 融合评估：对可用模型预测做 Avg、Weighted、Median、Trimmed 融合，并用 WAPE、MAPE、RMSE、MAE、R2 评估。
6. 结果输出：保存各组 results_*.csv、模型对比图、特征重要性图和模型对象。

## 整体建模结果

| 分组 | 最佳模型 | 最佳 WAPE | 最佳单模型 | 单模型 WAPE | 基线 | 基线 WAPE | 相对基线提升 |
|---|---:|---:|---:|---:|---:|---:|---:|
| high | RandomForest | 24.96% | RandomForest | 24.96% | Internal_Baseline | 52.11% | 52.09% |
| medium | Ensemble_Median | 29.58% | XGBoost | 30.08% | Internal_Baseline | 72.82% | 59.38% |
| low | RandomForest | 59.79% | RandomForest | 59.79% | Internal_Baseline | 96.45% | 38.01% |
| short | Ensemble_Median | 53.04% | RandomForest | 53.16% | Baseline_MovingAvg3 | 75.11% | 29.39% |

## 结果口径说明

- 主展示指标使用 WAPE，原因是部分销量接近 0 时 MAPE 会异常放大。
- high / medium / low 当前结果文件没有保留 baseline 明细，因此基线 WAPE 由 improvement_vs_baseline 反推。
- short 组结果文件保留了 baseline 明细，因此直接取 short 组 baseline 中 WAPE 最低者。
- low 组代码已按 notebook 逻辑调整，建议在最终汇报前重跑封装流程并复核 low 组最终指标。

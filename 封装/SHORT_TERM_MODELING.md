# Short-Term Series Modeling

This package now models short-term series instead of only filtering them out.

## Definition

Short-term series are product-store combinations whose observed span is below
`clustering.long_term_threshold` months. The clustering stage labels them as
`predictability_level = short`.

## Method

The short-term pipeline uses methods already present in the project:

- Baselines: mean, last value, 3-month moving average, seasonal naive.
- Tree models: RandomForest, GBDT, LightGBM, XGBoost.
- Ridge regression.
- Prophet sample evaluation.
- Ensemble fusion: average, weighted average, median, trimmed mean.

Short-term features avoid target lags from the test period. They use calendar,
categorical, price, group-code, and train-history summary features such as
history mean, last value, last-3 mean, zero rate, and trend.

## Config

`config.yaml` controls the behavior:

```yaml
short_term:
  enable_modeling: true
  enable_tuning: false
  enable_dl: false
  min_months: 6
  test_ratio: 0.2
  prophet_sample_size: 50
```

## Outputs

The short-term stage writes:

- `output/results_short.csv`
- `output/model_comparison_short.png`
- `output/feature_importance_short_{model}.png`

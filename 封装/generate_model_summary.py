from pathlib import Path
import csv
import json

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "output"
SUMMARY_DIR = ROOT / "model_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["high", "medium", "low", "short"]

COLORS = {
    "high": "#2563EB",
    "medium": "#0F766E",
    "low": "#B45309",
    "short": "#7C3AED",
    "baseline": "#CBD5E1",
    "ink": "#172033",
    "soft": "#5D6678",
    "line": "#D9DEE8",
    "bg": "#F7F8FA",
    "white": "#FFFFFF",
}


def load_font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def read_group_results(group):
    path = RESULT_DIR / f"results_{group}.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["wape"] = float(row["wape"])
        row["improvement_vs_baseline"] = float(row.get("improvement_vs_baseline") or 0)
    return rows


def summarize_group(group):
    rows = read_group_results(group)
    baselines = [r for r in rows if r["model"].startswith("Baseline")]
    models = [r for r in rows if not r["model"].startswith("Baseline")]
    single_models = [r for r in models if not r["model"].startswith("Ensemble")]

    best = min(models, key=lambda r: r["wape"])
    best_single = min(single_models, key=lambda r: r["wape"])

    if baselines:
        baseline = min(baselines, key=lambda r: r["wape"])
        baseline_model = baseline["model"]
        baseline_wape = baseline["wape"]
    else:
        # Long-series result files only keep the improvement percentage, so
        # reconstruct the baseline WAPE used by the pipeline.
        improvement = best["improvement_vs_baseline"]
        baseline_model = "Internal_Baseline"
        baseline_wape = best["wape"] / (1 - improvement / 100) if improvement < 100 else best["wape"]

    return {
        "group": group,
        "best_model": best["model"],
        "best_wape": best["wape"],
        "best_single_model": best_single["model"],
        "best_single_wape": best_single["wape"],
        "baseline_model": baseline_model,
        "baseline_wape": baseline_wape,
        "improvement_pct": best["improvement_vs_baseline"],
    }


def save_summary_files(summary):
    csv_path = SUMMARY_DIR / "overall_model_results_summary.csv"
    json_path = SUMMARY_DIR / "overall_model_results_summary.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def draw_results_table(summary):
    width, height = 1500, 620
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    title_font = load_font(36, True)
    header_font = load_font(22, True)
    body_font = load_font(23)
    small_font = load_font(19)

    draw.text((48, 36), "整体建模结果汇总", fill=COLORS["ink"], font=title_font)
    draw.text((50, 88), "口径：最佳模型不含 baseline；基线取各组 baseline 中 WAPE 最低者。", fill=COLORS["soft"], font=small_font)

    x0, y0 = 50, 145
    col_widths = [120, 250, 140, 250, 150, 240, 150]
    headers = ["分组", "最佳模型", "WAPE", "最佳单模型", "单模型WAPE", "最佳基线", "提升"]
    row_h = 70

    draw.rectangle((x0, y0, x0 + sum(col_widths), y0 + row_h), fill="#E8EEF8")
    x = x0
    for header, col_w in zip(headers, col_widths):
        draw.text((x + 16, y0 + 22), header, fill=COLORS["ink"], font=header_font)
        x += col_w

    for idx, row in enumerate(summary):
        y = y0 + row_h * (idx + 1)
        fill = COLORS["white"] if idx % 2 == 0 else "#F1F5F9"
        draw.rectangle((x0, y, x0 + sum(col_widths), y + row_h), fill=fill)
        values = [
            row["group"],
            row["best_model"],
            f'{row["best_wape"]:.2f}%',
            row["best_single_model"],
            f'{row["best_single_wape"]:.2f}%',
            row["baseline_model"].replace("Baseline_", ""),
            f'{row["improvement_pct"]:.2f}%',
        ]
        x = x0
        for col_idx, (value, col_w) in enumerate(zip(values, col_widths)):
            color = COLORS.get(row["group"], COLORS["ink"]) if col_idx in [0, 2, 6] else COLORS["ink"]
            draw.text((x + 16, y + 22), str(value), fill=color, font=body_font)
            x += col_w

    x = x0
    for col_w in col_widths:
        draw.line((x, y0, x, y0 + row_h * (len(summary) + 1)), fill=COLORS["line"], width=1)
        x += col_w
    draw.line((x, y0, x, y0 + row_h * (len(summary) + 1)), fill=COLORS["line"], width=1)

    draw.text(
        (50, 545),
        "注：high/medium/low 的基线由 improvement_vs_baseline 反推；low 组建议重跑后复核最终封装指标。",
        fill=COLORS["soft"],
        font=small_font,
    )
    image.save(SUMMARY_DIR / "overall_model_results_table.png")


def draw_results_chart(summary):
    width, height = 1500, 850
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    title_font = load_font(38, True)
    axis_font = load_font(20)
    label_font = load_font(22, True)
    small_font = load_font(18)

    draw.text((55, 35), "各分组最佳模型 WAPE 与基线对比", fill=COLORS["ink"], font=title_font)
    draw.text((58, 90), "WAPE 越低越好；彩色柱为最佳模型，灰色柱为最佳 baseline。", fill=COLORS["soft"], font=small_font)

    chart_x, chart_y = 95, 150
    chart_w, chart_h = 1240, 520
    max_value = max(max(r["best_wape"], r["baseline_wape"]) for r in summary) * 1.18

    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=COLORS["line"], width=2)
    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill=COLORS["line"], width=2)

    for tick in range(0, 101, 20):
        y = chart_y + chart_h - tick / max_value * chart_h
        if chart_y <= y <= chart_y + chart_h:
            draw.line((chart_x - 8, y, chart_x + chart_w, y), fill="#E5E7EB", width=1)
            draw.text((38, y - 12), f"{tick}%", fill=COLORS["soft"], font=axis_font)

    group_gap = chart_w / len(summary)
    bar_w = 86
    for idx, row in enumerate(summary):
        cx = chart_x + group_gap * idx + group_gap / 2
        for key, offset, color in [
            ("baseline_wape", -52, COLORS["baseline"]),
            ("best_wape", 52, COLORS[row["group"]]),
        ]:
            value = row[key]
            bar_h = value / max_value * chart_h
            x1 = cx + offset - bar_w / 2
            x2 = cx + offset + bar_w / 2
            y1 = chart_y + chart_h - bar_h
            y2 = chart_y + chart_h
            draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=color)
            draw.text((x1 - 4, y1 - 32), f"{value:.1f}%", fill=COLORS["ink"], font=small_font)
        draw.text((cx - 34, chart_y + chart_h + 28), row["group"], fill=COLORS["ink"], font=label_font)

    draw.rectangle((1030, 62, 1320, 112), fill=COLORS["white"], outline=COLORS["line"])
    draw.rectangle((1050, 80, 1078, 96), fill=COLORS["baseline"])
    draw.text((1088, 75), "最佳 baseline", fill=COLORS["soft"], font=small_font)
    draw.rectangle((1192, 80, 1220, 96), fill=COLORS["high"])
    draw.text((1230, 75), "最佳模型", fill=COLORS["soft"], font=small_font)

    draw.text(
        (95, 735),
        "结论：四个分组的最佳模型均优于对应 baseline，短时序建模后 short 组也形成了可量化提升。",
        fill=COLORS["ink"],
        font=label_font,
    )
    image.save(SUMMARY_DIR / "overall_model_results_chart.png")


def save_process_markdown(summary):
    table_rows = "\n".join(
        "| {group} | {best_model} | {best_wape:.2f}% | {best_single_model} | {best_single_wape:.2f}% | {baseline_model} | {baseline_wape:.2f}% | {improvement_pct:.2f}% |".format(
            **row
        )
        for row in summary
    )
    text = f"""# 整体建模流程与结果汇总

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
{table_rows}

## 结果口径说明

- 主展示指标使用 WAPE，原因是部分销量接近 0 时 MAPE 会异常放大。
- high / medium / low 当前结果文件没有保留 baseline 明细，因此基线 WAPE 由 improvement_vs_baseline 反推。
- short 组结果文件保留了 baseline 明细，因此直接取 short 组 baseline 中 WAPE 最低者。
- low 组代码已按 notebook 逻辑调整，建议在最终汇报前重跑封装流程并复核 low 组最终指标。
"""
    (SUMMARY_DIR / "overall_modeling_summary.md").write_text(text, encoding="utf-8")


def main():
    summary = [summarize_group(group) for group in GROUPS]
    save_summary_files(summary)
    save_process_markdown(summary)
    draw_results_table(summary)
    draw_results_chart(summary)
    print(SUMMARY_DIR)


if __name__ == "__main__":
    main()

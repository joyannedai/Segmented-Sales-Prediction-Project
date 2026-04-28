import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(path: str, columns: list = None) -> pd.DataFrame:
    if columns:
        df = pd.read_parquet(path, columns=columns, engine="pyarrow")
    else:
        df = pd.read_parquet(path, engine="pyarrow")
    for c in ["stock_out_date", "ref_branch_start_date", "ref_branch_end_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def remove_invalid_date_rows(df: pd.DataFrame) -> pd.DataFrame:
    infinite_end = pd.Timestamp("1900-01-01")
    invalid = (
        (df["stock_out_date"] < df["ref_branch_start_date"]) |
        ((df["ref_branch_end_date"] != infinite_end) & (df["stock_out_date"] > df["ref_branch_end_date"]))
    )
    return df.loc[~invalid].copy()


def clamp_negative_rsv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "RSV" in df.columns:
        df["RSV"] = df["RSV"].clip(lower=0)
    return df


def deduplicate_daily(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["ref_branch_code", "material_nature_sum_desc", "stock_out_date"]
    df = df.sort_values(keys)
    return df.groupby(keys, as_index=False).agg(
        RSV=("RSV", "mean"),
        price=("price", "first"),
    )


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stock_out_month"] = df["stock_out_date"].dt.to_period("M").dt.to_timestamp()
    agg = (
        df.groupby(["ref_branch_code", "material_nature_sum_desc", "stock_out_month"], as_index=False)
        .agg(RSV=("RSV", "sum"), price=("price", "mean"))
    )
    agg = agg.rename(columns={
        "material_nature_sum_desc": "material_nature_sum",
        "stock_out_month": "stock_out_date",
    })
    return agg[["ref_branch_code", "material_nature_sum", "stock_out_date", "RSV", "price"]]


def check_missing_months(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["ref_branch_code", "material_nature_sum"]
    df_sorted = df.sort_values(keys + ["stock_out_date"]).copy()
    summary_rows, detail_rows = [], []

    for (bcode, msum), g in df_sorted.groupby(keys, sort=False):
        g = g.sort_values("stock_out_date")
        min_dt, max_dt = g["stock_out_date"].min(), g["stock_out_date"].max()
        expected = pd.date_range(min_dt, max_dt, freq="MS")
        present = pd.DatetimeIndex(g["stock_out_date"].dropna().unique())
        missing = expected.difference(present)

        total_expected = len(expected)
        missing_count = len(missing)
        missing_rate = missing_count / total_expected if total_expected else np.nan

        summary_rows.append({
            "ref_branch_code": bcode,
            "material_nature_sum": msum,
            "min_stock_out_date": min_dt,
            "max_stock_out_date": max_dt,
            "expected_months": total_expected,
            "present_months": len(present),
            "missing_months_count": missing_count,
            "missing_rate": missing_rate,
        })

        for dt in missing:
            detail_rows.append({
                "ref_branch_code": bcode,
                "material_nature_sum": msum,
                "missing_stock_out_date": dt,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def drop_high_missing_rate(
    df: pd.DataFrame, missing_summary: pd.DataFrame, threshold: float = 0.8
) -> pd.DataFrame:
    keep_pairs = missing_summary.loc[missing_summary["missing_rate"].fillna(0) < threshold,
                                      ["ref_branch_code", "material_nature_sum"]]
    return df.merge(keep_pairs, on=["ref_branch_code", "material_nature_sum"], how="inner")


def fill_missing_months(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["ref_branch_code", "material_nature_sum"]
    df_sorted = df.sort_values(keys + ["stock_out_date"]).copy()
    filled_parts = []

    for (bcode, msum), g in df_sorted.groupby(keys, sort=False):
        g = g.sort_values("stock_out_date").set_index("stock_out_date")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g_full = g.reindex(full_idx)
        g_full["ref_branch_code"] = bcode
        g_full["material_nature_sum"] = msum

        for col in ["RSV", "price"]:
            s = g_full[col].copy()
            for i in range(len(s)):
                if pd.isna(s.iat[i]):
                    mov3 = s.iloc[max(0, i - 3):i].mean()
                    yoy = s.iat[i - 12] if i >= 12 else np.nan
                    s.iat[i] = np.nanmean([mov3, yoy])
            g_full[col] = s

        g_full = g_full.reset_index().rename(columns={"index": "stock_out_date"})
        filled_parts.append(g_full[["ref_branch_code", "material_nature_sum", "stock_out_date", "RSV", "price"]])

    df_full = pd.concat(filled_parts, ignore_index=True)
    df_full = df_full.sort_values(keys + ["stock_out_date"]).reset_index(drop=True)
    df_final = df_full.rename(columns={
        "stock_out_date": "month",
        "RSV": "monthly_sales",
        "material_nature_sum": "material_nature_sum_desc",
    })
    return df_final[["ref_branch_code", "material_nature_sum_desc", "month", "monthly_sales", "price"]]


def enrich_with_raw_features(aggregated_file: str, raw_file: str) -> pd.DataFrame:
    df_agg = pd.read_parquet(aggregated_file)
    df_agg = df_agg[["ref_branch_code", "material_nature_sum_desc", "month", "monthly_sales", "price"]].copy()

    raw_cols = [
        "ref_branch_code", "material_nature_sum_desc",
        "Business_Type_1_Desc", "Type_Group_Desc", "Shop_Style_Desc",
        "City_Description", "City", "Province_Description",
        "City_Level_Description", "District_Desc", "Geographic_Region_Desc",
        "Mall_Scale_Code_Desc", "shop_type_desc",
    ]
    df_raw = pd.read_parquet(raw_file, columns=raw_cols + ["stock_out_date"])
    df_raw = df_raw.rename(columns={"stock_out_date": "month"})
    df_raw["month"] = pd.to_datetime(df_raw["month"]).dt.to_period("M").dt.to_timestamp()
    df_raw = df_raw.drop_duplicates(subset=["ref_branch_code", "material_nature_sum_desc", "month"])

    return df_agg.merge(df_raw, on=["ref_branch_code", "material_nature_sum_desc", "month"], how="left")


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import chinese_calendar as cc
    except ImportError:
        logger.warning("chinese_calendar not installed, skipping holiday features")
        return df

    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month

    def get_holiday_ranges(year):
        ranges = {}
        ranges["NewYear"] = pd.DatetimeIndex([pd.Timestamp(f"{year}-01-01")])
        try:
            ranges["CNY"] = pd.date_range(start=cc.get_holiday_detail(pd.Timestamp(f"{year}-01-01"))[0],
                                           periods=7, freq="D") if cc.get_holiday_detail(pd.Timestamp(f"{year}-01-01"))[1] else pd.date_range(start=pd.Timestamp(f"{year}-02-01"), periods=7, freq="D")
        except Exception:
            ranges["CNY"] = pd.date_range(start=pd.Timestamp(f"{year}-02-01"), periods=7, freq="D")
        ranges["LaborDay"] = pd.date_range(start=pd.Timestamp(f"{year}-05-01"), periods=5, freq="D")
        ranges["NationalDay"] = pd.date_range(start=pd.Timestamp(f"{year}-10-01"), periods=7, freq="D")
        return ranges

    def count_holidays_in_month(month_ts):
        year = month_ts.year
        month_start = month_ts.replace(day=1)
        month_end = (month_start + pd.offsets.MonthEnd(0))
        ranges = get_holiday_ranges(year)
        counts = {}
        for name, dates in ranges.items():
            counts[name] = ((dates >= month_start) & (dates <= month_end)).sum()
        counts["total_holiday_days"] = sum(counts.values())
        for name in ["NewYear", "CNY", "LaborDay", "NationalDay"]:
            counts[f"holiday_{name}_flag"] = 1 if counts[name] > 0 else 0
        return counts

    holiday_data = df["month"].apply(count_holidays_in_month).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), holiday_data.reset_index(drop=True)], axis=1)


def run_data_pipeline(input_path: str, raw_file_path: str, missing_threshold: float = 0.8) -> pd.DataFrame:
    logger.info("Step 0: Load raw data")
    cols = [
        "ref_branch_code", "material_nature_sum_desc", "stock_out_date",
        "RSV", "price", "ref_branch_start_date", "ref_branch_end_date",
    ]
    df = load_raw_data(input_path, columns=cols)

    logger.info("Step 1: Remove invalid date rows")
    df = remove_invalid_date_rows(df)

    logger.info("Step 2: Clamp negative RSV")
    df = clamp_negative_rsv(df)

    logger.info("Step 3: Deduplicate daily")
    df = deduplicate_daily(df)

    logger.info("Step 4: Aggregate monthly")
    df_monthly = aggregate_monthly(df)

    logger.info("Step 5: Check missing months")
    missing_summary, missing_detail = check_missing_months(df_monthly)

    logger.info("Step 5.1: Drop high missing-rate pairs")
    df_monthly = drop_high_missing_rate(df_monthly, missing_summary, threshold=missing_threshold)

    logger.info("Step 6: Fill missing months")
    df_filled = fill_missing_months(df_monthly)

    logger.info("Step 7: Enrich with raw categorical features")
    raw_cols = [
        "ref_branch_code", "material_nature_sum_desc",
        "Business_Type_1_Desc", "Type_Group_Desc", "Shop_Style_Desc",
        "City_Description", "City", "Province_Description",
        "City_Level_Description", "District_Desc", "Geographic_Region_Desc",
        "Mall_Scale_Code_Desc", "shop_type_desc",
    ]
    df_raw = pd.read_parquet(raw_file_path, columns=raw_cols + ["stock_out_date"])
    df_raw = df_raw.rename(columns={"stock_out_date": "month"})
    df_raw["month"] = pd.to_datetime(df_raw["month"]).dt.to_period("M").dt.to_timestamp()
    df_raw = df_raw.drop_duplicates(subset=["ref_branch_code", "material_nature_sum_desc", "month"])
    df_filled = df_filled.merge(df_raw, on=["ref_branch_code", "material_nature_sum_desc", "month"], how="left")

    logger.info("Step 8: Add holiday features")
    df_filled = add_holiday_features(df_filled)

    return df_filled

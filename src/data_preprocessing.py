# src/data_preprocessing.py
"""
Data Preprocessing Module
Function: Raw data cleaning + monthly aggregation + missing value imputation
Input: Path to raw parquet file
Output: Cleaned monthly DataFrame ready for further analysis
"""

import pandas as pd
import numpy as np


def run_data_pipeline(input_path, save_path=None, verbose=True):
    """
    Data preparation pipeline
    
    Parameters
    ----------
    input_path : str
        Path to raw parquet file
    verbose : bool, default=True
        Whether to print progress information
    
    Returns
    -------
    df_final : pd.DataFrame
        Cleaned monthly data with columns:
        - ref_branch_code: store code
        - material_nature_sum_desc: product category
        - month: month
        - monthly_sales: monthly sales amount
        - price: monthly average price
    """
    
    # ========== Step 0: Load data ==========
    cols = [
        "ref_branch_code",
        "material_nature_sum_desc",
        "stock_out_date",
        "RSV",
        "price",
        "ref_branch_start_date",
        "ref_branch_end_date",
    ]
    
    df = pd.read_parquet(input_path, columns=cols, engine="pyarrow")
    
    # Ensure datetime types
    for c in ["stock_out_date", "ref_branch_start_date", "ref_branch_end_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    
    if verbose:
        print(f"[1/6] data loaded: {len(df):,} rows")
    
    # ========== Step 1: Remove invalid date rows ==========
    infinite_end = pd.Timestamp("1900-01-01")
    invalid = (
        (df["stock_out_date"] < df["ref_branch_start_date"]) |
        ((df["ref_branch_end_date"] != infinite_end) & 
         (df["stock_out_date"] > df["ref_branch_end_date"]))
    )
    df = df.loc[~invalid].copy()
    
    if verbose:
        print(f"[2/6] After removing invalid dates: {len(df):,} rows")
    
    # ========== Step 2: Clamp negative RSV to 0 ==========
    df["RSV"] = df["RSV"].clip(lower=0)

    df = df[[
        "ref_branch_code",
        "material_nature_sum_desc",
        "stock_out_date",
        "RSV",
        "price",
    ]]
    
    if verbose:
        print(f"[3/6] Negative RSV clamped to 0")
    
    # ========== Step 3: Deduplicate ==========
    keys3 = ["ref_branch_code", "material_nature_sum_desc", "stock_out_date"]
    df = df.sort_values(keys3)
    
    df = (
        df.groupby(keys3, as_index=False)
          .agg(
              RSV=("RSV", "mean"),
              price=("price", "first"),
          )
    )
    
    if verbose:
        print(f"[4/6] After deduplication: {len(df):,} rows")
    
    # ========== Step 4: Monthly aggregation ==========
    df = df.copy()
    df["stock_out_month"] = df["stock_out_date"].dt.to_period("M").dt.to_timestamp()
    
    df_monthly = (
        df.groupby(["ref_branch_code", "material_nature_sum_desc", "stock_out_month"], as_index=False)
          .agg(
              RSV=("RSV", "sum"),
              price=("price", "mean"),
          )
    )
    
    df_monthly = df_monthly.rename(columns={
        "material_nature_sum_desc": "material_nature_sum",
        "stock_out_month": "stock_out_date",
    })
    
    df_monthly = df_monthly[["ref_branch_code", "material_nature_sum", "stock_out_date", "RSV", "price"]]
    
    if verbose:
        print(f"[5/6] After monthly aggregation: {len(df_monthly):,} rows")
    
    # ========== Step 5: Check missing months, drop high-missing combos ==========
    keys5 = ["ref_branch_code", "material_nature_sum"]
    df_sorted = df_monthly.sort_values(keys5 + ["stock_out_date"]).copy()
    
    keep_pairs = []
    for (bcode, msum), g in df_sorted.groupby(keys5, sort=False):
        min_dt = g["stock_out_date"].min()
        max_dt = g["stock_out_date"].max()
        expected = pd.date_range(min_dt, max_dt, freq="MS")
        present = pd.DatetimeIndex(g["stock_out_date"].dropna().unique())
        missing_rate = 1 - len(present) / len(expected) if len(expected) > 0 else 1
        
        if missing_rate < 0.8:  # Keep pairs with missing rate < 80%
            keep_pairs.append((bcode, msum))
    
    keep_df = pd.DataFrame(keep_pairs, columns=keys5)
    df_filtered = df_monthly.merge(keep_df, on=keys5, how="inner")
    
    if verbose:
        print(f"[6/6] After dropping high-missing combos: {len(df_filtered):,} rows")
    
    # ========== Step 6: Fill missing months ==========
    df_sorted = df_filtered.sort_values(keys5 + ["stock_out_date"]).copy()
    filled_parts = []
    
    for (bcode, msum), g in df_sorted.groupby(keys5, sort=False):
        g = g.sort_values("stock_out_date").set_index("stock_out_date")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g_full = g.reindex(full_idx)
        g_full["ref_branch_code"] = bcode
        g_full["material_nature_sum"] = msum
        
        rsv = g_full["RSV"].copy()
        price = g_full["price"].copy()
        
        for i in range(len(rsv)):
            if pd.isna(rsv.iat[i]):
                mov3 = rsv.iloc[max(0, i-3):i].mean()
                yoy = rsv.iat[i-12] if i >= 12 else np.nan
                rsv.iat[i] = np.nanmean([mov3, yoy])
            
            if pd.isna(price.iat[i]):
                mov3 = price.iloc[max(0, i-3):i].mean()
                yoy = price.iat[i-12] if i >= 12 else np.nan
                price.iat[i] = np.nanmean([mov3, yoy])
        
        g_full["RSV"] = rsv
        g_full["price"] = price
        g_full = g_full.reset_index().rename(columns={"index": "stock_out_date"})
        filled_parts.append(g_full[["ref_branch_code", "material_nature_sum", "stock_out_date", "RSV", "price"]])
    
    df_full = pd.concat(filled_parts, ignore_index=True)
    
    # Rename output columns
    df_final = df_full.rename(columns={
        "stock_out_date": "month",
        "RSV": "monthly_sales",
        "material_nature_sum": "material_nature_sum_desc",
    })
    
    df_final = df_final[["ref_branch_code", "material_nature_sum_desc", "month", "monthly_sales", "price"]]
    
    if verbose:
        print(f"Data preparation complete! Final shape: {df_final.shape}")
    

    # save file
    if save_path:
        df_final.to_parquet(save_path, index=False)
        if verbose:
            print(f"Saved to: {save_path}")
    
    return df_final

        

# for testing
if __name__ == "__main__":
    df = run_data_pipeline(
        input_path="data/capstone_project_1000_data.parquet",
        save_path="data/monthly_aggregated_filled.parquet"
    )
    print("\nTest successful!")
    print(f"Final data shape: {df.shape}")
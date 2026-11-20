"""
Step 1: Download official Sicilian reservoir volume datasets (OpenDataSicilia /
Autorita di Bacino della Sicilia) and inspect completeness & dynamic range
to select the best benchmark reservoirs for Sentinel-1 SAR + Sentinel-2 fusion.
"""

from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "sicilia_dighe_volumi_giornalieri.csv": (
        "https://raw.githubusercontent.com/opendatasicilia/emergenza-idrica-sicilia/"
        "main/risorse/sicilia_dighe_volumi_giornalieri.csv"
    ),
    "sicilia_dighe_volumi.csv": (
        "https://raw.githubusercontent.com/opendatasicilia/emergenza-idrica-sicilia/"
        "main/risorse/sicilia_dighe_volumi.csv"
    ),
    "sicilia_dighe_anagrafica.csv": (
        "https://raw.githubusercontent.com/opendatasicilia/emergenza-idrica-sicilia/"
        "main/risorse/sicilia_dighe_anagrafica.csv"
    ),
}


def download_datasets() -> None:
    for fname, url in URLS.items():
        dest = RAW_DIR / fname
        if not dest.exists():
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, dest)
        else:
            print(f"Already exists: {dest}")


def inspect_reservoirs() -> None:
    daily = pd.read_csv(RAW_DIR / "sicilia_dighe_volumi_giornalieri.csv")
    monthly = pd.read_csv(RAW_DIR / "sicilia_dighe_volumi.csv")
    meta = pd.read_csv(RAW_DIR / "sicilia_dighe_anagrafica.csv")

    daily["data"] = pd.to_datetime(daily["data"])
    daily["volume_mm3"] = pd.to_numeric(daily["volume"], errors="coerce") / 1e6
    daily["volume_utile_mm3"] = pd.to_numeric(daily["volume_utile"], errors="coerce") / 1e6
    daily["volume_aut_mm3"] = pd.to_numeric(daily["volume_autorizzato"], errors="coerce") / 1e6
    daily["quota"] = pd.to_numeric(daily["quota"], errors="coerce")

    print("\n=== DAILY DATASET OVERVIEW ===")
    print(f"Total rows: {len(daily):,}")
    print(f"Date range: {daily['data'].min().date()} -> {daily['data'].max().date()}")
    print(f"Unique dams in daily data: {daily['cod'].nunique()}")

    # Aggregate per dam
    agg = (
        daily.dropna(subset=["volume_mm3"])
        .groupby(["cod", "diga"], as_index=False)
        .agg(
            n_days=("data", "nunique"),
            start_date=("data", "min"),
            end_date=("data", "max"),
            vol_min_mm3=("volume_mm3", "min"),
            vol_max_mm3=("volume_mm3", "max"),
            vol_aut_mm3=("volume_aut_mm3", "median"),
            quota_min=("quota", "min"),
            quota_max=("quota", "max"),
        )
    )
    agg["vol_swing_mm3"] = agg["vol_max_mm3"] - agg["vol_min_mm3"]
    agg["vol_ratio_max_min"] = agg["vol_max_mm3"] / agg["vol_min_mm3"].replace(0, np.nan)
    agg["fill_min_pct"] = 100.0 * agg["vol_min_mm3"] / agg["vol_aut_mm3"]
    agg["fill_max_pct"] = 100.0 * agg["vol_max_mm3"] / agg["vol_aut_mm3"]

    # Merge coordinates & capacity from anagrafica
    meta_cols = ["cod", "lat", "long", "lago", "provincia", "capacita", "utilizzazione"]
    merged = agg.merge(meta[meta_cols], on="cod", how="left")

    # Sort by capacity / swing and completeness
    merged = merged.sort_values(["n_days", "vol_swing_mm3"], ascending=[False, False])

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 180)
    print("\n=== RESERVOIR CANDIDATES (DAILY DATA 2023-2026) ===")
    cols_show = [
        "cod", "diga", "lago", "provincia", "n_days",
        "start_date", "end_date", "vol_aut_mm3",
        "vol_min_mm3", "vol_max_mm3", "vol_swing_mm3",
        "fill_min_pct", "fill_max_pct", "lat", "long"
    ]
    print(merged[cols_show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Check yearly distribution of daily observations for top 10 dams by vol_swing_mm3
    top10 = merged.sort_values("vol_swing_mm3", ascending=False).head(10)["cod"].tolist()
    print("\n=== YEARLY OBSERVATION COUNTS (TOP 10 BY VOLUME SWING) ===")
    daily["year"] = daily["data"].dt.year
    piv = (
        daily[daily["cod"].isin(top10)]
        .pivot_table(index=["cod", "diga"], columns="year", values="volume_mm3", aggfunc="count", fill_value=0)
    )
    print(piv.to_string())

    # Also check monthly history length
    monthly["data"] = pd.to_datetime(monthly["data"])
    m_agg = monthly.groupby("cod", as_index=False).agg(
        n_months=("data", "nunique"),
        m_start=("data", "min"),
        m_end=("data", "max"),
    )
    print("\n=== MONTHLY ARCHIVE LENGTH (TOP 10) ===")
    print(merged[merged["cod"].isin(top10)][["cod", "diga"]].merge(m_agg, on="cod").to_string(index=False))


if __name__ == "__main__":
    download_datasets()
    inspect_reservoirs()

"""
Step 4: End-to-End Multi-Modal Satellite Water Reservoir Experiment
-------------------------------------------------------------------
Strict out-of-time evaluation (Train: 2022-2024 | Test: Jan 2025 - Feb 2026):
1. Build basin envelope masks STRICTLY from 2022-2024 clear S2 passes.
2. Quantify Sentinel-2 cloud blindness over each reservoir in 2025-2026.
3. Extract 11 physics-informed SAR features (VV, VH, VH-VV, local means/stds,
   orbit direction — NO leaky x/y coordinates).
4. Train cross-modal self-supervised LightGBM SAR water detectors using S2 MNDWI
   pseudo-labels (zero manual labeling!):
   - S1-ML Pooled (trained on all 4 reservoirs, 2022-2024)
   - S1-ML LORO   (Leave-One-Reservoir-Out spatial generalization, 2022-2024)
5. Evaluate pixel-level IoU/F1 on held-out 2025-2026 clear S1-S2 pairs against
   classic SAR Otsu thresholding.
6. Fit hypsometric Area -> Volume curves strictly on 2022-2024 monthly official
   volumes, then evaluate predicted volumes (Mm^3) against 385 DAILY official
   dam readings in 2025-2026 for:
   - S2-Optical only (MNDWI)
   - S1-Otsu (classic SAR threshold)
   - S1-ML Pooled (self-supervised LightGBM)
   - S1-ML LORO (unseen-reservoir LightGBM)
   - Multi-Modal Fusion (S2 + S1-ML with causal smoothing)
"""

from __future__ import annotations

from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, minimum_filter, binary_dilation, label
from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu
from sklearn.metrics import f1_score, jaccard_score, r2_score, mean_absolute_error, root_mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore", category=RuntimeWarning)

CUBE_DIR = Path("data/cubes")
RAW_DIR = Path("data/raw")
OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESERVOIRS = {
    "rosamarina": {"cod": "dig-22", "name": "Lago Rosamarina", "cap_mm3": 73.03},
    "poma":       {"cod": "dig-18", "name": "Lago Poma",       "cap_mm3": 72.30},
    "castello":   {"cod": "dig-03", "name": "Lago Castello",   "cap_mm3": 19.49},
    "fanaco":     {"cod": "dig-07", "name": "Lago Fanaco",     "cap_mm3": 20.70},
}

TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
RES_M = 20.0
PIXEL_AREA_KM2 = (RES_M * RES_M) / 1e6  # 0.0004 km^2 per 20m pixel

FEATURE_NAMES = [
    "vv_db",
    "vh_db",
    "vh_minus_vv",
    "vv_mean_3",
    "vv_std_3",
    "vv_mean_7",
    "vv_std_7",
    "vv_min_3",
    "vh_mean_3",
    "vh_std_3",
    "orbit_asc",
]


def load_ground_truth():
    monthly = pd.read_csv(RAW_DIR / "sicilia_dighe_volumi.csv")
    monthly["date"] = pd.to_datetime(monthly["data"])
    monthly["volume_mm3"] = pd.to_numeric(monthly["volume"], errors="coerce")

    daily = pd.read_csv(RAW_DIR / "sicilia_dighe_volumi_giornalieri.csv")
    daily["date"] = pd.to_datetime(daily["data"])
    daily["volume_mm3"] = pd.to_numeric(daily["volume"], errors="coerce") / 1e6
    daily = daily.dropna(subset=["volume_mm3"]).sort_values("date")
    return monthly, daily


def load_cube(key: str):
    s2_npz = np.load(CUBE_DIR / f"{key}_s2.npz")
    s2_meta = pd.read_parquet(CUBE_DIR / f"{key}_s2_meta.parquet")
    s2_meta["dt"] = pd.to_datetime(s2_meta["date"])

    s1_npz = np.load(CUBE_DIR / f"{key}_s1.npz")
    s1_meta = pd.read_parquet(CUBE_DIR / f"{key}_s1_meta.parquet")
    s1_meta["dt"] = pd.to_datetime(s1_meta["date"])

    return {
        "s2_mndwi": s2_npz["mndwi"].astype(np.float32),
        "s2_scl": s2_npz["scl"],
        "s2_meta": s2_meta,
        "s1_vv": s1_npz["vv_db"].astype(np.float32),
        "s1_vh": s1_npz["vh_db"].astype(np.float32),
        "s1_meta": s1_meta,
        "utm_bbox": s2_npz["utm_bbox"],
    }


def keep_largest_components(mask: np.ndarray, min_pixels: int = 25, keep_ratio: float = 0.05) -> np.ndarray:
    """Keep the main reservoir body (largest component + any major arm >= keep_ratio of largest)."""
    labeled, n = label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())[1:]
    max_c = counts.max()
    if max_c < min_pixels:
        return np.zeros_like(mask, dtype=bool)
    keep_ids = np.where((counts >= min_pixels) & (counts >= max_c * keep_ratio))[0] + 1
    return np.isin(labeled, keep_ids)


def build_basin_envelope(cube: dict) -> np.ndarray:
    """
    Build the reservoir basin envelope STRICTLY from 2022-2024 clear S2 scenes.
    Union of all water pixels -> largest connected component -> 400 m dilation (20 px)
    so higher future 2025-2026 water levels fit comfortably inside the ROI.
    """
    meta = cube["s2_meta"]
    mndwi = cube["s2_mndwi"]
    scl = cube["s2_scl"]

    train_idx = meta.index[(meta["date"] <= TRAIN_END) & (meta["roi_cloud_frac"] < 0.08)].tolist()
    union_water = np.zeros(mndwi.shape[1:], dtype=bool)
    for i in train_idx:
        w = (mndwi[i] > 0.05) & (scl[i] == 6)
        union_water |= w

    # Keep strictly the single largest connected water body (the reservoir itself)
    labeled, n = label(union_water)
    if n == 0:
        raise RuntimeError("No water body found in training scenes!")
    counts = np.bincount(labeled.ravel())[1:]
    main_id = int(np.argmax(counts)) + 1
    main_lake = labeled == main_id

    # Dilate by 20 pixels (400 m) to create a generous basin envelope for future high-water expansion
    envelope = binary_dilation(main_lake, iterations=20)
    return envelope


def extract_sar_features(vv_db: np.ndarray, vh_db: np.ndarray, orbit_state: str, mask: np.ndarray) -> np.ndarray:
    """Compute 11 physics-informed SAR features for pixels inside `mask`."""
    # Replace NaNs with median before spatial filtering
    vv = np.where(np.isfinite(vv_db), vv_db, np.nanmedian(vv_db))
    vh = np.where(np.isfinite(vh_db), vh_db, np.nanmedian(vh_db))

    vv_m3 = uniform_filter(vv, size=3, mode="nearest")
    vv_s3 = np.sqrt(np.maximum(uniform_filter((vv - vv_m3) ** 2, size=3, mode="nearest"), 0.0))
    vv_m7 = uniform_filter(vv, size=7, mode="nearest")
    vv_s7 = np.sqrt(np.maximum(uniform_filter((vv - vv_m7) ** 2, size=7, mode="nearest"), 0.0))
    vv_min3 = minimum_filter(vv, size=3, mode="nearest")

    vh_m3 = uniform_filter(vh, size=3, mode="nearest")
    vh_s3 = np.sqrt(np.maximum(uniform_filter((vh - vh_m3) ** 2, size=3, mode="nearest"), 0.0))

    orb_val = 1.0 if str(orbit_state).lower().startswith("asc") else 0.0

    feats = np.stack(
        [
            vv[mask],
            vh[mask],
            (vh - vv)[mask],
            vv_m3[mask],
            vv_s3[mask],
            vv_m7[mask],
            vv_s7[mask],
            vv_min3[mask],
            vh_m3[mask],
            vh_s3[mask],
            np.full(int(mask.sum()), orb_val, dtype=np.float32),
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def s1_otsu_mask(vv_db: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """Classic unsupervised SAR water detector: 3x3 smoothed VV Otsu threshold inside basin envelope."""
    vv = np.where(np.isfinite(vv_db), vv_db, np.nanmedian(vv_db))
    vv_smooth = uniform_filter(vv, size=3, mode="nearest")
    vals = vv_smooth[envelope]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        return np.zeros_like(envelope, dtype=bool)
    try:
        thr = float(threshold_otsu(vals))
    except Exception:
        thr = -15.0
    # Guard against degenerate Otsu when reservoir is nearly empty
    thr = float(np.clip(thr, -20.0, -12.0))
    raw = (vv_smooth < thr) & envelope
    return keep_largest_components(raw, min_pixels=20, keep_ratio=0.05)


def s2_water_mask(mndwi: np.ndarray, scl: np.ndarray, envelope: np.ndarray) -> tuple[np.ndarray, float]:
    """Optical water mask (MNDWI > 0.0 & non-cloudy) + basin cloud fraction."""
    cloudy = np.isin(scl, [0, 1, 3, 8, 9, 10])
    basin_cloud_frac = float(cloudy[envelope].mean())
    raw = (mndwi > 0.0) & (~cloudy) & envelope
    clean = keep_largest_components(raw, min_pixels=20, keep_ratio=0.05)
    return clean, basin_cloud_frac


def fit_hypsometric(areas: np.ndarray, volumes: np.ndarray):
    """
    Fit a physically constrained zero-intercept power-law hypsometric curve:
        V(A) = c * A^gamma
    Zero surface area must correspond to zero stored volume (V(0) = 0), which
    prevents unphysical positive intercepts during drought extrapolation.
    """
    valid = np.isfinite(areas) & np.isfinite(volumes) & (areas > 0.02) & (volumes > 0.05)
    a_v, v_v = areas[valid], volumes[valid]

    def power_law(x, c, gamma):
        return c * np.power(np.maximum(x, 1e-4), gamma)

    popt, _ = curve_fit(
        power_law,
        a_v,
        v_v,
        p0=[float(np.median(v_v) / np.maximum(np.median(a_v), 0.1)), 1.5],
        bounds=([0.1, 0.5], [200.0, 3.5]),
        maxfev=10000,
    )
    return lambda x: power_law(np.asarray(x, dtype=float), *popt), [float(popt[0]), float(popt[1])]


def fuse_volume_series(df_s2: pd.DataFrame, df_s1: pd.DataFrame, s1_vol_col: str) -> pd.DataFrame:
    """
    Zero-lag multi-modal volume fusion: combines calibrated clear Sentinel-2
    volume observations (priority 2 on same-day coincidence) with all-weather
    calibrated Sentinel-1 SAR observations (priority 1). Strictly causal.
    """
    ev_s2 = df_s2[df_s2["is_clear"]][["dt", "vol_s2"]].rename(columns={"vol_s2": "vol_fused"}).copy()
    ev_s2["sensor"] = "S2"
    ev_s2["prio"] = 2

    ev_s1 = df_s1[["dt", s1_vol_col]].rename(columns={s1_vol_col: "vol_fused"}).copy()
    ev_s1["sensor"] = "S1"
    ev_s1["prio"] = 1

    events = pd.concat([ev_s2, ev_s1], ignore_index=True).sort_values(["dt", "prio"])
    events = events.groupby("dt", as_index=False).last()
    events["vol_fused_med3"] = events["vol_fused"].rolling(3, min_periods=1).median()
    return events


def main() -> None:
    t0 = time.perf_counter()
    monthly_gt, daily_gt = load_ground_truth()

    cubes = {}
    envelopes = {}
    train_data_by_res = {}
    rng = np.random.default_rng(2026)

    print("=== STEP 1: Building Basin Envelopes & Cross-Modal Self-Supervised Training Sets (2022-2024) ===")
    for key, info in RESERVOIRS.items():
        cube = load_cube(key)
        env = build_basin_envelope(cube)
        cubes[key] = cube
        envelopes[key] = env
        print(f"  {info['name']}: envelope pixels={int(env.sum()):,} ({env.sum()*PIXEL_AREA_KM2:.2f} km^2)")

        # Collect paired S1-S2 training samples strictly in 2022-2024
        s2_meta = cube["s2_meta"]
        s1_meta = cube["s1_meta"]
        s1_train = s1_meta[s1_meta["date"] <= TRAIN_END]

        X_list, y_list = [], []
        n_pairs = 0
        for i2, row2 in s2_meta[s2_meta["date"] <= TRAIN_END].iterrows():
            w_mask, b_cloud = s2_water_mask(cube["s2_mndwi"][i2], cube["s2_scl"][i2], env)
            if b_cloud > 0.02 or w_mask.sum() < 50:
                continue
            # Find closest S1 pass within +-4 days
            diffs = (s1_train["dt"] - row2["dt"]).abs().dt.days
            if len(diffs) == 0 or diffs.min() > 4:
                continue
            i1 = int(diffs.idxmin())

            # High-confidence pseudo-labels (exclude ambiguous MNDWI in [-0.1, 0.0])
            mndwi_i = cube["s2_mndwi"][i2]
            scl_i = cube["s2_scl"][i2]
            cloudy_i = np.isin(scl_i, [0, 1, 3, 8, 9, 10])
            pos = w_mask
            neg = (mndwi_i < -0.10) & (~cloudy_i) & env & (~pos)
            usable = pos | neg
            if pos.sum() < 30 or neg.sum() < 30:
                continue

            feats_all = extract_sar_features(
                cube["s1_vv"][i1], cube["s1_vh"][i1], s1_meta.loc[i1, "orbit_state"], env
            )
            usable_in_env = usable[env]
            y_in_env = pos[env].astype(np.int8)

            # Sample from the TRUE natural class distribution inside the basin envelope
            # (avoiding artificial 50/50 balancing that distorts posterior probabilities!)
            idx_usable = np.where(usable_in_env)[0]
            k_sel = min(len(idx_usable), 3000)
            sel = rng.choice(idx_usable, size=k_sel, replace=False)
            X_list.append(feats_all[sel])
            y_list.append(y_in_env[sel])
            n_pairs += 1

        X_r = np.vstack(X_list)
        y_r = np.concatenate(y_list)
        train_data_by_res[key] = (X_r, y_r)
        print(f"    -> {n_pairs} clear S1-S2 training pairs | {len(y_r):,} pixels ({y_r.mean()*100:.1f}% water)")

    # -----------------------------------------------------------------------
    # STEP 2: Train Pooled LightGBM & Leave-One-Reservoir-Out (LORO) Models
    # -----------------------------------------------------------------------
    print("\n=== STEP 2: Training Self-Supervised LightGBM SAR Detectors (CPU) ===")
    lgb_params = dict(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample_for_bin=100000,
        colsample_bytree=0.85,
        subsample=0.85,
        random_state=2026,
        n_jobs=-1,
        verbose=-1,
    )

    X_all = np.vstack([train_data_by_res[k][0] for k in RESERVOIRS])
    y_all = np.concatenate([train_data_by_res[k][1] for k in RESERVOIRS])
    t_tr = time.perf_counter()
    model_pooled = lgb.LGBMClassifier(**lgb_params)
    model_pooled.fit(X_all, y_all)
    print(f"  Pooled LightGBM trained on {len(y_all):,} pixels in {time.perf_counter()-t_tr:.2f}s")

    # Feature importance
    fi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": model_pooled.feature_importances_})
    fi["importance_pct"] = 100.0 * fi["importance"] / fi["importance"].sum()
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print("  Top 5 SAR features:")
    for r in fi.head(5).itertuples(index=False):
        print(f"    - {r.feature:14s}: {r.importance_pct:5.1f}%")

    models_loro = {}
    for holdout_key in RESERVOIRS:
        X_lo = np.vstack([train_data_by_res[k][0] for k in RESERVOIRS if k != holdout_key])
        y_lo = np.concatenate([train_data_by_res[k][1] for k in RESERVOIRS if k != holdout_key])
        m_lo = lgb.LGBMClassifier(**lgb_params)
        m_lo.fit(X_lo, y_lo)
        models_loro[holdout_key] = m_lo
    print("  All 4 Leave-One-Reservoir-Out (LORO) models trained.")

    # -----------------------------------------------------------------------
    # STEP 3: Full Inference (2022-2026), Pixel IoU/F1, Hypsometry, & Daily Eval
    # -----------------------------------------------------------------------
    print("\n=== STEP 3: Evaluating on Held-Out 2025-2026 Daily Ground Truth ===")
    pixel_metrics_rows = []
    volume_summary_rows = []
    cloud_summary_rows = []
    all_daily_series = []

    for key, info in RESERVOIRS.items():
        cube = cubes[key]
        env = envelopes[key]
        cod = info["cod"]
        cap = info["cap_mm3"]

        # 3a. Extract S2 water areas across all dates
        s2_rows = []
        s2_masks = {}
        for i2, row2 in cube["s2_meta"].iterrows():
            w_mask, b_cloud = s2_water_mask(cube["s2_mndwi"][i2], cube["s2_scl"][i2], env)
            is_clear = b_cloud < 0.05
            area_km2 = float(w_mask.sum() * PIXEL_AREA_KM2) if is_clear else np.nan
            s2_rows.append({
                "date": row2["date"],
                "dt": row2["dt"],
                "basin_cloud_frac": b_cloud,
                "is_clear": is_clear,
                "area_km2": area_km2,
            })
            if is_clear:
                s2_masks[i2] = w_mask
        df_s2 = pd.DataFrame(s2_rows)

        # Quantify S2 cloud blindness in 2025-2026
        s2_test = df_s2[df_s2["date"] >= TEST_START].copy()
        n_s2_total = len(s2_test)
        n_s2_cloudy = int((~s2_test["is_clear"]).sum())
        cloudy_pct = 100.0 * n_s2_cloudy / max(n_s2_total, 1)

        # Longest cloud-blind streak in days (between consecutive clear S2 passes in 2025-2026)
        clear_dates = s2_test[s2_test["is_clear"]]["dt"].sort_values().tolist()
        if len(clear_dates) >= 2:
            gaps = [(clear_dates[j] - clear_dates[j - 1]).days for j in range(1, len(clear_dates))]
            max_gap_idx = int(np.argmax(gaps))
            max_s2_gap_days = int(gaps[max_gap_idx])
            max_s2_gap_start = clear_dates[max_gap_idx].date().isoformat()
            max_s2_gap_end = clear_dates[max_gap_idx + 1].date().isoformat()
        else:
            max_s2_gap_days, max_s2_gap_start, max_s2_gap_end = 999, "", ""

        # Winter cloudiness (Nov-Mar inside 2025-2026)
        s2_winter = s2_test[s2_test["dt"].dt.month.isin([11, 12, 1, 2, 3])]
        winter_cloudy_pct = 100.0 * (~s2_winter["is_clear"]).mean() if len(s2_winter) else np.nan

        cloud_summary_rows.append({
            "reservoir": info["name"],
            "s2_passes_2025_26": n_s2_total,
            "s2_cloudy_passes": n_s2_cloudy,
            "s2_cloudy_pct": round(cloudy_pct, 1),
            "s2_winter_cloudy_pct": round(winter_cloudy_pct, 1),
            "max_s2_blind_gap_days": max_s2_gap_days,
            "max_gap_window": f"{max_s2_gap_start} -> {max_s2_gap_end}",
        })

        # 3b. Extract S1 areas for Otsu, Pooled LightGBM, and LORO LightGBM
        s1_rows = []
        s1_masks_otsu = {}
        s1_masks_ml = {}
        s1_masks_loro = {}

        for i1, row1 in cube["s1_meta"].iterrows():
            vv = cube["s1_vv"][i1]
            vh = cube["s1_vh"][i1]
            orb = row1["orbit_state"]

            m_otsu = s1_otsu_mask(vv, env)
            feats = extract_sar_features(vv, vh, orb, env)

            p_pool = model_pooled.predict_proba(feats)[:, 1]
            m_pool_raw = np.zeros_like(env, dtype=bool)
            m_pool_raw[env] = p_pool >= 0.50
            m_pool = keep_largest_components(m_pool_raw, min_pixels=20, keep_ratio=0.05)

            p_loro = models_loro[key].predict_proba(feats)[:, 1]
            m_loro_raw = np.zeros_like(env, dtype=bool)
            m_loro_raw[env] = p_loro >= 0.50
            m_loro = keep_largest_components(m_loro_raw, min_pixels=20, keep_ratio=0.05)

            s1_rows.append({
                "date": row1["date"],
                "dt": row1["dt"],
                "orbit_state": orb,
                "area_s1_otsu": float(m_otsu.sum() * PIXEL_AREA_KM2),
                "area_s1_ml": float(m_pool.sum() * PIXEL_AREA_KM2),
                "area_s1_loro": float(m_loro.sum() * PIXEL_AREA_KM2),
            })
            if row1["date"] >= TEST_START:
                s1_masks_otsu[i1] = m_otsu
                s1_masks_ml[i1] = m_pool
                s1_masks_loro[i1] = m_loro

        df_s1 = pd.DataFrame(s1_rows)
        # Deduplicate same-date S1 passes (if asc+desc on same day) by averaging
        df_s1_daily = df_s1.groupby("dt", as_index=False)[["area_s1_otsu", "area_s1_ml", "area_s1_loro"]].mean()
        df_s1_daily["date"] = df_s1_daily["dt"].dt.strftime("%Y-%m-%d")

        # 3c. Pixel-level IoU & F1 on held-out 2025-2026 clear S1-S2 pairs (+-2 days)
        s1_test_meta = cube["s1_meta"][cube["s1_meta"]["date"] >= TEST_START]
        for i2, row2 in cube["s2_meta"][cube["s2_meta"]["date"] >= TEST_START].iterrows():
            if i2 not in s2_masks:
                continue
            diffs = (s1_test_meta["dt"] - row2["dt"]).abs().dt.days
            if len(diffs) == 0 or diffs.min() > 2:
                continue
            i1 = int(diffs.idxmin())
            y_true = s2_masks[i2][env].astype(np.int8)
            if y_true.sum() < 20:
                continue
            for method_name, m_dict in [
                ("S1-Otsu", s1_masks_otsu),
                ("S1-ML (Pooled)", s1_masks_ml),
                ("S1-ML (LORO Unseen)", s1_masks_loro),
            ]:
                y_pred = m_dict[i1][env].astype(np.int8)
                pixel_metrics_rows.append({
                    "reservoir": info["name"],
                    "date": row2["date"],
                    "method": method_name,
                    "iou": float(jaccard_score(y_true, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                })

        # 3d. Hypsometric Area -> Volume Calibration STRICTLY on 2022-2024 Monthly Official Data
        m_gt = monthly_gt[(monthly_gt["cod"] == cod) & (monthly_gt["date"] >= "2022-01-01") & (monthly_gt["date"] <= TRAIN_END)].sort_values("date")

        s2_train_clear = df_s2[(df_s2["date"] <= TRAIN_END) & (df_s2["is_clear"])].sort_values("dt")
        cal_s2 = pd.merge_asof(m_gt[["date", "volume_mm3"]], s2_train_clear[["dt", "area_km2"]], left_on="date", right_on="dt", direction="nearest", tolerance=pd.Timedelta("12D")).dropna()
        hyp_s2_fn, hyp_s2_params = fit_hypsometric(cal_s2["area_km2"].values, cal_s2["volume_mm3"].values)

        s1_train_cal = df_s1_daily[df_s1_daily["date"] <= TRAIN_END].sort_values("dt")
        cal_otsu = pd.merge_asof(m_gt[["date", "volume_mm3"]], s1_train_cal[["dt", "area_s1_otsu"]], left_on="date", right_on="dt", direction="nearest", tolerance=pd.Timedelta("12D")).dropna()
        hyp_otsu_fn, _ = fit_hypsometric(cal_otsu["area_s1_otsu"].values, cal_otsu["volume_mm3"].values)

        cal_s1ml = pd.merge_asof(m_gt[["date", "volume_mm3"]], s1_train_cal[["dt", "area_s1_ml"]], left_on="date", right_on="dt", direction="nearest", tolerance=pd.Timedelta("12D")).dropna()
        hyp_s1ml_fn, _ = fit_hypsometric(cal_s1ml["area_s1_ml"].values, cal_s1ml["volume_mm3"].values)

        cal_loro = pd.merge_asof(m_gt[["date", "volume_mm3"]], s1_train_cal[["dt", "area_s1_loro"]], left_on="date", right_on="dt", direction="nearest", tolerance=pd.Timedelta("12D")).dropna()
        hyp_loro_fn, _ = fit_hypsometric(cal_loro["area_s1_loro"].values, cal_loro["volume_mm3"].values)

        # Compute calibrated pass-level volumes
        df_s2["vol_s2"] = hyp_s2_fn(df_s2["area_km2"].values)
        df_s1_daily["vol_s1_otsu"] = hyp_otsu_fn(df_s1_daily["area_s1_otsu"].values)
        df_s1_daily["vol_s1_ml"] = hyp_s1ml_fn(df_s1_daily["area_s1_ml"].values)
        df_s1_daily["vol_s1_loro"] = hyp_loro_fn(df_s1_daily["area_s1_loro"].values)

        # 3e. Zero-Lag Multi-Modal Fusion in Calibrated Volume Space
        df_fused_otsu = fuse_volume_series(df_s2, df_s1_daily, "vol_s1_otsu")
        df_fused_ml = fuse_volume_series(df_s2, df_s1_daily, "vol_s1_ml")

        # 3f. Evaluate against 2025-2026 DAILY official reservoir volumes!
        d_gt = daily_gt[(daily_gt["cod"] == cod) & (daily_gt["date"] >= TEST_START)][["date", "volume_mm3"]].sort_values("date").copy()
        cal_days = pd.DataFrame({"date": pd.date_range(TEST_START, d_gt["date"].max(), freq="D")})

        # S2 Optical daily (forward-fill latest clear S2 pass)
        s2_ev = df_s2[df_s2["is_clear"]][["dt", "vol_s2"]].copy()
        s2_ev["s2_obs_dt"] = s2_ev["dt"]
        daily_eval = pd.merge_asof(cal_days, s2_ev[["dt", "vol_s2", "s2_obs_dt"]], left_on="date", right_on="dt", direction="backward").drop(columns=["dt"])
        daily_eval["s2_age_days"] = (daily_eval["date"] - daily_eval["s2_obs_dt"]).dt.days

        # S1 Otsu, S1-ML Pooled, S1-ML LORO daily (forward-fill latest S1 pass)
        s1_ev = df_s1_daily[["dt", "vol_s1_otsu", "vol_s1_ml", "vol_s1_loro"]].copy()
        s1_ev["s1_obs_dt"] = s1_ev["dt"]
        daily_eval = pd.merge_asof(daily_eval, s1_ev[["dt", "vol_s1_otsu", "vol_s1_ml", "vol_s1_loro", "s1_obs_dt"]], left_on="date", right_on="dt", direction="backward").drop(columns=["dt"])
        daily_eval["s1_age_days"] = (daily_eval["date"] - daily_eval["s1_obs_dt"]).dt.days

        # Multi-Modal Fusion daily (S2 + S1-Otsu and S2 + S1-ML)
        fu_o = df_fused_otsu[["dt", "vol_fused"]].rename(columns={"vol_fused": "vol_fused_otsu"}).copy()
        fu_o["fused_obs_dt"] = fu_o["dt"]
        daily_eval = pd.merge_asof(daily_eval, fu_o[["dt", "vol_fused_otsu", "fused_obs_dt"]], left_on="date", right_on="dt", direction="backward").drop(columns=["dt"])
        daily_eval["fused_age_days"] = (daily_eval["date"] - daily_eval["fused_obs_dt"]).dt.days

        fu_m = df_fused_ml[["dt", "vol_fused", "vol_fused_med3"]].rename(columns={"vol_fused": "vol_fused_ml", "vol_fused_med3": "vol_fused_ml_med3"}).copy()
        daily_eval = pd.merge_asof(daily_eval, fu_m[["dt", "vol_fused_ml", "vol_fused_ml_med3"]], left_on="date", right_on="dt", direction="backward").drop(columns=["dt"])

        # Join official daily ground truth
        eval_gt = d_gt.merge(daily_eval, on="date", how="inner")
        eval_gt["reservoir"] = info["name"]
        eval_gt["key"] = key
        all_daily_series.append(eval_gt)

        for label_m, col_v, col_age in [
            ("S2-Optical Only", "vol_s2", "s2_age_days"),
            ("S1-Otsu (Classic SAR)", "vol_s1_otsu", "s1_age_days"),
            ("S1-ML (Pooled)", "vol_s1_ml", "s1_age_days"),
            ("S1-ML (LORO Unseen)", "vol_s1_loro", "s1_age_days"),
            ("Fusion (S2 + S1-Otsu)", "vol_fused_otsu", "fused_age_days"),
            ("Fusion (S2 + S1-ML)", "vol_fused_ml", "fused_age_days"),
        ]:
            sub = eval_gt.dropna(subset=["volume_mm3", col_v])
            yt, yp = sub["volume_mm3"].values, sub[col_v].values
            r2 = float(r2_score(yt, yp))
            mae = float(mean_absolute_error(yt, yp))
            rmse = float(root_mean_squared_error(yt, yp))
            nrmse_pct = 100.0 * rmse / cap
            fresh_7d_pct = 100.0 * float((sub[col_age] <= 7).mean())
            max_age = int(sub[col_age].max())

            volume_summary_rows.append({
                "reservoir": info["name"],
                "method": label_m,
                "n_days": len(sub),
                "r2": round(r2, 4),
                "mae_mm3": round(mae, 3),
                "rmse_mm3": round(rmse, 3),
                "nrmse_cap_pct": round(nrmse_pct, 2),
                "fresh_le_7d_pct": round(fresh_7d_pct, 1),
                "max_stale_days": max_age,
            })

        # Save per-reservoir pass-level tables & hypsometry params for plotting
        df_s2.to_csv(OUT_DIR / f"{key}_s2_passes.csv", index=False)
        df_s1_daily.to_csv(OUT_DIR / f"{key}_s1_passes.csv", index=False)
        df_fused_otsu.to_csv(OUT_DIR / f"{key}_fused_otsu_passes.csv", index=False)
        df_fused_ml.to_csv(OUT_DIR / f"{key}_fused_ml_passes.csv", index=False)
        with open(OUT_DIR / f"{key}_hypsometry.json", "w", encoding="utf-8") as f:
            json.dump({"c": hyp_s2_params[0], "gamma": hyp_s2_params[1]}, f, indent=2)

    # Save summary tables
    df_cloud = pd.DataFrame(cloud_summary_rows)
    df_pix = pd.DataFrame(pixel_metrics_rows)
    df_vol = pd.DataFrame(volume_summary_rows)
    df_daily_all = pd.concat(all_daily_series, ignore_index=True)

    df_cloud.to_csv(OUT_DIR / "cloud_blindness_summary.csv", index=False)
    df_pix.to_csv(OUT_DIR / "pixel_iou_f1_2025_26.csv", index=False)
    df_vol.to_csv(OUT_DIR / "volume_evaluation_2025_26.csv", index=False)
    df_daily_all.to_parquet(OUT_DIR / "daily_tracking_2025_26.parquet", index=False)

    # Print crisp executive summaries
    print("\n================ CLOUD BLINDNESS (SENTINEL-2, 2025-2026) ================")
    print(df_cloud.to_string(index=False))

    print("\n================ PIXEL-LEVEL WATER MASKING (HELD-OUT 2025-2026 PAIRS) ================")
    pix_agg = df_pix.groupby(["method"], as_index=False).agg(
        n_pairs=("iou", "count"),
        mean_iou=("iou", "mean"),
        median_iou=("iou", "median"),
        mean_f1=("f1", "mean"),
    )
    print(pix_agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n================ OPERATIONAL DAILY VOLUME TRACKING (2025-2026, ALL 4 RESERVOIRS) ================")
    print(df_vol.to_string(index=False))

    print("\n================ AGGREGATE ACROSS ALL 4 RESERVOIRS ================")
    agg_vol = df_vol.groupby("method", as_index=False).agg(
        mean_r2=("r2", "mean"),
        mean_mae_mm3=("mae_mm3", "mean"),
        mean_rmse_mm3=("rmse_mm3", "mean"),
        mean_nrmse_cap_pct=("nrmse_cap_pct", "mean"),
        mean_fresh_7d_pct=("fresh_le_7d_pct", "mean"),
        worst_stale_days=("max_stale_days", "max"),
    ).sort_values("mean_nrmse_cap_pct")
    print(agg_vol.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nExperiment finished in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()

"""
Scientific Benchmark for Environmental Data Science (Cambridge University Press)
--------------------------------------------------------------------------------
Rigorous out-of-time & Leave-One-Reservoir-Out (LORO) evaluation of multi-modal
Sentinel-1 SAR and Sentinel-2 optical water reservoir monitoring across four
Sicilian reservoirs (2022-2024 train/calibration | Jan 2025 - Feb 2026 daily test).

Evaluates 10 configurations:
  Optical:
    1. S2-MNDWI (Xu, 2006)
  Unsupervised SAR baselines:
    2. S1-Otsu (Otsu, 1979) on 3x3 smoothed VV
    3. S1-GMM  (2-component Gaussian Mixture Model on dual-pol [VV, VH] per scene)
  Cross-modal self-supervised ML (zero manual labels):
    4. S1-RF (Raw dB)                 - Random Forest on 11 raw SAR features
    5. S1-LGBM (Raw dB)               - LightGBM on 11 raw SAR features (ablation)
    6. S1-LGBM (Scene-Norm)           - LightGBM with per-scene robust normalization (proposed)
    7. S1-LGBM (Scene-Norm LORO)      - Leave-One-Reservoir-Out spatial generalization
  Multi-modal zero-lag volume fusion (strictly causal):
    8. Fusion (S2 + S1-Otsu)
    9. Fusion (S2 + S1-GMM)
   10. Fusion (S2 + S1-LGBM Scene-Norm)

Computes:
  - Pixel-level IoU, F1, Precision, Recall on 167 held-out 2025-2026 S1-S2 pairs
  - Daily volume R^2, MAE, RMSE, NRMSE (% cap), Bias, Freshness <=7d, Max Stale Days
  - 95% non-parametric bootstrap CIs (B = 2,000) and Wilcoxon signed-rank p-values
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
from scipy.stats import wilcoxon
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)
import lightgbm as lgb

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO_DIR = Path(__file__).resolve().parents[1]
CUBE_DIR = REPO_DIR / "data" / "cubes"
RAW_DIR = REPO_DIR / "data" / "raw"
OUT_DIR = REPO_DIR / "results" / "eds_benchmark"
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
PIXEL_AREA_KM2 = (RES_M * RES_M) / 1e6

RAW_FEATURE_NAMES = [
    "vv_db", "vh_db", "vh_minus_vv",
    "vv_mean_3", "vv_std_3", "vv_mean_7", "vv_std_7", "vv_min_3",
    "vh_mean_3", "vh_std_3", "orbit_asc",
]

NORM_EXTRA_NAMES = [
    "vv_z_scene",
    "vh_z_scene",
    "vv_m3_z_scene",
    "vh_m3_z_scene",
    "vv_minus_otsu",
]

ALL_FEATURE_NAMES = RAW_FEATURE_NAMES + NORM_EXTRA_NAMES


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
    }


def keep_largest_components(mask: np.ndarray, min_pixels: int = 20, keep_ratio: float = 0.05) -> np.ndarray:
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
    """Strictly from 2022-2024 clear S2 passes (zero test leakage)."""
    meta = cube["s2_meta"]
    mndwi = cube["s2_mndwi"]
    scl = cube["s2_scl"]
    train_idx = meta.index[(meta["date"] <= TRAIN_END) & (meta["roi_cloud_frac"] < 0.08)].tolist()
    union_water = np.zeros(mndwi.shape[1:], dtype=bool)
    for i in train_idx:
        union_water |= (mndwi[i] > 0.05) & (scl[i] == 6)
    labeled, n = label(union_water)
    if n == 0:
        raise RuntimeError("No water body found in training period.")
    main_id = int(np.argmax(np.bincount(labeled.ravel())[1:])) + 1
    return binary_dilation(labeled == main_id, iterations=20)


def s2_water_mask(mndwi: np.ndarray, scl: np.ndarray, envelope: np.ndarray) -> tuple[np.ndarray, float]:
    cloudy = np.isin(scl, [0, 1, 3, 8, 9, 10])
    b_cloud = float(cloudy[envelope].mean())
    raw = (mndwi > 0.0) & (~cloudy) & envelope
    return keep_largest_components(raw), b_cloud


def compute_scene_otsu(vv_m3: np.ndarray, envelope: np.ndarray) -> float:
    vals = vv_m3[envelope]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        return -15.0
    try:
        thr = float(threshold_otsu(vals))
    except Exception:
        thr = -15.0
    return float(np.clip(thr, -20.0, -12.0))


def s1_gmm_mask(vv_m3: np.ndarray, vh_m3: np.ndarray, envelope: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Unsupervised dual-pol 2-component Gaussian Mixture Model per scene."""
    X_env = np.column_stack([vv_m3[envelope], vh_m3[envelope]])
    n = len(X_env)
    sub_idx = rng.choice(n, size=min(n, 4000), replace=False)
    try:
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=2026, max_iter=100)
        gmm.fit(X_env[sub_idx])
        # Water component is the one with lower mean VV backscatter
        water_comp = int(np.argmin(gmm.means_[:, 0]))
        # Guard: ensure genuine separation (> 2.5 dB) between components
        if abs(gmm.means_[0, 0] - gmm.means_[1, 0]) < 2.5:
            return np.zeros_like(envelope, dtype=bool)
        preds = gmm.predict(X_env) == water_comp
    except Exception:
        return np.zeros_like(envelope, dtype=bool)
    raw = np.zeros_like(envelope, dtype=bool)
    raw[envelope] = preds
    return keep_largest_components(raw)


def extract_sar_features_all(vv_db: np.ndarray, vh_db: np.ndarray, orbit_state: str, envelope: np.ndarray):
    """
    Returns (feats_all [N_env, 16], vv_m3, vh_m3, otsu_thr).
    First 11 columns = RAW_FEATURE_NAMES; last 5 columns = per-scene robust normalized features.
    """
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

    # Per-scene robust statistics inside the basin envelope (median & IQR)
    vv_env = vv[envelope]
    vh_env = vh[envelope]
    vv_m3_env = vv_m3[envelope]
    vh_m3_env = vh_m3[envelope]

    med_vv = float(np.median(vv_env))
    iqr_vv = max(float(np.percentile(vv_env, 75) - np.percentile(vv_env, 25)), 1.0)
    med_vh = float(np.median(vh_env))
    iqr_vh = max(float(np.percentile(vh_env, 75) - np.percentile(vh_env, 25)), 1.0)

    otsu_thr = compute_scene_otsu(vv_m3, envelope)

    feats = np.stack(
        [
            vv_env,
            vh_env,
            (vh_env - vv_env),
            vv_m3_env,
            vv_s3[envelope],
            vv_m7[envelope],
            vv_s7[envelope],
            vv_min3[envelope],
            vh_m3_env,
            vh_s3[envelope],
            np.full(len(vv_env), orb_val, dtype=np.float32),
            # Scene-normalized physics features
            (vv_env - med_vv) / iqr_vv,
            (vh_env - med_vh) / iqr_vh,
            (vv_m3_env - med_vv) / iqr_vv,
            (vh_m3_env - med_vh) / iqr_vh,
            (vv_m3_env - otsu_thr),
        ],
        axis=1,
    ).astype(np.float32)

    return feats, vv_m3, vh_m3, otsu_thr


def fit_hypsometric(areas: np.ndarray, volumes: np.ndarray):
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
    ev_s2 = df_s2[df_s2["is_clear"]][["dt", "vol_s2"]].rename(columns={"vol_s2": "vol_fused"}).copy()
    ev_s2["prio"] = 2
    ev_s1 = df_s1[["dt", s1_vol_col]].rename(columns={s1_vol_col: "vol_fused"}).copy()
    ev_s1["prio"] = 1
    events = pd.concat([ev_s2, ev_s1], ignore_index=True).sort_values(["dt", "prio"])
    return events.groupby("dt", as_index=False).last()


def bootstrap_ci(arr: np.ndarray, stat_fn, n_boot: int = 2000, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = stat_fn(arr[idx])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> None:
    t0 = time.perf_counter()
    monthly_gt, daily_gt = load_ground_truth()
    rng = np.random.default_rng(2026)

    cubes, envelopes = {}, {}
    train_data_by_res = {}

    print("=== STEP 1: Building Basin Envelopes & 2022-2024 Self-Supervised Training Data ===")
    for key, info in RESERVOIRS.items():
        cube = load_cube(key)
        env = build_basin_envelope(cube)
        cubes[key] = cube
        envelopes[key] = env

        s2_meta = cube["s2_meta"]
        s1_meta = cube["s1_meta"]
        s1_train = s1_meta[s1_meta["date"] <= TRAIN_END]

        X_list, y_list = [], []
        n_pairs = 0
        for i2, row2 in s2_meta[s2_meta["date"] <= TRAIN_END].iterrows():
            w_mask, b_cloud = s2_water_mask(cube["s2_mndwi"][i2], cube["s2_scl"][i2], env)
            if b_cloud > 0.02 or w_mask.sum() < 50:
                continue
            diffs = (s1_train["dt"] - row2["dt"]).abs().dt.days
            if len(diffs) == 0 or diffs.min() > 4:
                continue
            i1 = int(diffs.idxmin())

            mndwi_i = cube["s2_mndwi"][i2]
            cloudy_i = np.isin(cube["s2_scl"][i2], [0, 1, 3, 8, 9, 10])
            pos = w_mask
            neg = (mndwi_i < -0.10) & (~cloudy_i) & env & (~pos)
            usable = pos | neg
            if pos.sum() < 30 or neg.sum() < 30:
                continue

            feats_all, _, _, _ = extract_sar_features_all(
                cube["s1_vv"][i1], cube["s1_vh"][i1], s1_meta.loc[i1, "orbit_state"], env
            )
            usable_in_env = usable[env]
            y_in_env = pos[env].astype(np.int8)

            idx_usable = np.where(usable_in_env)[0]
            sel = rng.choice(idx_usable, size=min(len(idx_usable), 3000), replace=False)
            X_list.append(feats_all[sel])
            y_list.append(y_in_env[sel])
            n_pairs += 1

        X_r = np.vstack(X_list)
        y_r = np.concatenate(y_list)
        train_data_by_res[key] = (X_r, y_r)
        print(f"  {info['name']}: {n_pairs} pairs | {len(y_r):,} pixels ({y_r.mean()*100:.1f}% water)")

    # -----------------------------------------------------------------------
    # STEP 2: Train Models (RF Raw, LGBM Raw, LGBM Scene-Norm, LGBM Scene-Norm LORO)
    # -----------------------------------------------------------------------
    print("\n=== STEP 2: Training Self-Supervised Models (CPU) ===")
    X_all = np.vstack([train_data_by_res[k][0] for k in RESERVOIRS])
    y_all = np.concatenate([train_data_by_res[k][1] for k in RESERVOIRS])

    lgb_params = dict(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.85,
        subsample=0.85,
        random_state=2026,
        n_jobs=-1,
        verbose=-1,
    )

    t1 = time.perf_counter()
    model_lgb_raw = lgb.LGBMClassifier(**lgb_params)
    model_lgb_raw.fit(X_all[:, :11], y_all)
    print(f"  LGBM (Raw dB) trained in {time.perf_counter()-t1:.2f}s")

    t1 = time.perf_counter()
    model_lgb_norm = lgb.LGBMClassifier(**lgb_params)
    model_lgb_norm.fit(X_all, y_all)
    print(f"  LGBM (Scene-Norm) trained in {time.perf_counter()-t1:.2f}s")

    # Feature importance for Scene-Norm LGBM
    fi = pd.DataFrame({"feature": ALL_FEATURE_NAMES, "importance": model_lgb_norm.feature_importances_})
    fi["importance_pct"] = 100.0 * fi["importance"] / fi["importance"].sum()
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(OUT_DIR / "eds_feature_importance_scene_norm.csv", index=False)
    print("  Top 6 features (Scene-Norm LGBM):")
    for r in fi.head(6).itertuples(index=False):
        print(f"    - {r.feature:16s}: {r.importance_pct:5.1f}%")

    # Random Forest baseline (subsample 150k pixels for fast reproducible CPU training)
    t1 = time.perf_counter()
    rf_idx = rng.choice(len(y_all), size=min(len(y_all), 150000), replace=False)
    model_rf_raw = RandomForestClassifier(
        n_estimators=120, max_depth=16, min_samples_leaf=10, random_state=2026, n_jobs=-1
    )
    model_rf_raw.fit(X_all[rf_idx, :11], y_all[rf_idx])
    print(f"  Random Forest (Raw dB) trained in {time.perf_counter()-t1:.2f}s")

    # Leave-One-Reservoir-Out (LORO) for Scene-Norm LGBM
    models_loro_norm = {}
    for holdout_key in RESERVOIRS:
        X_lo = np.vstack([train_data_by_res[k][0] for k in RESERVOIRS if k != holdout_key])
        y_lo = np.concatenate([train_data_by_res[k][1] for k in RESERVOIRS if k != holdout_key])
        m_lo = lgb.LGBMClassifier(**lgb_params)
        m_lo.fit(X_lo, y_lo)
        models_loro_norm[holdout_key] = m_lo
    print("  All 4 LORO Scene-Norm LGBM models trained.")

    # -----------------------------------------------------------------------
    # STEP 3: Full Inference, Pixel Metrics, Hypsometry, & Statistical Tests
    # -----------------------------------------------------------------------
    print("\n=== STEP 3: Running Out-of-Time Evaluation (Jan 2025 - Feb 2026) ===")
    pixel_rows = []
    vol_rows = []
    daily_frames = []

    for key, info in RESERVOIRS.items():
        cube = cubes[key]
        env = envelopes[key]
        cod = info["cod"]
        cap = info["cap_mm3"]

        # S2 passes
        s2_rows, s2_masks = [], {}
        for i2, row2 in cube["s2_meta"].iterrows():
            w_mask, b_cloud = s2_water_mask(cube["s2_mndwi"][i2], cube["s2_scl"][i2], env)
            is_clear = b_cloud < 0.05
            s2_rows.append({
                "date": row2["date"],
                "dt": row2["dt"],
                "basin_cloud_frac": b_cloud,
                "is_clear": is_clear,
                "area_km2": float(w_mask.sum() * PIXEL_AREA_KM2) if is_clear else np.nan,
            })
            if is_clear:
                s2_masks[i2] = w_mask
        df_s2 = pd.DataFrame(s2_rows)

        # S1 passes
        s1_rows = []
        s1_test_masks = {m: {} for m in ["S1-Otsu", "S1-GMM", "S1-RF (Raw)", "S1-LGBM (Raw)", "S1-LGBM (Scene-Norm)", "S1-LGBM (Scene-Norm LORO)"]}

        for i1, row1 in cube["s1_meta"].iterrows():
            feats, vv_m3, vh_m3, otsu_thr = extract_sar_features_all(
                cube["s1_vv"][i1], cube["s1_vh"][i1], row1["orbit_state"], env
            )

            # 1. Otsu
            m_otsu = keep_largest_components((vv_m3 < otsu_thr) & env)
            # 2. GMM
            m_gmm = s1_gmm_mask(vv_m3, vh_m3, env, rng)

            def _pred_mask(model, X_in):
                probs = model.predict_proba(X_in)[:, 1]
                raw = np.zeros_like(env, dtype=bool)
                raw[env] = probs >= 0.50
                return keep_largest_components(raw)

            m_rf = _pred_mask(model_rf_raw, feats[:, :11])
            m_lgb_raw = _pred_mask(model_lgb_raw, feats[:, :11])
            m_lgb_norm = _pred_mask(model_lgb_norm, feats)
            m_loro = _pred_mask(models_loro_norm[key], feats)

            s1_rows.append({
                "dt": row1["dt"],
                "area_otsu": float(m_otsu.sum() * PIXEL_AREA_KM2),
                "area_gmm": float(m_gmm.sum() * PIXEL_AREA_KM2),
                "area_rf": float(m_rf.sum() * PIXEL_AREA_KM2),
                "area_lgb_raw": float(m_lgb_raw.sum() * PIXEL_AREA_KM2),
                "area_lgb_norm": float(m_lgb_norm.sum() * PIXEL_AREA_KM2),
                "area_loro_norm": float(m_loro.sum() * PIXEL_AREA_KM2),
            })
            if row1["date"] >= TEST_START:
                s1_test_masks["S1-Otsu"][i1] = m_otsu
                s1_test_masks["S1-GMM"][i1] = m_gmm
                s1_test_masks["S1-RF (Raw)"][i1] = m_rf
                s1_test_masks["S1-LGBM (Raw)"][i1] = m_lgb_raw
                s1_test_masks["S1-LGBM (Scene-Norm)"][i1] = m_lgb_norm
                s1_test_masks["S1-LGBM (Scene-Norm LORO)"][i1] = m_loro

        df_s1 = pd.DataFrame(s1_rows).groupby("dt", as_index=False).mean()
        df_s1["date"] = df_s1["dt"].dt.strftime("%Y-%m-%d")

        # Pixel-level IoU, F1, Precision, Recall on held-out 2025-2026 clear S1-S2 pairs (+-2d)
        s1_test_meta = cube["s1_meta"][cube["s1_meta"]["date"] >= TEST_START]
        for i2, row2 in cube["s2_meta"][cube["s2_meta"]["date"] >= TEST_START].iterrows():
            if i2 not in s2_masks:
                continue
            diffs = (s1_test_meta["dt"] - row2["dt"]).abs().dt.days
            if len(diffs) == 0 or diffs.min() > 2:
                continue
            i1 = int(diffs.idxmin())
            yt = s2_masks[i2][env].astype(np.int8)
            if yt.sum() < 20:
                continue
            for m_name, m_dict in s1_test_masks.items():
                yp = m_dict[i1][env].astype(np.int8)
                pixel_rows.append({
                    "reservoir": info["name"],
                    "date": row2["date"],
                    "method": m_name,
                    "iou": float(jaccard_score(yt, yp, zero_division=0)),
                    "f1": float(f1_score(yt, yp, zero_division=0)),
                    "precision": float(precision_score(yt, yp, zero_division=0)),
                    "recall": float(recall_score(yt, yp, zero_division=0)),
                })

        # Hypsometric calibration strictly on 2022-2024 monthly official volumes
        m_gt = monthly_gt[(monthly_gt["cod"] == cod) & (monthly_gt["date"] >= "2022-01-01") & (monthly_gt["date"] <= TRAIN_END)].sort_values("date")
        s2_tr = df_s2[(df_s2["date"] <= TRAIN_END) & (df_s2["is_clear"])].sort_values("dt")
        c_s2 = pd.merge_asof(m_gt[["date", "volume_mm3"]], s2_tr[["dt", "area_km2"]], left_on="date", right_on="dt", direction="nearest", tolerance=pd.Timedelta("12D")).dropna()
        hyp_s2, _ = fit_hypsometric(c_s2["area_km2"].values, c_s2["volume_mm3"].values)
        df_s2["vol_s2"] = hyp_s2(df_s2["area_km2"].values)

        s1_tr = df_s1[df_s1["date"] <= TRAIN_END].sort_values("dt")
        for acol, vcol in [
            ("area_otsu", "vol_otsu"),
            ("area_gmm", "vol_gmm"),
            ("area_rf", "vol_rf"),
            ("area_lgb_raw", "vol_lgb_raw"),
            ("area_lgb_norm", "vol_lgb_norm"),
            ("area_loro_norm", "vol_loro_norm"),
        ]:
            c1 = pd.merge_asof(m_gt[["date", "volume_mm3"]], s1_tr[["dt", acol]], left_on="date", right_on="dt", direction="nearest", tolerance=pd.Timedelta("12D")).dropna()
            fn1, _ = fit_hypsometric(c1[acol].values, c1["volume_mm3"].values)
            df_s1[vcol] = fn1(df_s1[acol].values)

        # Multi-Modal Zero-Lag Fusions
        fu_otsu = fuse_volume_series(df_s2, df_s1, "vol_otsu")
        fu_gmm = fuse_volume_series(df_s2, df_s1, "vol_gmm")
        fu_norm = fuse_volume_series(df_s2, df_s1, "vol_lgb_norm")

        # Operational daily evaluation (Jan 2025 - Feb 2026)
        d_gt = daily_gt[(daily_gt["cod"] == cod) & (daily_gt["date"] >= TEST_START)][["date", "volume_mm3"]].sort_values("date").copy()
        cal_days = pd.DataFrame({"date": pd.date_range(TEST_START, d_gt["date"].max(), freq="D")})

        def _attach(base_df, src_df, col_in, col_out, age_out):
            tmp = src_df[["dt", col_in]].rename(columns={col_in: col_out}).copy()
            tmp["_obs_dt"] = tmp["dt"]
            m = pd.merge_asof(base_df, tmp.sort_values("dt"), left_on="date", right_on="dt", direction="backward").drop(columns=["dt"])
            m[age_out] = (m["date"] - m["_obs_dt"]).dt.days
            return m.drop(columns=["_obs_dt"])

        ev = _attach(cal_days, df_s2[df_s2["is_clear"]], "vol_s2", "vol_s2", "age_s2")
        for vc in ["vol_otsu", "vol_gmm", "vol_rf", "vol_lgb_raw", "vol_lgb_norm", "vol_loro_norm"]:
            ev = _attach(ev, df_s1, vc, vc, f"age_{vc}")
        ev = _attach(ev, fu_otsu, "vol_fused", "vol_fu_otsu", "age_fu")
        ev = _attach(ev, fu_gmm, "vol_fused", "vol_fu_gmm", "age_fu_gmm")
        ev = _attach(ev, fu_norm, "vol_fused", "vol_fu_norm", "age_fu_norm")

        eval_gt = d_gt.merge(ev, on="date", how="inner")
        eval_gt["reservoir"] = info["name"]
        eval_gt["key"] = key
        daily_frames.append(eval_gt)

        specs = [
            ("S2-MNDWI (Optical)", "vol_s2", "age_s2"),
            ("S1-Otsu (Classic SAR)", "vol_otsu", "age_vol_otsu"),
            ("S1-GMM (Dual-Pol)", "vol_gmm", "age_vol_gmm"),
            ("S1-RF (Raw dB)", "vol_rf", "age_vol_rf"),
            ("S1-LGBM (Raw dB)", "vol_lgb_raw", "age_vol_lgb_raw"),
            ("S1-LGBM (Scene-Norm)", "vol_lgb_norm", "age_vol_lgb_norm"),
            ("S1-LGBM (Scene-Norm LORO)", "vol_loro_norm", "age_vol_loro_norm"),
            ("Fusion (S2 + S1-Otsu)", "vol_fu_otsu", "age_fu"),
            ("Fusion (S2 + S1-GMM)", "vol_fu_gmm", "age_fu_gmm"),
            ("Fusion (S2 + S1-LGBM Norm)", "vol_fu_norm", "age_fu_norm"),
        ]

        for m_label, vcol, acol in specs:
            sub = eval_gt.dropna(subset=["volume_mm3", vcol])
            yt, yp = sub["volume_mm3"].values, sub[vcol].values
            abs_err = np.abs(yp - yt)
            mae = float(mean_absolute_error(yt, yp))
            rmse = float(root_mean_squared_error(yt, yp))
            mae_lo, mae_hi = bootstrap_ci(abs_err, lambda a: a.mean(axis=1))
            vol_rows.append({
                "reservoir": info["name"],
                "method": m_label,
                "n_days": len(sub),
                "r2": round(float(r2_score(yt, yp)), 4),
                "mae_mm3": round(mae, 3),
                "mae_ci95_lo": round(mae_lo, 3),
                "mae_ci95_hi": round(mae_hi, 3),
                "rmse_mm3": round(rmse, 3),
                "nrmse_cap_pct": round(100.0 * rmse / cap, 2),
                "bias_mm3": round(float(np.mean(yp - yt)), 3),
                "fresh_7d_pct": round(100.0 * float((sub[acol] <= 7).mean()), 1),
                "max_stale_days": int(sub[acol].max()),
            })

    df_pix = pd.DataFrame(pixel_rows)
    df_vol = pd.DataFrame(vol_rows)
    df_daily_all = pd.concat(daily_frames, ignore_index=True)

    df_pix.to_csv(OUT_DIR / "eds_pixel_metrics_2025_26.csv", index=False)
    df_vol.to_csv(OUT_DIR / "eds_volume_metrics_2025_26.csv", index=False)
    df_daily_all.to_parquet(OUT_DIR / "eds_daily_series_2025_26.parquet", index=False)

    # Statistical significance tests across all 1,533 reservoir-days (Wilcoxon signed-rank on daily |error|)
    stat_tests = []
    comparisons = [
        ("S1-LGBM (Scene-Norm) vs S1-LGBM (Raw dB)", "vol_lgb_norm", "vol_lgb_raw"),
        ("S1-LGBM (Scene-Norm) vs S1-Otsu", "vol_lgb_norm", "vol_otsu"),
        ("S1-LGBM (Scene-Norm) vs S1-RF (Raw dB)", "vol_lgb_norm", "vol_rf"),
        ("Fusion (S2 + S1-LGBM Norm) vs S2-MNDWI", "vol_fu_norm", "vol_s2"),
        ("Fusion (S2 + S1-LGBM Norm) vs Fusion (S2 + S1-Otsu)", "vol_fu_norm", "vol_fu_otsu"),
        ("Fusion (S2 + S1-Otsu) vs S2-MNDWI", "vol_fu_otsu", "vol_s2"),
    ]
    for name_cmp, cA, cB in comparisons:
        sub = df_daily_all.dropna(subset=["volume_mm3", cA, cB])
        eA = np.abs(sub[cA].values - sub["volume_mm3"].values)
        eB = np.abs(sub[cB].values - sub["volume_mm3"].values)
        stat, pval = wilcoxon(eA, eB, zero_method="wilcox")
        stat_tests.append({
            "comparison": name_cmp,
            "n_days": len(sub),
            "median_abs_err_A": round(float(np.median(eA)), 3),
            "median_abs_err_B": round(float(np.median(eB)), 3),
            "wilcoxon_stat": float(stat),
            "p_value": float(pval),
        })
    df_stats = pd.DataFrame(stat_tests)
    df_stats.to_csv(OUT_DIR / "eds_wilcoxon_tests.csv", index=False)

    print("\n================ PIXEL-LEVEL SEGMENTATION (167 HELD-OUT 2025-2026 PAIRS) ================")
    pix_agg = df_pix.groupby("method", as_index=False)[["iou", "f1", "precision", "recall"]].mean().sort_values("iou", ascending=False)
    print(pix_agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n================ PER-RESERVOIR DAILY VOLUME ACCURACY (2025-2026) ================")
    print(df_vol[["reservoir", "method", "r2", "mae_mm3", "rmse_mm3", "nrmse_cap_pct", "bias_mm3"]].to_string(index=False))

    print("\n================ AGGREGATE ACROSS ALL 4 RESERVOIRS ================")
    agg = df_vol.groupby("method", as_index=False).agg(
        mean_r2=("r2", "mean"),
        mean_mae=("mae_mm3", "mean"),
        mean_rmse=("rmse_mm3", "mean"),
        mean_nrmse_pct=("nrmse_cap_pct", "mean"),
        mean_fresh_7d=("fresh_7d_pct", "mean"),
        worst_stale=("max_stale_days", "max"),
    ).sort_values("mean_nrmse_pct")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n================ WILCOXON SIGNED-RANK TESTS (ALL 1,533 RESERVOIR-DAYS) ================")
    print(df_stats.to_string(index=False))
    print(f"\nEDS scientific benchmark completed in {time.perf_counter()-t0:.1f}s.")


if __name__ == "__main__":
    main()

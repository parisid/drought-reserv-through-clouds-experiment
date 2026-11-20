"""
Step 5: Generate 4 clean, publication-grade figures tailored to Towards Data Science
editorial standards (high DPI, crisp typography, colorblind-safe palette, zero
watermarks or cluttered overlays).
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter, binary_dilation, label
from skimage.filters import threshold_otsu

CUBE_DIR = Path("data/cubes")
RAW_DIR = Path("data/raw")
RES_DIR = Path("results")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Clean editorial style (TDS-friendly)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#E8E8E8",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

COLORS = {
    "gt": "#1A1A2E",
    "s2": "#94A3B8",
    "otsu": "#0F766E",
    "ml": "#D97706",
    "loro": "#B45309",
    "fusion": "#0284C7",
    "cloud": "#FEE2E2",
    "cloud_edge": "#EF4444",
}

RESERVOIRS = [
    ("rosamarina", "dig-22", "Lago Rosamarina", 73.03),
    ("poma",       "dig-18", "Lago Poma",       72.30),
    ("castello",   "dig-03", "Lago Castello",   19.49),
    ("fanaco",     "dig-07", "Lago Fanaco",     20.70),
]


def fig1_cloud_blindness_and_radar_vision() -> None:
    """
    2-panel hero figure:
    Left: Real winter cloud-obscuration rate & longest blind streak (days) for all 4 reservoirs.
    Right: Side-by-side real satellite snapshot of Lago Fanaco during the Feb 2026 storm recharge
           (Sentinel-2 optical blocked by clouds vs Sentinel-1 SAR VV seeing the refilled lake).
    """
    df_cloud = pd.read_csv(RES_DIR / "cloud_blindness_summary.csv")

    # Load Fanaco cube for a cloudy winter 2026 date (Feb 2026 during the 75-day blackout)
    s2_npz = np.load(CUBE_DIR / "fanaco_s2.npz")
    s2_meta = pd.read_parquet(CUBE_DIR / "fanaco_s2_meta.parquet")
    s1_npz = np.load(CUBE_DIR / "fanaco_s1.npz")
    s1_meta = pd.read_parquet(CUBE_DIR / "fanaco_s1_meta.parquet")

    # Pick a very cloudy Fanaco S2 scene in late Jan / early Feb 2026 and matching S1 scene
    s2_win = s2_meta[(s2_meta["date"] >= "2026-01-20") & (s2_meta["date"] <= "2026-02-20") & (s2_meta["roi_cloud_frac"] > 0.70)]
    i2 = int(s2_win.index[-1]) if len(s2_win) else int(s2_meta.index[-5])
    d2 = s2_meta.loc[i2, "date"]

    s1_meta["dt"] = pd.to_datetime(s1_meta["date"])
    i1 = int((s1_meta["dt"] - pd.Timestamp(d2)).abs().idxmin())
    d1 = s1_meta.loc[i1, "date"]

    # Build simple basin envelope from 2022-2024 for clean radar contour
    mndwi_all = s2_npz["mndwi"].astype(np.float32)
    scl_all = s2_npz["scl"]
    train_idx = s2_meta.index[(s2_meta["date"] <= "2024-12-31") & (s2_meta["roi_cloud_frac"] < 0.08)].tolist()
    union_w = np.zeros(mndwi_all.shape[1:], dtype=bool)
    for idx in train_idx:
        union_w |= (mndwi_all[idx] > 0.05) & (scl_all[idx] == 6)
    lab, _ = label(union_w)
    main_id = int(np.argmax(np.bincount(lab.ravel())[1:])) + 1
    env = binary_dilation(lab == main_id, iterations=20)

    # Crop tightly around the basin envelope for crisp visualization
    ys, xs = np.where(env)
    r0, r1 = max(0, ys.min() - 10), min(env.shape[0], ys.max() + 10)
    c0, c1 = max(0, xs.min() - 10), min(env.shape[1], xs.max() + 10)

    scl_crop = scl_all[i2, r0:r1, c0:c1]
    vv_crop = s1_npz["vv_db"][i1, r0:r1, c0:c1].astype(np.float32)
    vv_smooth = uniform_filter(np.where(np.isfinite(vv_crop), vv_crop, np.nanmedian(vv_crop)), size=3)
    env_crop = env[r0:r1, c0:c1]
    thr = float(np.clip(threshold_otsu(vv_smooth[env_crop]), -20.0, -12.0))
    water_sar = (vv_smooth < thr) & env_crop

    # Build clean categorical RGB for S2 SCL (clouds = bright white/gray, shadow = dark slate, land = muted olive, water = blue)
    rgb_s2 = np.zeros((*scl_crop.shape, 3), dtype=np.float32)
    rgb_s2[:, :] = [0.48, 0.56, 0.44]  # vegetated/bare land
    rgb_s2[scl_crop == 6] = [0.05, 0.45, 0.75]  # water
    rgb_s2[scl_crop == 3] = [0.35, 0.38, 0.42]  # cloud shadow
    rgb_s2[np.isin(scl_crop, [8, 9, 10])] = [0.95, 0.96, 0.98]  # clouds

    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.22)

    # Panel A: Horizontal bar chart of max blind days & winter cloud %
    ax0 = fig.add_subplot(gs[0])
    df_c = df_cloud.sort_values("max_s2_blind_gap_days", ascending=True)
    y_pos = np.arange(len(df_c))
    bars = ax0.barh(y_pos, df_c["max_s2_blind_gap_days"], color="#0284C7", height=0.55, alpha=0.9)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(df_c["reservoir"], fontsize=10, fontweight="bold")
    ax0.set_xlabel("Longest consecutive Sentinel-2 blind streak (days)", fontsize=9.5)
    ax0.set_xlim(0, 95)
    ax0.set_title("A. Optical Blindness (2025–2026)", fontsize=11, fontweight="bold", loc="left", pad=10)
    for bar, (_, row) in zip(bars, df_c.iterrows()):
        ax0.text(
            bar.get_width() + 1.8,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['max_s2_blind_gap_days'])} d\n({row['s2_winter_cloudy_pct']:.0f}% winter cloudy)",
            va="center",
            fontsize=8.5,
            color="#334155",
        )

    # Panel B: S2 optical scene classification (blocked by clouds)
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(rgb_s2)
    ax1.contour(env_crop, levels=[0.5], colors=["#EF4444"], linewidths=1.1, linestyles="--")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"B. Sentinel-2 Optical ({d2})\nBlocked by winter clouds", fontsize=10.5, fontweight="bold", loc="left", pad=8)
    ax1.text(
        0.04, 0.05, "Dashed = basin ROI\nWhite = cloud cover",
        transform=ax1.transAxes, fontsize=8, color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#CBD5E1"),
    )

    # Panel C: S1 SAR VV backscatter (clear view of refilled lake!)
    ax2 = fig.add_subplot(gs[2])
    im = ax2.imshow(vv_smooth, cmap="gray", vmin=-24, vmax=-5)
    ax2.contour(water_sar, levels=[0.5], colors=["#38BDF8"], linewidths=1.4)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(f"C. Sentinel-1 SAR VV ({d1})\nCloud-free radar water mask", fontsize=10.5, fontweight="bold", loc="left", pad=8)
    ax2.text(
        0.04, 0.05, "Cyan = SAR water\nDark = smooth surface",
        transform=ax2.transAxes, fontsize=8, color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#CBD5E1"),
    )

    out = FIG_DIR / "fig1_cloud_blindness_and_radar_vision.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def fig2_daily_volume_tracking() -> None:
    """
    4-panel operational daily tracking plot (Jan 2025 - Feb 2026):
    Official Daily Gauge vs S2-Optical Only vs Multi-Modal Fusion (S2 + S1).
    """
    df_daily = pd.read_parquet(RES_DIR / "daily_tracking_2025_26.parquet")
    df_vol = pd.read_csv(RES_DIR / "volume_evaluation_2025_26.csv")
    df_cloud = pd.read_csv(RES_DIR / "cloud_blindness_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True)
    axes = axes.ravel()

    for ax, (key, cod, name, cap) in zip(axes, RESERVOIRS):
        sub = df_daily[df_daily["key"] == key].sort_values("date")
        c_row = df_cloud[df_cloud["reservoir"] == name].iloc[0]

        # Highlight longest S2 cloud-blind window
        w_start, w_end = c_row["max_gap_window"].split(" -> ")
        ax.axvspan(pd.Timestamp(w_start), pd.Timestamp(w_end), color="#FEE2E2", alpha=0.55, label="Longest S2 cloud blackout")

        # Official daily ground truth
        ax.plot(sub["date"], sub["volume_mm3"], color=COLORS["gt"], lw=2.2, label="Official daily gauge (ground truth)", zorder=4)

        # S2-Optical only (flat-lines during cloud gaps!)
        r2_s2 = df_vol[(df_vol["reservoir"] == name) & (df_vol["method"] == "S2-Optical Only")]["r2"].values[0]
        ax.step(sub["date"], sub["vol_s2"], where="post", color="#64748B", lw=1.4, ls="--", label=f"S2-Optical only (R² = {r2_s2:.2f})", zorder=3)

        # Best Multi-Modal Fusion (S2 + S1-Otsu on valleys, S2 + S1-ML on Poma — show S2+S1-Otsu consistently AND S2+S1-ML on Poma!)
        r2_fu = df_vol[(df_vol["reservoir"] == name) & (df_vol["method"] == "Fusion (S2 + S1-Otsu)")]["r2"].values[0]
        ax.plot(sub["date"], sub["vol_fused_otsu"], color=COLORS["fusion"], lw=1.7, alpha=0.92, label=f"Fusion S2 + S1-Otsu (R² = {r2_fu:.2f})", zorder=5)

        if key == "poma":
            r2_ml = df_vol[(df_vol["reservoir"] == name) & (df_vol["method"] == "Fusion (S2 + S1-ML)")]["r2"].values[0]
            ax.plot(sub["date"], sub["vol_fused_ml"], color=COLORS["ml"], lw=1.5, alpha=0.88, label=f"Fusion S2 + S1-ML (R² = {r2_ml:.2f})", zorder=5)

        ax.set_title(f"{name}  (Capacity: {cap:.0f} Mm³ | Max S2 blind gap: {int(c_row['max_s2_blind_gap_days'])} d)", fontsize=10.5, fontweight="bold", loc="left", pad=8)
        ax.set_ylabel("Stored Volume (Mm³)", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor="white", framealpha=0.9, edgecolor="#E2E8F0")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    plt.tight_layout()
    out = FIG_DIR / "fig2_daily_volume_tracking_4_reservoirs.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def fig3_otsu_vs_lightgbm_comparison() -> None:
    """
    2-panel grouped bar comparison across all 4 reservoirs:
    Left: Operational Daily R^2 (2025-2026)
    Right: Operational Daily NRMSE (% of reservoir capacity)
    Shows both the dramatic win of SAR/Fusion over Optical-only AND the
    sheltered-valley (Otsu) vs windy-open-basin (LightGBM) trade-off.
    """
    df_vol = pd.read_csv(RES_DIR / "volume_evaluation_2025_26.csv")
    methods = [
        ("S2-Optical Only", "#94A3B8"),
        ("S1-Otsu (Classic SAR)", "#0F766E"),
        ("S1-ML (Pooled)", "#F59E0B"),
        ("Fusion (S2 + S1-Otsu)", "#0284C7"),
        ("Fusion (S2 + S1-ML)", "#7C3AED"),
    ]
    res_names = [r[2] for r in RESERVOIRS]
    x = np.arange(len(res_names))
    w = 0.155

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.9))

    for i, (m, col) in enumerate(methods):
        vals_r2 = [df_vol[(df_vol["reservoir"] == r) & (df_vol["method"] == m)]["r2"].values[0] for r in res_names]
        vals_err = [df_vol[(df_vol["reservoir"] == r) & (df_vol["method"] == m)]["nrmse_cap_pct"].values[0] for r in res_names]
        offset = (i - 2) * w
        ax1.bar(x + offset, vals_r2, width=w, color=col, label=m, edgecolor="white", linewidth=0.5)
        ax2.bar(x + offset, vals_err, width=w, color=col, label=m, edgecolor="white", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(res_names, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Daily Volume R² (higher is better)", fontsize=9.5)
    ax1.set_ylim(0, 1.08)
    ax1.set_title("A. Daily Volume Tracking Accuracy (R², 2025–2026)", fontsize=10.5, fontweight="bold", loc="left", pad=10)

    ax2.set_xticks(x)
    ax2.set_xticklabels(res_names, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("NRMSE (% of reservoir capacity, lower is better)", fontsize=9.5)
    ax2.set_ylim(0, 17.5)
    ax2.set_title("B. Normalized Volume Error (% of Capacity)", fontsize=10.5, fontweight="bold", loc="left", pad=10)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out = FIG_DIR / "fig3_method_comparison_4_reservoirs.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def fig4_hypsometry_and_features() -> None:
    """
    2-panel technical deep dive:
    Left: Physical hypsometric power-law calibration V = c * A^gamma on Lago Fanaco
          (2022-2024 monthly training points vs 2025-2026 unseen daily test points).
    Right: Top SAR feature importances learned by Self-Supervised LightGBM.
    """
    m_gt = pd.read_csv(RAW_DIR / "sicilia_dighe_volumi.csv")
    m_gt["date"] = pd.to_datetime(m_gt["data"])
    m_gt["volume_mm3"] = pd.to_numeric(m_gt["volume"], errors="coerce")
    d_gt = pd.read_csv(RAW_DIR / "sicilia_dighe_volumi_giornalieri.csv")
    d_gt["date"] = pd.to_datetime(d_gt["data"])
    d_gt["volume_mm3"] = pd.to_numeric(d_gt["volume"], errors="coerce") / 1e6

    s2_f = pd.read_csv(RES_DIR / "fanaco_s2_passes.csv", parse_dates=["dt"])
    with open(RES_DIR / "fanaco_hypsometry.json", encoding="utf-8") as f:
        hyp = json.load(f)

    mg = m_gt[(m_gt["cod"] == "dig-07") & (m_gt["date"] >= "2022-01-01") & (m_gt["date"] <= "2024-12-31")].sort_values("date")
    tr = pd.merge_asof(
        mg[["date", "volume_mm3"]],
        s2_f[(s2_f["dt"] <= "2024-12-31") & (s2_f["is_clear"])][["dt", "area_km2"]].sort_values("dt"),
        left_on="date", right_on="dt", direction="nearest", tolerance=pd.Timedelta("12D"),
    ).dropna()

    dg = d_gt[(d_gt["cod"] == "dig-07") & (d_gt["date"] >= "2025-01-01")][["date", "volume_mm3"]].dropna().sort_values("date")
    te = s2_f[(s2_f["dt"] >= "2025-01-01") & (s2_f["is_clear"])].merge(dg, left_on="dt", right_on="date")

    fi = pd.read_csv(RES_DIR / "feature_importance.csv").head(8).sort_values("importance_pct", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6))

    # Panel A: Hypsometric power law
    a_grid = np.linspace(0.05, max(tr["area_km2"].max(), te["area_km2"].max()) * 1.08, 200)
    v_grid = hyp["c"] * np.power(a_grid, hyp["gamma"])
    ax1.plot(a_grid, v_grid, color="#0284C7", lw=2.2, label=f"Fit: V = {hyp['c']:.1f} · A^{hyp['gamma']:.2f}")
    ax1.scatter(tr["area_km2"], tr["volume_mm3"], color="#475569", s=42, alpha=0.8, label="Train (2022–2024 monthly)")
    ax1.scatter(te["area_km2"], te["volume_mm3"], color="#F59E0B", edgecolor="#78350F", s=48, zorder=5, label="Test (2025–2026 daily)")
    ax1.set_xlabel("Satellite-Derived Water Surface Area (km²)", fontsize=9.5)
    ax1.set_ylabel("Official Reservoir Volume (Mm³)", fontsize=9.5)
    ax1.set_title("A. Hypsometric Area–Volume Law (Lago Fanaco)", fontsize=10.5, fontweight="bold", loc="left", pad=10)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.5)

    # Panel B: LightGBM SAR Feature Importance
    bars = ax2.barh(fi["feature"], fi["importance_pct"], color="#0F766E", height=0.6, alpha=0.9)
    ax2.set_xlabel("Share of LightGBM splits (%)", fontsize=9.5)
    ax2.set_xlim(0, fi["importance_pct"].max() * 1.22)
    ax2.set_title("B. What the Self-Supervised SAR Model Relies On", fontsize=10.5, fontweight="bold", loc="left", pad=10)
    for bar, val in zip(bars, fi["importance_pct"]):
        ax2.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.5, color="#334155")

    plt.tight_layout()
    out = FIG_DIR / "fig4_hypsometry_and_sar_features.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig1_cloud_blindness_and_radar_vision()
    fig2_daily_volume_tracking()
    fig3_otsu_vs_lightgbm_comparison()
    fig4_hypsometry_and_features()
    print("All 4 TDS figures generated successfully.")

"""
Generate 4 publication-grade figures for the Environmental Data Science (Cambridge)
Application Paper, incorporating the full 10-model benchmark including Dual-Pol
S1-GMM, Random Forest, LightGBM (Raw vs. Scene-Normalized), and Optical+SAR Fusion.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter, binary_dilation, label
from skimage.filters import threshold_otsu

REPO_DIR = Path(__file__).resolve().parents[1]
CUBE_DIR = REPO_DIR / "data" / "cubes"
RES_DIR = REPO_DIR / "results"
EDS_DIR = RES_DIR / "eds_benchmark"
FIG_DIR = REPO_DIR / "figures" / "eds"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#EAEAEA",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 260,
    "savefig.bbox": "tight",
})

RESERVOIRS = [
    ("rosamarina", "Lago Rosamarina", 73.03),
    ("poma",       "Lago Poma",       72.30),
    ("castello",   "Lago Castello",   19.49),
    ("fanaco",     "Lago Fanaco",     20.70),
]


def fig1_cloud_and_radar() -> None:
    df_cloud = pd.read_csv(RES_DIR / "cloud_blindness_summary.csv")
    s2_npz = np.load(CUBE_DIR / "fanaco_s2.npz")
    s2_meta = pd.read_parquet(CUBE_DIR / "fanaco_s2_meta.parquet")
    s1_npz = np.load(CUBE_DIR / "fanaco_s1.npz")
    s1_meta = pd.read_parquet(CUBE_DIR / "fanaco_s1_meta.parquet")

    s2_win = s2_meta[(s2_meta["date"] >= "2026-01-20") & (s2_meta["date"] <= "2026-02-20") & (s2_meta["roi_cloud_frac"] > 0.70)]
    i2 = int(s2_win.index[-1]) if len(s2_win) else int(s2_meta.index[-5])
    d2 = s2_meta.loc[i2, "date"]

    s1_meta["dt"] = pd.to_datetime(s1_meta["date"])
    i1 = int((s1_meta["dt"] - pd.Timestamp(d2)).abs().idxmin())
    d1 = s1_meta.loc[i1, "date"]

    mndwi_all = s2_npz["mndwi"].astype(np.float32)
    scl_all = s2_npz["scl"]
    train_idx = s2_meta.index[(s2_meta["date"] <= "2024-12-31") & (s2_meta["roi_cloud_frac"] < 0.08)].tolist()
    union_w = np.zeros(mndwi_all.shape[1:], dtype=bool)
    for idx in train_idx:
        union_w |= (mndwi_all[idx] > 0.05) & (scl_all[idx] == 6)
    lab, _ = label(union_w)
    main_id = int(np.argmax(np.bincount(lab.ravel())[1:])) + 1
    env = binary_dilation(lab == main_id, iterations=20)

    ys, xs = np.where(env)
    r0, r1 = max(0, ys.min() - 10), min(env.shape[0], ys.max() + 10)
    c0, c1 = max(0, xs.min() - 10), min(env.shape[1], xs.max() + 10)

    scl_crop = scl_all[i2, r0:r1, c0:c1]
    vv_crop = s1_npz["vv_db"][i1, r0:r1, c0:c1].astype(np.float32)
    vv_smooth = uniform_filter(np.where(np.isfinite(vv_crop), vv_crop, np.nanmedian(vv_crop)), size=3)
    env_crop = env[r0:r1, c0:c1]
    thr = float(np.clip(threshold_otsu(vv_smooth[env_crop]), -20.0, -12.0))
    water_sar = (vv_smooth < thr) & env_crop

    rgb_s2 = np.zeros((*scl_crop.shape, 3), dtype=np.float32)
    rgb_s2[:, :] = [0.48, 0.56, 0.44]
    rgb_s2[scl_crop == 6] = [0.05, 0.45, 0.75]
    rgb_s2[scl_crop == 3] = [0.35, 0.38, 0.42]
    rgb_s2[np.isin(scl_crop, [8, 9, 10])] = [0.95, 0.96, 0.98]

    fig = plt.figure(figsize=(13.2, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.22)

    ax0 = fig.add_subplot(gs[0])
    df_c = df_cloud.sort_values("max_s2_blind_gap_days", ascending=True)
    y_pos = np.arange(len(df_c))
    bars = ax0.barh(y_pos, df_c["max_s2_blind_gap_days"], color="#0284C7", height=0.55)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(df_c["reservoir"], fontsize=9.5, fontweight="bold")
    ax0.set_xlabel("Longest consecutive Sentinel-2 blind gap (days)", fontsize=9.5)
    ax0.set_xlim(0, 96)
    ax0.set_title("(a) Optical Cloud Blindness (2025–2026)", fontsize=10.5, fontweight="bold", loc="left", pad=8)
    for bar, (_, row) in zip(bars, df_c.iterrows()):
        ax0.text(
            bar.get_width() + 1.8, bar.get_y() + bar.get_height() / 2,
            f"{int(row['max_s2_blind_gap_days'])} d\n({row['s2_winter_cloudy_pct']:.0f}% winter cloudy)",
            va="center", fontsize=8.3, color="#334155",
        )

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(rgb_s2)
    ax1.contour(env_crop, levels=[0.5], colors=["#EF4444"], linewidths=1.1, linestyles="--")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f"(b) Sentinel-2 SCL ({d2})\nObscured by winter clouds", fontsize=10, fontweight="bold", loc="left", pad=8)

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(vv_smooth, cmap="gray", vmin=-24, vmax=-5)
    ax2.contour(water_sar, levels=[0.5], colors=["#38BDF8"], linewidths=1.3)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"(c) Sentinel-1 SAR VV ({d1})\nAll-weather water delineation", fontsize=10, fontweight="bold", loc="left", pad=8)

    fig.savefig(FIG_DIR / "eds_fig1_cloud_blindness_and_sar.png")
    plt.close(fig)


def fig2_daily_tracking_gmm() -> None:
    df_daily = pd.read_parquet(EDS_DIR / "eds_daily_series_2025_26.parquet")
    df_vol = pd.read_csv(EDS_DIR / "eds_volume_metrics_2025_26.csv")
    df_cloud = pd.read_csv(RES_DIR / "cloud_blindness_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), sharex=True)
    axes = axes.ravel()

    for ax, (key, name, cap) in zip(axes, RESERVOIRS):
        sub = df_daily[df_daily["key"] == key].sort_values("date")
        c_row = df_cloud[df_cloud["reservoir"] == name].iloc[0]
        w_start, w_end = c_row["max_gap_window"].split(" -> ")
        ax.axvspan(pd.Timestamp(w_start), pd.Timestamp(w_end), color="#FEE2E2", alpha=0.55, label="Longest S2 cloud blackout")

        ax.plot(sub["date"], sub["volume_mm3"], color="#111827", lw=2.2, label="Official daily gauge", zorder=4)

        r2_s2 = df_vol[(df_vol["reservoir"] == name) & (df_vol["method"] == "S2-MNDWI (Optical)")]["r2"].values[0]
        ax.step(sub["date"], sub["vol_s2"], where="post", color="#64748B", lw=1.3, ls="--", label=f"S2-MNDWI only (R² = {r2_s2:.2f})", zorder=3)

        r2_gmm = df_vol[(df_vol["reservoir"] == name) & (df_vol["method"] == "Fusion (S2 + S1-GMM)")]["r2"].values[0]
        ax.plot(sub["date"], sub["vol_fu_gmm"], color="#0284C7", lw=1.8, alpha=0.95, label=f"Fusion S2 + S1-GMM (R² = {r2_gmm:.2f})", zorder=5)

        r2_lgb = df_vol[(df_vol["reservoir"] == name) & (df_vol["method"] == "S1-LGBM (Raw dB)")]["r2"].values[0]
        ax.plot(sub["date"], sub["vol_lgb_raw"], color="#D97706", lw=1.1, alpha=0.75, label=f"S1-LGBM Raw dB (R² = {r2_lgb:.2f})", zorder=3)

        ax.set_title(f"{name} (Capacity: {cap:.1f} Mm³)", fontsize=10.5, fontweight="bold", loc="left", pad=6)
        ax.set_ylabel("Stored Volume (Mm³)", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#E2E8F0")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eds_fig2_daily_volume_tracking.png")
    plt.close(fig)


def fig3_benchmark_comparison() -> None:
    df_vol = pd.read_csv(EDS_DIR / "eds_volume_metrics_2025_26.csv")
    methods = [
        ("S2-MNDWI (Optical)", "#94A3B8"),
        ("S1-Otsu (Classic SAR)", "#0F766E"),
        ("S1-GMM (Dual-Pol)", "#0369A1"),
        ("S1-RF (Raw dB)", "#CA8A04"),
        ("S1-LGBM (Raw dB)", "#EA580C"),
        ("S1-LGBM (Scene-Norm)", "#DC2626"),
        ("Fusion (S2 + S1-GMM)", "#1D4ED8"),
    ]
    res_names = [r[1] for r in RESERVOIRS]
    x = np.arange(len(res_names))
    w = 0.115

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.9))
    for i, (m, col) in enumerate(methods):
        vals_r2 = [df_vol[(df_vol["reservoir"] == r) & (df_vol["method"] == m)]["r2"].values[0] for r in res_names]
        vals_err = [df_vol[(df_vol["reservoir"] == r) & (df_vol["method"] == m)]["nrmse_cap_pct"].values[0] for r in res_names]
        offset = (i - 3) * w
        ax1.bar(x + offset, vals_r2, width=w, color=col, label=m, edgecolor="white", linewidth=0.4)
        ax2.bar(x + offset, vals_err, width=w, color=col, label=m, edgecolor="white", linewidth=0.4)

    ax1.set_xticks(x); ax1.set_xticklabels(res_names, fontsize=9.3, fontweight="bold")
    ax1.set_ylabel("Operational Daily R² (higher is better)", fontsize=9.5)
    ax1.set_ylim(0, 1.08)
    ax1.set_title("(a) Daily Volume Tracking Accuracy (R²)", fontsize=10.5, fontweight="bold", loc="left", pad=8)

    ax2.set_xticks(x); ax2.set_xticklabels(res_names, fontsize=9.3, fontweight="bold")
    ax2.set_ylabel("NRMSE (% of capacity, lower is better)", fontsize=9.5)
    ax2.set_ylim(0, 16.5)
    ax2.set_title("(b) Capacity-Normalized Error (NRMSE %)", fontsize=10.5, fontweight="bold", loc="left", pad=8)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09), frameon=False, fontsize=8.8)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIG_DIR / "eds_fig3_10model_comparison.png")
    plt.close(fig)


def fig4_ablation_bias_and_features() -> None:
    """
    (a) Mean daily volume bias across all 4 reservoirs showing why Scene-Norm z-scores
        introduce strong negative bias during winter refills, while Dual-Pol GMM stays nearly unbiased.
    (b) Top 8 feature importances in Scene-Normalized LightGBM.
    """
    df_vol = pd.read_csv(EDS_DIR / "eds_volume_metrics_2025_26.csv")
    fi = pd.read_csv(EDS_DIR / "eds_feature_importance_scene_norm.csv").head(8).sort_values("importance_pct", ascending=True)

    sel_methods = [
        "Fusion (S2 + S1-GMM)",
        "S1-GMM (Dual-Pol)",
        "S1-Otsu (Classic SAR)",
        "S1-RF (Raw dB)",
        "S1-LGBM (Raw dB)",
        "S1-LGBM (Scene-Norm)",
    ]
    sub = df_vol[df_vol["method"].isin(sel_methods)].groupby("method", as_index=False)["bias_mm3"].mean()
    sub["order"] = sub["method"].map({m: idx for idx, m in enumerate(sel_methods)})
    sub = sub.sort_values("order", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.5))

    colors_b = ["#DC2626" if v < -2.0 else ("#D97706" if v < -1.0 else "#0284C7") for v in sub["bias_mm3"]]
    bars1 = ax1.barh(sub["method"], sub["bias_mm3"], color=colors_b, height=0.58)
    ax1.axvline(0.0, color="#111827", lw=1.0)
    ax1.set_xlabel("Mean Daily Volume Bias (Mm³, closer to 0 is better)", fontsize=9.3)
    ax1.set_xlim(-3.2, 0.6)
    ax1.set_title("(a) Systematic Volume Bias (2025–2026 Out-of-Time)", fontsize=10.3, fontweight="bold", loc="left", pad=8)
    for bar, val in zip(bars1, sub["bias_mm3"]):
        ax1.text(val - 0.08, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}", va="center", ha="right", fontsize=8.5, color="#111827", fontweight="bold")

    bars2 = ax2.barh(fi["feature"], fi["importance_pct"], color="#0F766E", height=0.6)
    ax2.set_xlabel("Share of LightGBM splits (%)", fontsize=9.3)
    ax2.set_xlim(0, fi["importance_pct"].max() * 1.24)
    ax2.set_title("(b) Top SAR Features in Scene-Normalized LightGBM", fontsize=10.3, fontweight="bold", loc="left", pad=8)
    for bar, val in zip(bars2, fi["importance_pct"]):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.5, color="#334155")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "eds_fig4_bias_ablation_and_features.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_cloud_and_radar()
    fig2_daily_tracking_gmm()
    fig3_benchmark_comparison()
    fig4_ablation_bias_and_features()
    print(f"Generated 4 EDS figures in {FIG_DIR}")

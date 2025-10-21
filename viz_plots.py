# viz_plots.py
# matplotlib-only; contextily가 없으면 기본 산점도/차트로 대체
from typing import Optional, Sequence, Tuple, List, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from config import OUTDIR
except Exception:
    OUTDIR = os.getcwd()

# contextily는 옵션
_HAS_CTX = False
try:
    import contextily as cx  # type: ignore
    _HAS_CTX = True
except Exception:
    _HAS_CTX = False

# ---------- 유틸 ----------
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def _finalize(fig: plt.Figure, save_name: str, title: Optional[str] = None):
    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    if not os.path.isabs(save_name):
        save_path = os.path.join(OUTDIR, save_name)
    else:
        save_path = save_name
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# =========================================================
# 0) HGNN: 보조 플롯들 (NEW)
# =========================================================
def plot_hgnn_training_curve(losses: np.ndarray, title: str = "HGNN Training Loss", save_name: str = "fig_hgnn_loss.png"):
    if losses is None or len(losses) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(1, len(losses)+1)
    ax.plot(x, losses, lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Hinge Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    _finalize(fig, save_name)

def _pca_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:2].T
    return Z

def plot_poi_embedding_pca(pdf: pd.DataFrame, emb: np.ndarray,
                           title: str = "POI Embedding PCA",
                           save_name: str = "fig_hgnn_poi_pca.png"):
    if pdf is None or pdf.empty or emb is None or len(emb) == 0:
        return
    d = min(emb.shape[0], len(pdf))
    Z = _pca_2d(emb[:d])
    cats = ["교육","교통","편의시설"]
    color_map = {"교육": "tab:blue", "교통": "tab:green", "편의시설": "tab:purple"}

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for cat in cats:
        mask = (pdf.iloc[:d]["super_cat"].astype(str) == cat).to_numpy()
        if mask.any():
            ax.scatter(Z[mask,0], Z[mask,1], s=18, alpha=0.8, label=cat, c=color_map.get(cat, "tab:gray"))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    _finalize(fig, save_name)

def plot_edge_bucket_hist(edf: pd.DataFrame, title: str = "Edge Distance Buckets", save_name: str = "fig_hgnn_edge_buckets.png"):
    if edf is None or edf.empty or "bucket" not in edf.columns:
        return
    counts = edf["bucket"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(counts.index.astype(int).astype(str), counts.values)
    ax.set_xlabel("Distance bucket (0:<300,1:<500,2:<1000,3:<2000,4:≥2000)")
    ax.set_ylabel("#Edges")
    ax.set_title(title)
    _finalize(fig, save_name)

def plot_supercat_bar(sc_emb: Dict[str, np.ndarray],
                      title: str = "Super-Category Embedding (L2 norm)",
                      save_name: str = "fig_hgnn_supercat_bar.png"):
    if sc_emb is None or len(sc_emb) == 0:
        return
    cats = ["교육","교통","편의시설"]
    vals = [float(np.linalg.norm(np.asarray(sc_emb.get(c, np.zeros(1)), float))) for c in cats]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(cats, vals)
    ax.set_ylabel("L2 norm")
    ax.set_title(title)
    _finalize(fig, save_name)

# =========================================================
# 1) HGNN 그래프 시각화: 매물-POI 및 엣지(선) 표시
# =========================================================
def plot_hgnn_graph(
    cdf: pd.DataFrame,
    pdf: pd.DataFrame,
    edf: pd.DataFrame,
    title: str = "HGNN Graph around Target",
    save_name: str = "fig_hgnn_graph.png"
) -> None:
    if cdf is None or cdf.empty:
        raise ValueError("plot_hgnn_graph: complex dataframe(cdf)가 비어있습니다.")
    if pdf is None:
        pdf = pd.DataFrame()
    if edf is None:
        edf = pd.DataFrame()

    fig, ax = plt.subplots(figsize=(8, 7))

    # Target complex
    cx = float(cdf.iloc[0]["lon"])
    cy = float(cdf.iloc[0]["lat"])
    ax.scatter([cx], [cy], s=150, c="tab:red", marker="*", label="Target complex", zorder=5)

    # POIs by super-category
    if not pdf.empty:
        cats = ["교육", "교통", "편의시설"]
        color_map = {"교육": "tab:blue", "교통": "tab:green", "편의시설": "tab:purple"}
        for cat in cats:
            sub = pdf[pdf.get("super_cat", "").astype(str) == cat]
            if len(sub):
                ax.scatter(sub["lon"].astype(float), sub["lat"].astype(float),
                           s=18, c=color_map.get(cat, "tab:gray"), label=f"POI-{cat}", alpha=0.7, zorder=3)

    # Edges: complex → POI
    if not edf.empty and not pdf.empty:
        pid2coord = {
            str(r["poi_id"]): (float(r["lon"]), float(r["lat"]))
            for _, r in pdf.iterrows()
        }
        for _, e in edf.iterrows():
            pid = str(e["poi_id"])
            coord = pid2coord.get(pid)
            if coord:
                px, py = coord
                ax.plot([cx, px], [cy, py], lw=0.6, c="lightgray", alpha=0.6, zorder=1)

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9, frameon=True)
    _finalize(fig, save_name)

# =========================================================
# 2) Retrieval 지도
# =========================================================
def plot_retrieval_map(
    master: pd.DataFrame,
    target_id: str,
    neighbor_ids: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    topk_label: int = 10,
    title: str = "Target and Retrieved Neighbors",
    save_name: str = "fig_retrieval_map.png"
) -> None:
    if master is None or master.empty:
        raise ValueError("plot_retrieval_map: master가 비었습니다.")
    trow = master.loc[master["complex_id"].astype(str) == str(target_id)]
    if trow.empty:
        raise ValueError(f"plot_retrieval_map: target_id({target_id})를 master에서 찾지 못했습니다.")
    tx, ty = float(trow.iloc[0]["lon"]), float(trow.iloc[0]["lat"])

    ndf = master[master["complex_id"].astype(str).isin([str(i) for i in neighbor_ids])].copy()
    if ndf.empty:
        raise ValueError("plot_retrieval_map: neighbor_ids에 해당하는 항목이 master에 없습니다.")

    if scores is not None and len(scores) == len(ndf):
        ndf = ndf.assign(_score=np.asarray(scores, dtype=float))
        ndf = ndf.sort_values("_score", ascending=False)
    else:
        ndf = ndf.assign(_score=np.linspace(1.0, 0.1, len(ndf)))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter([tx], [ty], s=180, c="tab:red", marker="*", label=f"Target {target_id}", zorder=5)

    sc = ndf["_score"]
    sizes = 50 + 250 * (sc - sc.min()) / (sc.max() - sc.min() + 1e-6)
    pts = ax.scatter(ndf["lon"].astype(float), ndf["lat"].astype(float),
                     s=sizes, c=sc, cmap="viridis", alpha=0.85, zorder=3, label="Neighbors")

    for i, (_, r) in enumerate(ndf.head(topk_label).iterrows(), start=1):
        ax.annotate(f"{i}", (float(r["lon"]), float(r["lat"])),
                    xytext=(3, 3), textcoords="offset points", fontsize=8, color="black")

    cb = fig.colorbar(pts, ax=ax, shrink=0.8)
    cb.set_label("Similarity score")

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    _finalize(fig, save_name)

# =========================================================
# 3) pseudo-cold 비교
# =========================================================
def plot_pseudo_cold_compare(
    monthly: pd.DataFrame,
    target_id: str,
    months_all: pd.DatetimeIndex,
    T_in: int,
    H: int,
    synth_ctx: Optional[np.ndarray] = None,
    title: str = "Pseudo Cold vs Ground Truth",
    save_name: str = "fig_pseudo_cold.png"
) -> None:
    sub = monthly.loc[monthly["complex_id"].astype(str) == str(target_id), ["ym", "ppsm_median"]].drop_duplicates("ym")
    if sub.empty:
        raise ValueError("plot_pseudo_cold_compare: 대상 매물의 monthly 데이터가 없습니다.")
    s = sub.set_index("ym").reindex(months_all)["ppsm_median"].astype(float).ffill().bfill().values

    if len(s) < T_in + H:
        raise ValueError("plot_pseudo_cold_compare: 시계열 길이가 T_in+H보다 짧습니다.")

    ctx_gt = s[-(T_in + H): -H]
    future_gt = s[-H:]
    zero_ctx = np.zeros_like(ctx_gt)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x_ctx = np.arange(1, T_in + 1)
    x_fut = np.arange(T_in + 1, T_in + H + 1)

    ax.plot(x_ctx, ctx_gt, label="GT context", lw=2, alpha=0.9)
    ax.plot(x_ctx, zero_ctx, label="Zero context (baseline)", lw=1.5, linestyle="--", alpha=0.8)

    if synth_ctx is not None and len(synth_ctx) == T_in:
        ax.plot(x_ctx, synth_ctx, label="Synthetic context", lw=2, alpha=0.9)

    ax.plot(x_fut, future_gt, label="GT future", lw=2.2, c="tab:green", alpha=0.9)

    ax.set_xlabel("Steps (context→future)"); ax.set_ylabel("Median price per sqm")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    _finalize(fig, save_name)

# =========================================================
# 4) 예측 시각화
# =========================================================
def plot_forecast_result(
    monthly: pd.DataFrame,
    target_id: str,
    months_all: pd.DatetimeIndex,
    pred: np.ndarray,
    H: int,
    q10: Optional[np.ndarray] = None,
    q90: Optional[np.ndarray] = None,
    title: str = "Forecast",
    save_name: str = "fig_forecast.png",
    show_hist_len: int = 60
) -> None:
    sub = monthly.loc[monthly["complex_id"].astype(str) == str(target_id), ["ym", "ppsm_median"]].drop_duplicates("ym")
    if sub.empty:
        raise ValueError("plot_forecast_result: 대상 매물의 monthly 데이터가 없습니다.")
    serie = sub.set_index("ym").reindex(months_all)["ppsm_median"].astype(float).ffill().bfill().values

    if len(serie) < H:
        raise ValueError("plot_forecast_result: 시계열 길이가 H보다 짧습니다.")

    hist = serie[:-H] if H > 0 else serie.copy()
    fut_gt = serie[-H:] if H > 0 else np.array([])
    hist = hist[-show_hist_len:] if len(hist) > show_hist_len else hist

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x_hist = np.arange(1, len(hist) + 1)
    x_fut = np.arange(len(hist) + 1, len(hist) + H + 1)

    ax.plot(x_hist, hist, lw=2, label="History")
    if H > 0 and len(fut_gt) == H:
        ax.plot(x_fut, fut_gt, lw=2, label="Future GT", c="tab:green")

    if pred is not None and len(pred) == H:
        ax.plot(x_fut, pred, lw=2.2, label="Prediction", c="tab:orange")
        if q10 is not None and q90 is not None and len(q10) == H and len(q90) == H:
            ax.fill_between(x_fut, q10, q90, alpha=0.2, label="Confidence band")

    ax.set_xlabel("Time"); ax.set_ylabel("Median price per sqm")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    _finalize(fig, save_name)

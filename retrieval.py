# retrieval.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import os
import numpy as np
import pandas as pd

from config import USE_FAISS, NEIGHBOR_TOPK, GEO_NEAREST_N, GEO_MIX_BETA, SCALE_ALIGN_WINDOW

# ───────────────── Spatial Embedding ─────────────────
def _try_load_hgnn_embeddings(master: pd.DataFrame, id_col: str = "complex_id"):
    """
    OUTDIR/hgnn_emb.csv 또는 hgnn_emb.csv가 있으면 (id_col, e0,e1,...) 형식으로 로드.
    없으면 None 반환 → meta_embed의 기본 임베딩 사용.
    """
    candidates = []
    try:
        from config import OUTDIR
        candidates.append(os.path.join(OUTDIR, "hgnn_emb.csv"))
    except Exception:
        pass
    candidates.append("hgnn_emb.csv")
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if id_col not in df.columns:
                continue
            df[id_col] = df[id_col].astype(str)
            dim_cols = [c for c in df.columns if c != id_col]
            ids = master[id_col].astype(str).tolist()
            emb_map = {str(r[id_col]): r[dim_cols].values.astype(np.float32) for _, r in df.iterrows()}
            X = np.stack([emb_map.get(i, np.zeros(len(dim_cols), np.float32)) for i in ids], axis=0)
            return X
    return None

def meta_embed(master: pd.DataFrame) -> np.ndarray:
    """
    기존 구현 유지: lat/lon/year 정규화 임베딩.
    단, HGNN 임베딩 파일이 있으면 우선 사용(기존 기능 보존 + 확장).
    """
    X_h = _try_load_hgnn_embeddings(master)
    if X_h is not None:
        return X_h

    # (기존 코드 유지) lat lon construction_year 모두 z정규화. :contentReference[oaicite:4]{index=4}
    lat = master["lat"].astype(float).to_numpy()
    lon = master["lon"].astype(float).to_numpy()
    yr  = master["construction_year"].astype(float).to_numpy()
    latn = (lat - np.nanmean(lat)) / (np.nanstd(lat) + 1e-6)
    lonn = (lon - np.nanmean(lon)) / (np.nanstd(lon) + 1e-6)
    yrn  = (yr  - np.nanmean(yr))  / (np.nanstd(yr)  + 1e-6)
    X = np.stack([latn, lonn, yrn], axis=1).astype(np.float32)
    return X

# ───────────────── Retrieval index (FAISS/SK) ─────────────────
def build_retrieval_index(X: np.ndarray):
    # (기존 기능 유지) :contentReference[oaicite:5]{index=5}
    if USE_FAISS:
        try:
            import faiss
            index = faiss.IndexFlatL2(X.shape[1])
            index.add(X.astype(np.float32))
            return ("faiss", index)
        except Exception:
            pass
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(NEIGHBOR_TOPK+GEO_NEAREST_N+1, len(X)), metric="euclidean")
    nn.fit(X)
    return ("sk", nn)

def geo_nearest(master: pd.DataFrame, target_row: pd.Series, topn: int) -> np.ndarray:
    # (기존 기능 유지) :contentReference[oaicite:6]{index=6}
    lat0, lon0 = float(target_row["lat"]), float(target_row["lon"])
    lat = master["lat"].to_numpy(); lon = master["lon"].to_numpy()
    d = (lat - lat0)**2 + (lon - lon0)**2
    idx = np.argsort(d)
    return idx[1: topn+1]  # exclude self

def find_neighbors(ids, master, X, target_index, topk=NEIGHBOR_TOPK, geo_n=GEO_NEAREST_N):
    # (기존 기능 유지) 메타+지오 후보 병합. :contentReference[oaicite:7]{index=7}
    kind, index = build_retrieval_index(X)
    if kind == "faiss":
        D, I = index.search(X[target_index:target_index+1], topk+1)
        cand = I[0][1:]
    else:
        D, I = index.kneighbors(X[target_index:target_index+1])
        cand = I[0][1: topk+1]
    gi = geo_nearest(master, master.iloc[target_index], geo_n)
    merged = list(dict.fromkeys(list(cand) + list(gi)))
    return np.array(merged, dtype=int)

def scale_align(monthly: pd.DataFrame, target_id, neigh_ids, window=SCALE_ALIGN_WINDOW):
    # (기존 기능 유지) 최근 window 중앙값 비율로 스케일 정렬. :contentReference[oaicite:8]{index=8}
    df_t = monthly.loc[monthly["complex_id"]==target_id].sort_values("ym").tail(window)
    if df_t.empty:
        return {nid:1.0 for nid in neigh_ids}
    t_med = df_t["ppsm_median"].median()
    weights = {}
    for nid in neigh_ids:
        df_n = monthly.loc[monthly["complex_id"]==nid].sort_values("ym").tail(window)
        if df_n.empty or df_n["ppsm_median"].isna().all():
            weights[nid] = 1.0
        else:
            n_med = df_n["ppsm_median"].median()
            weights[nid] = (t_med/(n_med+1e-6)) if np.isfinite(n_med) else 1.0
    return weights

# ───────────────── Hybrid weights (meta+geo+sem) ─────────────────
def _safe_minmax(x: np.ndarray) -> np.ndarray:
    # (기존 기능 유지) :contentReference[oaicite:9]{index=9}
    x = np.asarray(x, float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x) + 0.5
    return (x - mn) / (mx - mn + 1e-12)

def neighbor_weights_hybrid(
    master: pd.DataFrame,
    target_idx: int,
    cand_idx: np.ndarray,
    X_meta: np.ndarray,
    sc_emb_target: Optional[Dict[str, np.ndarray]] = None,
    sc_emb_pool: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    w_meta: float = 0.55,
    w_geo: float  = 0.25,
    w_sem: float  = 0.20,
    ):
    """
    (기존 기능 유지) meta+geo+sem 조합 가중치 및 정렬 인덱스 반환. :contentReference[oaicite:10]{index=10}
    """
    ids = master["complex_id"].astype(str).tolist()
    t_id = ids[target_idx]
    t_lat, t_lon = float(master.iloc[target_idx]["lat"]), float(master.iloc[target_idx]["lon"])

    # meta sim(L2→1/(1+d))
    t_vec = X_meta[target_idx:target_idx+1]; c_vec = X_meta[cand_idx]
    d_meta = np.sqrt(((c_vec - t_vec)**2).sum(axis=1))
    meta_sim = _safe_minmax(1.0/(1.0 + d_meta))

    # geo sim
    lat = master.iloc[cand_idx]["lat"].to_numpy(dtype=float)
    lon = master.iloc[cand_idx]["lon"].to_numpy(dtype=float)
    d_geo = (lat - t_lat)**2 + (lon - t_lon)**2
    geo_sim = _safe_minmax(1.0 / (1.0 + d_geo / (np.nanmean(d_geo) + 1e-9)))

    # sem sim(옵션)
    if sc_emb_target is not None and sc_emb_pool is not None:
        def _stack_sc(sc: Dict[str,np.ndarray]) -> np.ndarray:
            ks = ["교육","교통","편의시설"]
            vecs = []
            for k in ks:
                v = sc.get(k, None)
                if v is None: vecs.append(np.zeros(64, dtype=np.float32))
                else: vecs.append(np.asarray(v, dtype=np.float32))
            return np.concatenate(vecs, axis=0)
        t_sc = _stack_sc(sc_emb_target)
        pool_sc = []
        for i in cand_idx:
            sc_dict = sc_emb_pool.get(ids[i], None)
            pool_sc.append(_stack_sc(sc_dict) if sc_dict is not None else np.zeros_like(t_sc))
        pool_sc = np.stack(pool_sc, axis=0)
        t_norm = np.linalg.norm(t_sc) + 1e-8
        p_norm = np.linalg.norm(pool_sc, axis=1) + 1e-8
        sem_sim = _safe_minmax((pool_sc @ t_sc) / (t_norm * p_norm))
    else:
        sem_sim = np.zeros(len(cand_idx), dtype=np.float32)

    # 합성 가중치
    comb = w_meta*meta_sim + w_geo*geo_sim + w_sem*sem_sim
    comb = np.maximum(comb, 0)
    comb = (comb/comb.sum()) if comb.sum()>0 else np.ones_like(comb)/len(comb)
    order = np.argsort(-comb)
    return meta_sim, geo_sim, sem_sim, comb[order], cand_idx[order]

# (옵션) 튜닝 유틸도 유지. :contentReference[oaicite:11]{index=11}
def tune_hybrid_weights(*args, **kwargs):
    from eval import evaluate_ids  # 지연 임포트(원본 구조 유지)
    master, months_all, ids, X_meta, hot_ids = args[1], args[2], args[3], args[4], args[5]
    make_forecaster_fn, H = args[6], args[7]
    grid = kwargs.get("grid", ([0.45,0.55,0.65],[0.15,0.25,0.35],[0.00,0.10,0.20]))
    seed = kwargs.get("seed", 42)
    rng = np.random.RandomState(seed)
    pool = list(hot_ids)[:]
    if len(pool) > kwargs.get("max_eval", 800):
        pool = list(rng.choice(pool, size=kwargs.get("max_eval", 800), replace=False))
    best = (0.55,0.25,0.20); best_mape = 1e9
    for wm in grid[0]:
        for wg in grid[1]:
            for ws in grid[2]:
                if abs(wm+wg+ws - 1.0) > 1e-6:
                    continue
                f = make_forecaster_fn(wm, wg, ws)
                res = evaluate_ids(args[0], pool, f, months_all, H)
                mape = res.get("MAPE", np.nan)
                if np.isfinite(mape) and mape < best_mape:
                    best_mape = mape; best = (wm,wg,ws)
    return best

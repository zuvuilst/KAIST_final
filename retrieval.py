# retrieval.py
from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd

from config import USE_FAISS, NEIGHBOR_TOPK, GEO_NEAREST_N, GEO_MIX_BETA, SCALE_ALIGN_WINDOW
from eval import evaluate_ids

def meta_embed(master: pd.DataFrame) -> np.ndarray:
    # lat lon construction_year 모두 z 정규화하여 스케일 균형
    lat = master["lat"].astype(float).to_numpy()
    lon = master["lon"].astype(float).to_numpy()
    yr  = master["construction_year"].astype(float).to_numpy()
    latn = (lat - np.nanmean(lat)) / (np.nanstd(lat) + 1e-6)
    lonn = (lon - np.nanmean(lon)) / (np.nanstd(lon) + 1e-6)
    yrn  = (yr  - np.nanmean(yr))  / (np.nanstd(yr)  + 1e-6)
    X = np.stack([latn, lonn, yrn], axis=1).astype(np.float32)
    return X

def build_retrieval_index(X: np.ndarray):
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
    lat0, lon0 = float(target_row["lat"]), float(target_row["lon"])
    lat = master["lat"].to_numpy(); lon = master["lon"].to_numpy()
    d = (lat - lat0)**2 + (lon - lon0)**2
    idx = np.argsort(d)
    return idx[1: topn+1]  # exclude self

def find_neighbors(ids, master, X, target_index, topk=NEIGHBOR_TOPK, geo_n=GEO_NEAREST_N):
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

# ====================== NEW: 하이브리드 유사도 및 가중치 튜닝 ======================

def _safe_minmax(x: np.ndarray) -> np.ndarray:
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
    meta(임베딩 거리) + geo(위치 거리) + sem(POI 임베딩 코사인 유사도) 조합
    반환: (meta_sim, geo_sim, sem_sim, hybrid_w, order_idx)
    """
    ids = master["complex_id"].astype(str).tolist()
    t_id = ids[target_idx]
    t_lat, t_lon = float(master.iloc[target_idx]["lat"]), float(master.iloc[target_idx]["lon"])

    # --- meta sim (거리 → 유사도)
    t_vec = X_meta[target_idx:target_idx+1]       # [1, d]
    c_vec = X_meta[cand_idx]                      # [N, d]
    d_meta = np.sqrt(((c_vec - t_vec)**2).sum(axis=1))  # L2
    meta_sim = 1.0 / (1.0 + d_meta)
    meta_sim = _safe_minmax(meta_sim)

    # --- geo sim (좌표 거리 → 유사도)
    lat = master.iloc[cand_idx]["lat"].to_numpy(dtype=float)
    lon = master.iloc[cand_idx]["lon"].to_numpy(dtype=float)
    d_geo = (lat - t_lat)**2 + (lon - t_lon)**2
    geo_sim = 1.0 / (1.0 + d_geo / (np.nanmean(d_geo) + 1e-9))
    geo_sim = _safe_minmax(geo_sim)

    # --- sem sim (POI 슈퍼카테고리 임베딩 코사인)
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
        # cosine
        t_norm = np.linalg.norm(t_sc) + 1e-8
        p_norm = np.linalg.norm(pool_sc, axis=1) + 1e-8
        sem_sim = (pool_sc @ t_sc) / (t_norm * p_norm)
        sem_sim = _safe_minmax(sem_sim)
    else:
        sem_sim = np.zeros(len(cand_idx), dtype=np.float32)

    # --- 합성 가중치
    comb = w_meta*meta_sim + w_geo*geo_sim + w_sem*sem_sim
    comb = np.maximum(comb, 0)
    if comb.sum() <= 0:
        comb = np.ones_like(comb) / len(comb)
    else:
        comb = comb / comb.sum()

    order = np.argsort(-comb)  # 내림차순(가중치 큰 순)
    return meta_sim, geo_sim, sem_sim, comb[order], cand_idx[order]

def tune_hybrid_weights(
    monthly: pd.DataFrame,
    master: pd.DataFrame,
    months_all: pd.DatetimeIndex,
    ids: list,
    X_meta: np.ndarray,
    hot_ids: list,
    make_forecaster_fn,   # (w_meta, w_geo, w_sem) -> forecast_fn
    H: int,
    max_eval: int = 800,
    grid: Tuple[List[float], List[float], List[float]] = ([0.45,0.55,0.65], [0.15,0.25,0.35], [0.00,0.10,0.20]),
    seed: int = 42
) -> Tuple[float,float,float]:
    """HOT 일부로 w_meta/w_geo/w_sem 그리드 튜닝 → MAPE 최소 조합 리턴"""
    rng = np.random.RandomState(seed)
    pool = list(hot_ids)[:]
    if len(pool) > max_eval:
        pool = list(rng.choice(pool, size=max_eval, replace=False))
    best = (0.55,0.25,0.20); best_mape = 1e9
    for wm in grid[0]:
        for wg in grid[1]:
            for ws in grid[2]:
                if abs(wm+wg+ws - 1.0) > 1e-6:
                    continue
                f = make_forecaster_fn(wm, wg, ws)
                res = evaluate_ids(monthly, pool, f, months_all, H)
                mape = res.get("MAPE", np.nan)
                if np.isfinite(mape) and mape < best_mape:
                    best_mape = mape; best = (wm,wg,ws)
    return best

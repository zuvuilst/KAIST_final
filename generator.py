# generator.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 경제지표 모듈 (저장 없이 호출) ────────────────────────────────────────────
from econ_api import fetch_default_indicators  # 캐시는 ENV로 제어
from econ_correlate import (
    make_capital_ids_from_trades,
    aggregate_capital_price,
    align_and_corr,
)
# 선택: econ_bridge 가 있으면 가산형 보정에 활용
try:
    from econ_bridge import compute_econ_component  # Optional
except Exception:
    compute_econ_component = None

# 확산모델 래퍼(선택) : 없으면 폴백
try:
    import torch
    from models_diffusion import diff_RATD, diff_CSDI  # 선택적으로 제공되는 파일
    _DIFF_OK = True
except Exception:
    torch = None
    diff_RATD = diff_CSDI = None
    _DIFF_OK = False

# ── 컬럼 후보( dataset.py 스키마 반영 ) ──────────────────────────────────────
_ID_CANDS = ["complex_id", "apartment_complex_sequence", "apt_id", "danji_id", "aptcode", "apt_code", "id"]
_DATE_CANDS = ["ym", "date", "month", "contract_ym", "ymd"]
_PRICE_CANDS = ["ppsm_median", "ppsm", "price", "p", "p_sqm", "deal_price", "price_norm", "trade_price", "avg_price"]

# ── 유틸 ────────────────────────────────────────────────────────────────────
def _pick_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    if df is None or df.empty: return None
    lowered = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in lowered: return lowered[k]
    for c in df.columns:  # fuzzy
        lc = c.lower()
        for k in cands:
            if k in lc: return c
    return None

def _ensure_months(months: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(months, pd.DatetimeIndex):
        months = pd.to_datetime(months)
    return months.to_period("M").to_timestamp()

def _nanfill_fwd_back(x: np.ndarray) -> np.ndarray:
    y = x.astype("float64", copy=True)
    last = np.nan
    for i in range(len(y)):
        if np.isfinite(y[i]): last = y[i]
        elif np.isfinite(last): y[i] = last
    nxt = np.nan
    for i in range(len(y)-1, -1, -1):
        if np.isfinite(y[i]): nxt = y[i]
        elif np.isfinite(nxt): y[i] = nxt
    return y

def _smooth_diffusion(y: np.ndarray, passes: int = 2, lam: float = 0.15) -> np.ndarray:
    if y.size == 0: return y
    z = _nanfill_fwd_back(y.astype("float64"))
    lam = float(max(0.0, min(1.0, lam))); n = len(z)
    for _ in range(max(0, passes)):
        z2 = z.copy()
        for t in range(n):
            l = z[t-1] if t-1 >= 0 else z[t]
            r = z[t+1] if t+1 < n else z[t]
            neigh = 0.5*(l+r)
            if np.isfinite(neigh) and np.isfinite(z[t]):
                z2[t] = (1 - lam) * z[t] + lam * neigh
        z = z2
    return z

def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    win = int(max(1, win))
    if y.size == 0 or win == 1: return y
    z = _nanfill_fwd_back(y)
    c = np.convolve(z, np.ones(win, dtype="float64")/win, mode="same")
    return c

# ── 시계열 추출/이웃 행렬 ──────────────────────────────────────────────────
def _extract_series(monthly: pd.DataFrame, unit_id: str, months_ctx: pd.DatetimeIndex, agg: str = "median") -> pd.Series:
    months_ctx = _ensure_months(months_ctx)
    if monthly is None or monthly.empty:
        return pd.Series(index=months_ctx, dtype="float64")
    id_col = _pick_col(monthly, _ID_CANDS)
    dt_col = _pick_col(monthly, _DATE_CANDS)
    price_col = _pick_col(monthly, _PRICE_CANDS)
    if id_col is None or dt_col is None or price_col is None:
        return pd.Series(index=months_ctx, dtype="float64")
    df = monthly[[id_col, dt_col, price_col]].copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce").dt.to_period("M").to_timestamp()
    df = df[df[id_col].astype(str) == str(unit_id)]
    if df.empty:
        return pd.Series(index=months_ctx, dtype="float64")
    if agg == "median": s = df.groupby(dt_col)[price_col].median()
    elif agg == "mean": s = df.groupby(dt_col)[price_col].mean()
    else:               s = df.groupby(dt_col)[price_col].last()
    return s.sort_index().reindex(months_ctx).astype("float64")

def _neighbor_matrix(neigh_monthly: pd.DataFrame, neigh_ids: List[str], months_ctx: pd.DatetimeIndex, agg: str = "median") -> Tuple[np.ndarray, List[str]]:
    rows, ok = [], []
    for nid in neigh_ids:
        s = _extract_series(neigh_monthly, nid, months_ctx, agg=agg).to_numpy(dtype="float64")
        if np.isfinite(s).sum() == 0: continue
        rows.append(_nanfill_fwd_back(s)); ok.append(nid)
    if not rows: return np.zeros((0, len(months_ctx)), dtype="float64"), []
    return np.vstack(rows), ok  # (K,L), [ids]

# ── 옵션 ───────────────────────────────────────────────────────────────────
@dataclass
class GenOptions:
    trend_smooth_win: int = 3
    vol_match: bool = True
    add_noise_ratio: float = 0.03
    recency_half_life: Optional[int] = None  # 월 단위

    # Econ 공통
    econ_corr_threshold: float = 0.80
    econ_max_lag: int = 12
    econ_disable_cache: bool = True

    # econ_level_trend
    econ_alpha_level: float = 0.12
    econ_beta_trend: float = 0.20

    # econ_add (bridge)
    econ_add_strength: float = 0.25

# ── 이웃 최종 가중치(최근성 결합) ──────────────────────────────────────────
def _final_neighbor_weights(base_w: np.ndarray, neigh_ids: List[str], neigh_monthly: pd.DataFrame, months: pd.DatetimeIndex, half_life: Optional[int]) -> np.ndarray:
    w = np.asarray(base_w, dtype="float64")
    if w.ndim != 1 or w.size != len(neigh_ids):
        w = np.ones(len(neigh_ids), dtype="float64")
    w = np.maximum(w, 0)
    if half_life is None or half_life <= 0:
        s = w.sum(); return (w / s) if s > 0 else np.ones_like(w) / max(1, len(w))

    id_col = _pick_col(neigh_monthly, _ID_CANDS)
    dt_col = _pick_col(neigh_monthly, _DATE_CANDS)
    price_col = _pick_col(neigh_monthly, _PRICE_CANDS)
    if id_col and dt_col and price_col:
        P = neigh_monthly[[id_col, dt_col, price_col]].copy()
        P[dt_col] = pd.to_datetime(P[dt_col], errors="coerce").dt.to_period("M").to_timestamp()
        pivot = P.pivot_table(index=dt_col, columns=id_col, values=price_col).reindex(_ensure_months(months)).ffill().bfill()
        cutoff = _ensure_months(months)[-1]
        rec = []
        for nid in neigh_ids:
            col = str(nid)
            if col not in pivot.columns: rec.append(0.0); continue
            last = pivot[col].loc[:cutoff].last_valid_index()
            if last is None: rec.append(0.0); continue
            stale = max(0, (_ensure_months(pd.DatetimeIndex([cutoff]))[0].to_period("M") - pd.Period(last, "M")).n)
            lam = np.log(2) / max(1e-6, float(half_life))
            rec.append(float(np.exp(-lam * stale)))
        rec = np.asarray(rec, dtype="float64"); rec = rec / (rec.sum() + 1e-12)
        w = w * rec
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w) / max(1, len(w))

# ── 기본 블렌드 ────────────────────────────────────────────────────────────
def _blend_basic(M: np.ndarray, w: np.ndarray, T_in: int, opt: GenOptions) -> np.ndarray:
    M = M[:, -T_in:]
    ctx = (w @ M).astype("float64")
    ctx = _smooth_diffusion(ctx, passes=2, lam=0.15)
    if opt.trend_smooth_win and opt.trend_smooth_win > 1:
        ctx = 0.6 * ctx + 0.4 * _moving_average(ctx, opt.trend_smooth_win)
    return ctx.astype("float32")

def _blend_mean_nosmooth(M: np.ndarray, w: np.ndarray, T_in: int, opt: GenOptions) -> np.ndarray:
    M = M[:, -T_in:]; w = np.ones(M.shape[0], dtype="float64") / max(1, M.shape[0])
    return (w @ M).astype("float32")

def _blend_zero(M: np.ndarray, w: np.ndarray, T_in: int, opt: GenOptions) -> np.ndarray:
    return np.zeros((T_in,), dtype=np.float32)

# ── Econ 보정 ──────────────────────────────────────────────────────────────
def _apply_econ_add(ctx: np.ndarray, months: pd.DatetimeIndex, monthly_all: Optional[pd.DataFrame], trades_all: Optional[pd.DataFrame], T_in: int, opt: GenOptions) -> np.ndarray:
    if compute_econ_component is None or monthly_all is None or trades_all is None:
        return ctx
    comp = compute_econ_component(
        monthly=monthly_all,
        trades=trades_all,
        months_ctx=_ensure_months(months[-T_in:]),
        T_in=T_in,
        corr_threshold=float(opt.econ_corr_threshold),
        max_lag=int(opt.econ_max_lag),
        strength=float(opt.econ_add_strength),
        disable_cache=bool(opt.econ_disable_cache),
    )
    if comp is None or len(comp) != len(ctx): return ctx
    s_ctx = float(np.std(ctx) + 1e-6)
    return (ctx.astype("float64") + comp.astype("float64") * s_ctx).astype("float32")

def _apply_econ_level_trend(ctx: np.ndarray, months: pd.DatetimeIndex, monthly_all: Optional[pd.DataFrame], trades_all: Optional[pd.DataFrame], T_in: int, opt: GenOptions) -> np.ndarray:
    if monthly_all is None or trades_all is None: return ctx
    idx = _ensure_months(months[-T_in:])
    cap_ids = make_capital_ids_from_trades(trades_all)
    price_df = aggregate_capital_price(monthly_all, cap_ids)
    if price_df is None or price_df.empty: return ctx

    start, end = idx[0].strftime("%Y-%m"), idx[-1].strftime("%Y-%m")
    prev = os.getenv("ECON_DISABLE_CACHE", "")
    if opt.econ_disable_cache: os.environ["ECON_DISABLE_CACHE"] = "1"
    ind = fetch_default_indicators(start=start, end=end)
    if opt.econ_disable_cache:
        (os.environ.__setitem__("ECON_DISABLE_CACHE", prev) if prev
         else os.environ.pop("ECON_DISABLE_CACHE", None))
    if ind is None or ind.empty: return ctx

    corr_tbl = align_and_corr(price_df, ind, max_lag=int(opt.econ_max_lag), topk=9999)
    if corr_tbl is None or corr_tbl.empty: return ctx
    use = corr_tbl[np.abs(corr_tbl["best_r"]) >= float(opt.econ_corr_threshold)].copy()
    if use.empty: return ctx

    mats, wts = [], []
    for _, r in use.iterrows():
        col = str(r.get("indicator", "")).strip()
        if not col or col not in ind.columns: continue
        rep = str(r.get("repr", "level")).lower()
        lag = int(r.get("best_lag", 0) or 0)
        s = ind[col].reindex(idx).astype("float64")
        if rep == "pct": s = s.pct_change() * 100.0
        if lag != 0: s = s.shift(lag)
        sd = float(s.std())
        if not np.isfinite(sd) or sd < 1e-9: continue
        z = (s - float(s.mean())) / sd
        mats.append(z.to_numpy(dtype="float64"))
        wts.append(abs(float(r["best_r"])))
    if not mats: return ctx
    W = np.asarray(wts, float); W = W / (W.sum() + 1e-12)
    Z = np.stack(mats, axis=0)
    E = (W.reshape(1, -1) @ Z).reshape(-1)  # (T,)
    dE = np.diff(E, prepend=E[0])

    std_ctx = float(np.nanstd(ctx))
    ctx = (ctx.astype("float64") * (1.0 + float(opt.econ_alpha_level) * E))
    if std_ctx > 0 and float(opt.econ_beta_trend) != 0:
        ctx = ctx + float(opt.econ_beta_trend) * dE * std_ctx
    bad = ~np.isfinite(ctx)
    if bad.any():
        ctx = _nanfill_fwd_back(ctx)
    return ctx.astype("float32")

# ── 후처리(변동성 정합/노이즈) ─────────────────────────────────────────────
def _postprocess_ctx(ctx: np.ndarray, M: np.ndarray, opt: GenOptions) -> np.ndarray:
    out = ctx.astype("float64")
    if opt.vol_match if hasattr(opt, "vol_match") else True:
        neigh_std = np.nanmedian(np.nanstd(M, axis=1))
        ctx_std = float(np.nanstd(out))
        if np.isfinite(neigh_std) and neigh_std > 1e-6 and np.isfinite(ctx_std) and ctx_std > 1e-6:
            out *= neigh_std / ctx_std
    if opt.add_noise_ratio and opt.add_noise_ratio > 0:
        std = float(np.nanstd(out))
        if np.isfinite(std) and std > 0:
            out = out + np.random.normal(0.0, std * float(opt.add_noise_ratio), size=out.shape)
    if not np.isfinite(out).all():
        out = _nanfill_fwd_back(out)
        out = np.nan_to_num(out, nan=float(np.nanmean(out) if np.isfinite(out).any() else 0.0))
    return out.astype("float32")

# ── 확산 샘플링 도움함수(체크포인트 없으면 None) ───────────────────────────
def _load_ckpt(ckpt_path: str, model, device: str = "cpu") -> bool:
    if not (ckpt_path and os.path.exists(ckpt_path)): return False
    try:
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        return True
    except Exception:
        return False

def _stack_neighbors(neigh_monthly: pd.DataFrame, neigh_ids: List[str], months: pd.DatetimeIndex, col="ppsm_median") -> np.ndarray:
    K, L = len(neigh_ids), len(months)
    out = np.zeros((K, L), dtype=np.float32)
    id_col = _pick_col(neigh_monthly, _ID_CANDS)
    dt_col = _pick_col(neigh_monthly, _DATE_CANDS)
    price_col = _pick_col(neigh_monthly, _PRICE_CANDS) if col is None else col
    for i, nid in enumerate(neigh_ids):
        s = (neigh_monthly.loc[neigh_monthly[id_col].astype(str)==str(nid), [dt_col, price_col]]
             .drop_duplicates(dt_col).set_index(dt_col))
        s.index = pd.to_datetime(s.index, errors="coerce").to_period("M").to_timestamp()
        s = s.reindex(months)[price_col].astype(float)
        out[i,:] = s.fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=np.float32)
    return out

def _weighted_1d(neigh_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = (weights / (weights.sum() + 1e-12)).astype(np.float32)
    return (neigh_stack * w[:, None]).sum(axis=0).astype(np.float32)

def _sample_diffusion_radiff(neigh_monthly: pd.DataFrame, neigh_ids: List[str], weights: np.ndarray,
                             months: pd.DatetimeIndex, *, ckpt_path: Optional[str],
                             device: str = "cpu") -> Optional[np.ndarray]:
    if not (_DIFF_OK and diff_RATD and torch): return None
    K, L = len(neigh_ids), len(months)
    neigh_stack = _stack_neighbors(neigh_monthly, neigh_ids, months)  # (K,L)

    # 간단 cond/reference 구성 (프로덕션에서는 학습 시 사용한 특징에 맞춰야 함)
    side_dim, ref_dim = 8, 16
    wctx = _weighted_1d(neigh_stack, weights)         # (L,)
    wstd = (neigh_stack * (weights[:,None]**2)).std(axis=0)  # (L,)
    cond_base = np.stack([wctx, wstd] + [wctx*0]*(side_dim-2), axis=0)  # (S,L)
    cond_info = np.repeat(cond_base[None,:,None,:], K, axis=2)          # (1,S,K,L)
    reference = np.repeat(neigh_stack[:,:,None], ref_dim, axis=2)[None, ...]  # (1,K,L,ref_dim)
    x0 = np.zeros((1, 2, K, L), dtype=np.float32)
    mask = (~np.isnan(neigh_stack)).astype(np.float32)
    x0[:,0,:,:] = np.nan_to_num(neigh_stack, nan=0.0)
    x0[:,1,:,:] = mask

    config = dict(channels=64, layers=4, num_steps=1000,
                  diffusion_embedding_dim=128, side_dim=side_dim,
                  ref_size=ref_dim, h_size=32, nheads=4, is_linear=False)
    model = diff_RATD(config, inputdim=2, use_ref=True).to(device)

    if not _load_ckpt(ckpt_path or os.path.join("runs","radiff_ckpt.pt"), model, device=device):
        return None

    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x0).to(device)
        c_t = torch.from_numpy(cond_info).to(device)
        r_t = torch.from_numpy(reference).to(device)
        t_idx = torch.tensor([config["num_steps"]-1], dtype=torch.long, device=device)
        y = model(x_t, c_t, t_idx, r_t).squeeze(0).detach().cpu().numpy()  # (K,L)
        return _weighted_1d(y, weights)

def _sample_diffusion_csdi(neigh_monthly: pd.DataFrame, neigh_ids: List[str], weights: np.ndarray,
                           months: pd.DatetimeIndex, *, ckpt_path: Optional[str],
                           device: str = "cpu") -> Optional[np.ndarray]:
    if not (_DIFF_OK and diff_CSDI and torch): return None
    K, L = len(neigh_ids), len(months)
    neigh_stack = _stack_neighbors(neigh_monthly, neigh_ids, months)
    side_dim = 8
    wctx = _weighted_1d(neigh_stack, weights)
    wstd = (neigh_stack * (weights[:,None]**2)).std(axis=0)
    cond_base = np.stack([wctx, wstd] + [wctx*0]*(side_dim-2), axis=0)  # (S,L)
    cond_info = np.repeat(cond_base[None,:,None,:], K, axis=2)          # (1,S,K,L)
    x0 = np.zeros((1, 2, K, L), dtype=np.float32)
    mask = (~np.isnan(neigh_stack)).astype(np.float32)
    x0[:,0,:,:] = np.nan_to_num(neigh_stack, nan=0.0)
    x0[:,1,:,:] = mask

    config = dict(channels=64, layers=4, num_steps=1000,
                  diffusion_embedding_dim=128, side_dim=side_dim,
                  nheads=4, is_linear=False)
    model = diff_CSDI(config, inputdim=2).to(device)
    if not _load_ckpt(ckpt_path or os.path.join("runs","csdi_ckpt.pt"), model, device=device):
        return None

    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x0).to(device)
        c_t = torch.from_numpy(cond_info).to(device)
        t_idx = torch.tensor([config["num_steps"]-1], dtype=torch.long, device=device)
        y = model(x_t, c_t, t_idx).squeeze(0).detach().cpu().numpy()  # (K,L)
        return _weighted_1d(y, weights)

# ── 공개 API ───────────────────────────────────────────────────────────────
def build_synthetic_context(
    target_id: str,
    months: pd.DatetimeIndex,
    neigh_monthly: pd.DataFrame,
    neigh_ids: List[str],
    hybrid_weights: np.ndarray,
    T_in: int,
    *,
    # 기존 옵션(기본값 유지)
    trend_smooth_win: int = 3,
    vol_match: bool = True,
    add_noise_ratio: float = 0.03,
    recency_half_life: Optional[int] = None,
    monthly_all: Optional[pd.DataFrame] = None,
    trades_all: Optional[pd.DataFrame] = None,
    econ_corr_threshold: float = 0.80,
    econ_alpha_level: float = 0.12,
    econ_beta_trend: float = 0.20,
    econ_max_lag: int = 12,
    econ_disable_cache: bool = True,
    econ_add_strength: float = 0.25,
    # 신규: 제너레이터 선택 + 확산 체크포인트/디바이스
    gen_model: str = "econ_level_trend",
    diffusion_ckpt_radiff: Optional[str] = None,
    diffusion_ckpt_csdi: Optional[str] = None,
    diffusion_device: str = "cpu",
    # 호환: model 키로 들어오면 덮어쓰기
    **kwargs,
) -> np.ndarray:
    """
    합성 컨텍스트 생성:
      gen_model ∈ {"basic","econ_add","econ_level_trend","mean_nosmooth","zero","radiff","csdi"}
    - 확산(radaiff/csdi)은 체크포인트/의존성이 없으면 경제보정형으로 폴백.
    - main의 기존 호출과 호환(모델 미지정 시 'econ_level_trend').
    """
    if "model" in kwargs and kwargs["model"]:
        gen_model = str(kwargs["model"]).lower()

    months = _ensure_months(months)
    if len(months) < T_in:
        raise ValueError("months length is shorter than T_in")

    opt = GenOptions(
        trend_smooth_win=trend_smooth_win,
        vol_match=vol_match,
        add_noise_ratio=add_noise_ratio,
        recency_half_life=recency_half_life,
        econ_corr_threshold=econ_corr_threshold,
        econ_max_lag=econ_max_lag,
        econ_disable_cache=econ_disable_cache,
        econ_alpha_level=econ_alpha_level,
        econ_beta_trend=econ_beta_trend,
        econ_add_strength=econ_add_strength,
    )

    # 이웃 행렬 & 가중치
    M_full, ok_ids = _neighbor_matrix(neigh_monthly, neigh_ids, months)
    if M_full.shape[0] == 0:
        return np.zeros((T_in,), dtype=np.float32)
    M = M_full[:, -T_in:]
    w = _final_neighbor_weights(hybrid_weights, ok_ids, neigh_monthly, months[-T_in:], opt.recency_half_life)

    # 1) 확산 옵션 우선 시도
    gen_model = (gen_model or "econ_level_trend").lower()
    if gen_model in ("radiff", "csdi"):
        synth = None
        device = diffusion_device
        if gen_model == "radiff":
            synth = _sample_diffusion_radiff(
                neigh_monthly, ok_ids, w, months[-T_in:],
                ckpt_path=diffusion_ckpt_radiff, device=device
            )
        else:
            synth = _sample_diffusion_csdi(
                neigh_monthly, ok_ids, w, months[-T_in:],
                ckpt_path=diffusion_ckpt_csdi, device=device
            )
        if synth is not None and np.all(np.isfinite(synth)):
            ctx = synth.astype(np.float32)
            return _postprocess_ctx(ctx, M, opt)  # 확산 산출물도 동일 후처리 적용

        # 실패 시 econ_level_trend로 폴백
        gen_model = "econ_level_trend"

    # 2) 비확산 계열
    if gen_model == "zero":
        ctx = _blend_zero(M, w, T_in, opt)
    elif gen_model == "mean_nosmooth":
        ctx = _blend_mean_nosmooth(M, w, T_in, opt)
    else:
        # 기본 블렌드
        ctx = _blend_basic(M, w, T_in, opt)
        if gen_model == "econ_add":
            ctx = _apply_econ_add(ctx, months, monthly_all, trades_all, T_in, opt)
        elif gen_model == "econ_level_trend":
            ctx = _apply_econ_level_trend(ctx, months, monthly_all, trades_all, T_in, opt)

    # 공통 후처리
    return _postprocess_ctx(ctx, M, opt)

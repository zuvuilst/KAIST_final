# -*- coding: utf-8 -*-
"""
Generator: Synthetic past context (pseudo history) builders with pluggable models.
- 기본 이웃 혼합
- 경제지표 가산(bridge 기반)
- 경제지표 레벨×/추세+ 보정(econ_api+econ_correlate 기반)
- 최근성(recency) 가중치 옵션
- 변동성 정합, 소량 노이즈 옵션

사용법 예시
-----------
ctx = build_synthetic_context(
    model="econ_level_trend",            # "basic" | "econ_add" | "econ_level_trend" | "mean_nosmooth" | "zero"
    target_id=cid,
    months=months_ctx,                   # pd.DatetimeIndex (MS)
    neigh_monthly=neigh_monthly_df,      # 이웃 월별 시계열 (complex_id, ym, ppsm_median)
    neigh_ids=ids_kept,                  # 사용 이웃 id 리스트
    hybrid_weights=w_kept,               # 이웃 가중치 (메타×지리)
    T_in=T_IN,
    # 글로벌 DF (경제 보정에 필요)
    monthly_all=monthly_df,
    trades_all=trades_df,
    # 옵션
    recency_half_life=6,
    econ_corr_threshold=0.80,
    econ_alpha_level=0.12,
    econ_beta_trend=0.20,
    econ_max_lag=12,
    econ_disable_cache=True,
)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 경제지표 모듈: 사용자의 요청대로 명시 import ─────────────────────────────
from econ_api import fetch_default_indicators  # 저장 없음, 캐시는 env로 차단
from econ_correlate import (
    make_capital_ids_from_trades,
    aggregate_capital_price,
    align_and_corr,
)
# 선택적: bridge 방식(합성 성분 가산)
try:
    from econ_bridge import compute_econ_component  # Optional
except Exception:
    compute_econ_component = None  # 런타임에 없으면 미사용

# ── 컬럼 후보( dataset.py 스키마 반영 ) ──────────────────────────────────────
_ID_CANDS = [
    "complex_id", "apartment_complex_sequence", "apt_id", "danji_id", "aptcode", "apt_code", "id",
]
_DATE_CANDS = ["ym", "date", "month", "contract_ym", "ymd"]
_PRICE_CANDS = [
    "ppsm_median", "ppsm", "price", "p", "p_sqm", "deal_price", "price_norm", "trade_price", "avg_price",
]

# ── 유틸 ────────────────────────────────────────────────────────────────────

def _pick_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lowered = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in lowered:
            return lowered[k]
    for c in df.columns:  # fuzzy
        lc = c.lower()
        for k in cands:
            if k in lc:
                return c
    return None


def _ensure_months(months: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(months, pd.DatetimeIndex):
        months = pd.to_datetime(months)
    return months.to_period("M").to_timestamp()


def _nanfill_fwd_back(x: np.ndarray) -> np.ndarray:
    y = x.astype("float64", copy=True)
    last = np.nan
    for i in range(len(y)):
        if np.isfinite(y[i]):
            last = y[i]
        elif np.isfinite(last):
            y[i] = last
    nxt = np.nan
    for i in range(len(y) - 1, -1, -1):
        if np.isfinite(y[i]):
            nxt = y[i]
        elif np.isfinite(nxt):
            y[i] = nxt
    return y


def _smooth_diffusion(y: np.ndarray, passes: int = 2, lam: float = 0.15) -> np.ndarray:
    if y.size == 0:
        return y
    z = _nanfill_fwd_back(y.astype("float64"))
    n = len(z)
    lam = float(max(0.0, min(1.0, lam)))
    for _ in range(max(0, passes)):
        z2 = z.copy()
        for t in range(n):
            l = z[t - 1] if t - 1 >= 0 else z[t]
            r = z[t + 1] if t + 1 < n else z[t]
            neigh = 0.5 * (l + r)
            if np.isfinite(neigh) and np.isfinite(z[t]):
                z2[t] = (1 - lam) * z[t] + lam * neigh
        z = z2
    return z


def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    win = int(max(1, win))
    if y.size == 0 or win == 1:
        return y
    z = _nanfill_fwd_back(y)
    c = np.convolve(z, np.ones(win, dtype="float64") / win, mode="same")
    return c


# ── 이웃 시계열 행렬 ───────────────────────────────────────────────────────

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
    if agg == "median":
        s = df.groupby(dt_col)[price_col].median()
    elif agg == "mean":
        s = df.groupby(dt_col)[price_col].mean()
    else:
        s = df.groupby(dt_col)[price_col].last()
    return s.sort_index().reindex(months_ctx).astype("float64")


def _neighbor_matrix(neigh_monthly: pd.DataFrame, neigh_ids: List[str], months_ctx: pd.DatetimeIndex, agg: str = "median") -> Tuple[np.ndarray, List[str]]:
    rows = []
    ok: List[str] = []
    for nid in neigh_ids:
        s = _extract_series(neigh_monthly, nid, months_ctx, agg=agg)
        v = s.to_numpy(dtype="float64")
        if np.isfinite(v).sum() == 0:
            continue
        v = _nanfill_fwd_back(v)
        rows.append(v)
        ok.append(nid)
    if not rows:
        return np.zeros((0, len(months_ctx)), dtype="float64"), []
    return np.vstack(rows), ok


# ── 모델 인터페이스 ────────────────────────────────────────────────────────
@dataclass
class GenOptions:
    trend_smooth_win: int = 3
    vol_match: bool = True
    add_noise_ratio: float = 0.03
    recency_half_life: Optional[int] = None  # 월단위; None이면 사용 안함

    # Econ (공통)
    econ_corr_threshold: float = 0.80
    econ_max_lag: int = 12
    econ_disable_cache: bool = True

    # econ_level_trend
    econ_alpha_level: float = 0.12
    econ_beta_trend: float = 0.20

    # econ_add (bridge)
    econ_add_strength: float = 0.25


# ── 보조: 이웃 가중치 만들기(최근성 결합) ───────────────────────────────────

def _final_neighbor_weights(base_w: np.ndarray, neigh_ids: List[str], neigh_monthly: pd.DataFrame, months: pd.DatetimeIndex, half_life: Optional[int]) -> np.ndarray:
    w = np.asarray(base_w, dtype="float64")
    if w.ndim != 1 or w.size != len(neigh_ids):
        w = np.ones(len(neigh_ids), dtype="float64")
    w = np.maximum(w, 0)
    if half_life is None or half_life <= 0:
        s = w.sum()
        return (w / s) if s > 0 else np.ones_like(w) / max(1, len(w))

    # recency 가중치 계산
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
            if col not in pivot.columns:
                rec.append(0.0)
                continue
            last = pivot[col].loc[:cutoff].last_valid_index()
            if last is None:
                rec.append(0.0)
                continue
            stale = max(0, (_ensure_months(pd.DatetimeIndex([cutoff]))[0].to_period("M") - pd.Period(last, freq="M")).n)
            lam = np.log(2) / max(1e-6, float(half_life))
            rec.append(float(np.exp(-lam * stale)))
        rec = np.asarray(rec, dtype="float64")
        rec = rec / (rec.sum() + 1e-12)
        w = w * rec
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w) / max(1, len(w))


# ── 모델 구현들 ────────────────────────────────────────────────────────────

def _blend_basic(M: np.ndarray, w: np.ndarray, T_in: int, opt: GenOptions) -> np.ndarray:
    """이웃 혼합 + 확산평활 + 이동평균 블렌드."""
    M = M[:, -T_in:]
    ctx = (w @ M).astype("float64")  # (T,)
    ctx = _smooth_diffusion(ctx, passes=2, lam=0.15)
    if opt.trend_smooth_win and opt.trend_smooth_win > 1:
        ctx = 0.6 * ctx + 0.4 * _moving_average(ctx, opt.trend_smooth_win)
    return ctx.astype("float32")


def _blend_mean_nosmooth(M: np.ndarray, w: np.ndarray, T_in: int, opt: GenOptions) -> np.ndarray:
    M = M[:, -T_in:]
    w = np.ones(M.shape[0], dtype="float64") / max(1, M.shape[0])
    return (w @ M).astype("float32")


def _blend_zero(M: np.ndarray, w: np.ndarray, T_in: int, opt: GenOptions) -> np.ndarray:
    return np.zeros((T_in,), dtype=np.float32)


def _apply_econ_add(ctx: np.ndarray, months: pd.DatetimeIndex, monthly_all: Optional[pd.DataFrame], trades_all: Optional[pd.DataFrame], T_in: int, opt: GenOptions) -> np.ndarray:
    """econ_bridge 기반: 합성 성분을 ctx에 가산 (스케일 매칭)."""
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
    if comp is None or len(comp) != len(ctx):
        return ctx
    s_ctx = float(np.std(ctx) + 1e-6)
    return (ctx.astype("float64") + comp.astype("float64") * s_ctx).astype("float32")


def _apply_econ_level_trend(ctx: np.ndarray, months: pd.DatetimeIndex, monthly_all: Optional[pd.DataFrame], trades_all: Optional[pd.DataFrame], T_in: int, opt: GenOptions) -> np.ndarray:
    """econ_api + econ_correlate 기반: E_t(레벨), dE_t(추세) 보정."""
    if monthly_all is None or trades_all is None:
        return ctx
    idx = _ensure_months(months[-T_in:])

    # 수도권 가격 시계열
    cap_ids = make_capital_ids_from_trades(trades_all)
    price_df = aggregate_capital_price(monthly_all, cap_ids)
    if price_df is None or price_df.empty:
        return ctx

    # 지표 로딩(무저장, 캐시차단 가능)
    start, end = idx[0].strftime("%Y-%m"), idx[-1].strftime("%Y-%m")
    prev = os.getenv("ECON_DISABLE_CACHE", "")
    if opt.econ_disable_cache:
        os.environ["ECON_DISABLE_CACHE"] = "1"
    ind = fetch_default_indicators(start=start, end=end)
    if opt.econ_disable_cache:
        (os.environ.__setitem__("ECON_DISABLE_CACHE", prev) if prev else os.environ.pop("ECON_DISABLE_CACHE", None))
    if ind is None or ind.empty:
        return ctx

    # 상관/lag/표현 선정
    corr_tbl = align_and_corr(price_df, ind, max_lag=int(opt.econ_max_lag), topk=9999)
    if corr_tbl is None or corr_tbl.empty:
        return ctx
    use = corr_tbl[np.abs(corr_tbl["best_r"]) >= float(opt.econ_corr_threshold)].copy()
    if use.empty:
        return ctx

    # E_t, dE_t 구성
    mats, wts = [], []
    for _, r in use.iterrows():
        col = str(r.get("indicator", "")).strip()
        if not col or col not in ind.columns:
            continue
        rep = str(r.get("repr", "level")).lower()
        lag = int(r.get("best_lag", 0) or 0)
        s = ind[col].reindex(idx).astype("float64")
        if rep == "pct":
            s = s.pct_change() * 100.0
        if lag != 0:
            s = s.shift(lag)
        sd = float(s.std())
        if not np.isfinite(sd) or sd < 1e-9:
            continue
        z = (s - float(s.mean())) / sd
        mats.append(z.to_numpy(dtype="float64"))
        wts.append(abs(float(r["best_r"])) )
    if not mats:
        return ctx
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


# ── 공통 후처리(변동성 정합/노이즈) ─────────────────────────────────────────

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


# ── 메인 엔트리 ────────────────────────────────────────────────────────────

def build_synthetic_context(
    model: str,
    target_id: str,
    months: pd.DatetimeIndex,
    neigh_monthly: pd.DataFrame,
    neigh_ids: List[str],
    hybrid_weights: np.ndarray,
    T_in: int,
    *,
    monthly_all: Optional[pd.DataFrame] = None,
    trades_all: Optional[pd.DataFrame] = None,
    # 옵션 모음 (개별 인자로 넘겨도 되고, GenOptions 전달도 가능)
    trend_smooth_win: int = 3,
    vol_match: bool = True,
    add_noise_ratio: float = 0.03,
    recency_half_life: Optional[int] = None,
    econ_corr_threshold: float = 0.80,
    econ_max_lag: int = 12,
    econ_disable_cache: bool = True,
    econ_alpha_level: float = 0.12,
    econ_beta_trend: float = 0.20,
    econ_add_strength: float = 0.25,
) -> np.ndarray:
    """여러 synthetic context 모델을 공통 인터페이스로 제공.

    model:
      - "basic": 이웃 혼합 + 평활
      - "mean_nosmooth": 단순 평균
      - "zero": 0 컨텍스트
      - "econ_add": bridge 기반 가산 보정 (compute_econ_component 필요)
      - "econ_level_trend": econ_api+econ_correlate 기반 레벨×/추세+ 보정
    """
    months = _ensure_months(months)
    if len(months) < T_in:
        raise ValueError("months length is shorter than T_in")

    # 옵션 객체 생성
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

    # 모델 분기
    model = (model or "basic").lower()
    if model == "zero":
        ctx = _blend_zero(M, w, T_in, opt)
    elif model == "mean_nosmooth":
        ctx = _blend_mean_nosmooth(M, w, T_in, opt)
    else:
        ctx = _blend_basic(M, w, T_in, opt)
        if model == "econ_add":
            ctx = _apply_econ_add(ctx, months, monthly_all, trades_all, T_in, opt)
        elif model == "econ_level_trend":
            ctx = _apply_econ_level_trend(ctx, months, monthly_all, trades_all, T_in, opt)

    # 공통 후처리
    ctx = _postprocess_ctx(ctx, M, opt)
    return ctx

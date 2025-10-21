# generator.py
import numpy as np
import pandas as pd
from typing import Optional
# (옵션) nowcast 보정시 temporal model을 이용할 수 있게 import
from temporal_models import forecast_with_model

def weighted_series(neigh_df: pd.DataFrame, weights: np.ndarray, months: pd.DatetimeIndex):
    pivot = neigh_df.pivot_table(index="ym", columns="complex_id", values="ppsm_median")
    pivot = pivot.reindex(months).ffill().bfill()
    M = pivot.values  # [T, N]
    w = weights.reshape(1,-1)
    mu = np.nansum(np.where(np.isnan(M), 0, M) * w, axis=1)
    var = np.nansum(np.where(np.isnan(M), 0, (M - mu.reshape(-1,1))**2) * w, axis=1)
    std = np.sqrt(np.maximum(var, 0))
    q10 = mu - 1.2816*std
    q90 = mu + 1.2816*std
    return mu, q10, q90

def make_synthetic_history_and_forecast(
    target_id: str,
    months: pd.DatetimeIndex,
    neigh_monthly: pd.DataFrame,
    neigh_ids: list,
    sim_weights: np.ndarray,
    geo_weights: np.ndarray,
    alpha_sim_vs_geo: float = 0.6,
    history_T: int = 12,
    forecast_H: int = 6
):
    w = alpha_sim_vs_geo*sim_weights + (1-alpha_sim_vs_geo)*geo_weights
    w = w / (w.sum()+1e-12)

    if len(months) == 0:
        raise ValueError("months index is empty")
    last = months.max()
    hist_months = pd.date_range(end=last, periods=history_T, freq="MS")
    mu, q10, q90 = weighted_series(
        neigh_monthly.query("complex_id in @neigh_ids"), w, hist_months
    )
    hist_df = pd.DataFrame({
        "ym": hist_months, "ppsm_syn": mu, "q10": q10, "q90": q90
    })

    y = np.log(np.clip(mu, 1e-3, None))
    t = np.arange(len(y))
    A = np.vstack([t, np.ones_like(t)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef[0], coef[1]
    fut_months = pd.date_range(start=last + pd.offsets.MonthBegin(1), periods=forecast_H, freq="MS")
    t_fut = np.arange(len(y), len(y)+forecast_H)
    y_f = slope*t_fut + intercept
    mu_f = np.exp(y_f)

    last_std = float(np.nanstd(mu - np.exp(slope*t + intercept)))
    q10_f = mu_f - 1.2816*last_std
    q90_f = mu_f + 1.2816*last_std

    fut_df = pd.DataFrame({"ym": fut_months, "ppsm_forecast": mu_f, "q10": q10_f, "q90": q90_f})
    return hist_df, fut_df

# ================== 개선: 합성 컨텍스트 (최근성 가중치 + nowcasting 훅) ==================

def build_synthetic_context(
    target_id: str,
    months: pd.DatetimeIndex,
    neigh_monthly: pd.DataFrame,
    neigh_ids: list,
    hybrid_weights: np.ndarray,
    T_in: int = 12,
    trend_smooth_win: int = 3,
    vol_match: bool = True,
    add_noise_ratio: float = 0.03,
    # NEW ▼
    nowcast_model=None,                 # 전역 temporal model로 이웃 시계열 컷오프까지 보정하고 싶을 때 사용
    recency_half_life: int = 6,         # 최근성 half-life (개월)
):
    """
    이웃 시계열을 컷오프 시점까지 nowcast + 최근성(신선도) 지수감쇠 가중치로 신뢰도 반영.
    최종 가중치 = 유사도(hybrid) × 최근성(recency)
    """
    # 예측 시작 직전 월(컨텍스트의 마지막 시점)
    cutoff = months[-1 - 0]  # months는 보통 전체 기간, 첫 예측의 바로 이전을 컨텍스트 끝으로 사용

    pivot = neigh_monthly.pivot_table(index="ym", columns="complex_id", values="ppsm_median")
    pivot = pivot.reindex(months).ffill().bfill()

    # --- 최근성(신선도) 가중치 ---
    rec_w = []
    for cid in neigh_ids:
        series = pivot[cid].copy()
        last_obs = series.loc[:cutoff].last_valid_index()
        if last_obs is None:
            rec_w.append(0.0); continue
        staleness = max(0, (cutoff.to_period("M") - last_obs.to_period("M")).n)
        lam = np.log(2) / max(1e-6, recency_half_life)
        rw = float(np.exp(-lam * staleness))
        rec_w.append(rw)

        # --- (옵션) nowcasting 훅 ---
        # 실제로 nowcast하려면 아래 주석처럼 series의 마지막 관측 이후 부분을
        # forecast_with_model로 메우면 됩니다(연산량 고려해 선택).
        # if nowcast_model is not None and staleness > 0:
        #     # 필요 시 series를 보정해 pivot에 되돌려놓을 수 있음.
        #     pass

    rec_w = np.asarray(rec_w, dtype=np.float32)
    rec_w = rec_w / (rec_w.sum() + 1e-12)

    # --- 최종 가중치 = hybrid × recency ---
    w = (hybrid_weights.reshape(-1) * rec_w)
    w = w / (w.sum() + 1e-12)

    # --- 가중 평균으로 컨텍스트 생성 ---
    M = pivot[neigh_ids].to_numpy(dtype=float)  # [T, N]
    mu = np.nansum(np.where(np.isnan(M), 0, M) * w.reshape(1,-1), axis=1)  # [T]
    ctx = mu[-T_in:].copy()

    # 이동평균으로 부드럽게
    if trend_smooth_win and trend_smooth_win > 1:
        k = trend_smooth_win
        kernel = np.ones(k)/k
        ctx = np.convolve(ctx, kernel, mode="same")

    # 변동성 매칭
    if vol_match:
        tail = mu[-T_in:]
        s_ref = tail.std() + 1e-6
        s_ctx = ctx.std() + 1e-6
        if s_ctx > 0:
            ctx = ctx.mean() + (ctx - ctx.mean()) * (s_ref / s_ctx)

    # 약간의 노이즈
    if add_noise_ratio and add_noise_ratio > 0:
        ctx = ctx * (1.0 + add_noise_ratio * np.random.randn(*ctx.shape).astype(np.float32))

    return ctx.astype(np.float32)

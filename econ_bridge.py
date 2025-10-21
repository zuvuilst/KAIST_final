# econ_bridge.py
from __future__ import annotations
from typing import Optional, Tuple, List
import os
import numpy as np
import pandas as pd

# 재사용할 함수들 가져오기
from econ_api import fetch_default_indicators  # 지표 수집 (CSV 저장은 __main__에서만) :contentReference[oaicite:1]{index=1}
from econ_correlate import (
    make_capital_ids_from_trades,   # 수도권 complex_id 뽑기 :contentReference[oaicite:2]{index=2}
    aggregate_capital_price,        # 수도권 월별 중앙값(m²당) 시계열 :contentReference[oaicite:3]{index=3}
    align_and_corr,                 # 레벨/증감률 + ±lag 상관탐색 :contentReference[oaicite:4]{index=4}
)

def compute_econ_component(
    monthly: pd.DataFrame,
    trades: pd.DataFrame,
    months_ctx: pd.DatetimeIndex,
    T_in: int,
    corr_threshold: float = 0.80,
    max_lag: int = 12,
    strength: float = 0.25,   # 합성 성분 가중치(컨텍스트 표준편차 대비)
    disable_cache: bool = True,
) -> Optional[np.ndarray]:
    """
    컨텍스트 마지막 시점까지의 월축(months_ctx)에 맞춰, 상관≥threshold 지표들의
    (표현: level or pct, 최적 lag 적용) 표준화 합을 만든 뒤, 컨텍스트 스케일에 맞춰 반환.
    저장 부작용(CSV) 없이 동작. 길이=T_in, dtype=float32.
    """
    if len(months_ctx) < T_in:
        return None

    # 1) 수도권 가격 시계열 (타깃)
    capital_ids = make_capital_ids_from_trades(trades)
    price_df = aggregate_capital_price(monthly, capital_ids)  # index=ym, col=ppsm_capital :contentReference[oaicite:5]{index=5}
    if price_df.empty:
        return None

    # 2) 기간 맞춰 경제지표 수집 (CSV 저장 X, 캐시는 옵션)
    start = months_ctx[0].strftime("%Y-%m")
    end   = months_ctx[-1].strftime("%Y-%m")

    # 캐시 완전 비활성화가 필요하면 임시 환경변수로 스킵
    prev_disable = os.getenv("ECON_DISABLE_CACHE", "")
    if disable_cache:
        os.environ["ECON_DISABLE_CACHE"] = "1"  # econ_api에서 이 신호를 인식해 캐시 RW를 건너뛰게 함(아래 패치 참조)

    ind_df = fetch_default_indicators(start=start, end=end)   # index=ym, 여러 indicator cols :contentReference[oaicite:6]{index=6}

    if disable_cache:
        os.environ["ECON_DISABLE_CACHE"] = prev_disable

    if ind_df.empty:
        return None

    # 3) 상관계수/lag/표현(level/pct) 선정
    corr_df = align_and_corr(price_df, ind_df, max_lag=max_lag, topk=9999)  # 정렬된 테이블 반환 :contentReference[oaicite:7]{index=7}
    if corr_df is None or corr_df.empty:
        return None
    winners = corr_df[np.abs(corr_df["best_r"]) >= float(corr_threshold)].copy()
    if winners.empty:
        return None

    # 4) 컨텍스트 구간으로 재정렬
    idx = pd.date_range(start=months_ctx[0], end=months_ctx[-1], freq="MS")
    price_aligned = price_df.reindex(idx)["ppsm_capital"].astype(float)
    if price_aligned.dropna().empty:
        return None

    # 5) 보정 성분 구성: 각 지표를 (표현/lag 적용) z-score 후, r의 부호로 정렬 → 평균
    base_lvl = price_aligned.values  # 표준화 스케일용
    comp_list: List[np.ndarray] = []
    for _, row in winners.iterrows():
        col = row["indicator"]
        repr_kind = row["repr"]       # "level" 또는 "pct"
        lag = int(row["best_lag"])
        r = float(row["best_r"])

        s = ind_df[col].reindex(idx).astype(float)
        if s.dropna().empty:
            continue
        if repr_kind == "pct":
            s = s.pct_change() * 100.0
        # lag 적용: lag>0 이면 "지표 선행" → 지표를 +lag 만큼 과거로 당겨서 가격과 정렬
        if lag > 0:
            s = s.shift(lag)
        elif lag < 0:
            s = s.shift(lag)  # 가격이 선행이면 지표를 미래로 밀어 배열을 맞춤

        v = s.to_numpy(dtype="float64")
        mask = np.isfinite(v)
        if mask.sum() < max(6, T_in // 2):
            continue
        v = (v - np.nanmean(v[mask])) / (np.nanstd(v[mask]) + 1e-6)
        v = np.nan_to_num(v, nan=0.0)
        # 상관부호에 맞춰 방향 통일
        sign = 1.0 if r >= 0 else -1.0
        comp_list.append(sign * v.astype("float32"))

    if not comp_list:
        return None

    comp = np.mean(np.vstack(comp_list), axis=0)  # [len(idx)]
    comp = comp[-T_in:]  # 최근 T_in만
    if comp.shape[0] != T_in:
        return None

    # 6) 컨텍스트 스케일로 매칭
    # 합성 컨텍스트의 변동성 기준이 "최근 T_in" 구간이라서, 그 표준편차에 맞춰 축척
    # 이 함수 자체에서는 ctx 표준편차를 모르므로 comp를 자체 표준편차로 정규화 → strength로 가중
    std_comp = float(np.std(comp) + 1e-6)
    comp = (comp / std_comp) * float(strength)   # 상대적 비중만 반영, 절대 스케일은 호출측에서 맞춤
    return comp.astype("float32")

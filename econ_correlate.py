# econ_correlate.py
from __future__ import annotations
from typing import Tuple
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from utils.config import OUTDIR
from econ_api import fetch_default_indicators

RUNS = Path(OUTDIR) / "runs"
RUNS.mkdir(parents=True, exist_ok=True)

# 수도권 시/도명 키워드
SEOUL_KEYS   = {"서울특별시", "서울"}
INCHEON_KEYS = {"인천광역시", "인천"}
GG_KEYS      = {"경기도", "경기"}

# ----------------------------
# 로컬 테이블 로딩 & 정규화
# ----------------------------
def _normalize_month_index(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df[col] = df[col].dt.to_period("M").dt.to_timestamp()
    return df

def load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master  = pd.read_csv(Path(OUTDIR) / "master.csv",  low_memory=False)
    monthly = pd.read_csv(Path(OUTDIR) / "monthly.csv", low_memory=False)
    trades  = pd.read_csv(Path(OUTDIR) / "trades.csv",  low_memory=False)

    # monthly: ym 필수
    if "ym" not in monthly.columns:
        raise KeyError("[econ_corr] monthly.csv에 'ym' 컬럼이 없습니다.")
    monthly = _normalize_month_index(monthly, "ym")

    # trades: ymd 또는 ym
    if "ymd" in trades.columns:
        trades = _normalize_month_index(trades, "ymd")
        trades["ym"] = trades["ymd"]
    elif "ym" in trades.columns:
        trades = _normalize_month_index(trades, "ym")
    else:
        raise KeyError("[econ_corr] trades.csv에 'ymd' 또는 'ym' 컬럼이 없습니다.")

    # monthly 최소 컬럼 체크
    need = {"complex_id", "ym", "ppsm_median"}
    if not need.issubset(monthly.columns):
        raise KeyError(f"[econ_corr] monthly.csv에 필요한 컬럼이 없습니다: {sorted(list(need))}")
    return master, monthly, trades

# ----------------------------
# 수도권 단지군 구성 & 가격 집계
# ----------------------------
def make_capital_ids_from_trades(trades: pd.DataFrame) -> pd.Series:
    if not {"complex_id", "광역시도 명"}.issubset(trades.columns):
        raise KeyError("[econ_corr] trades.csv에 'complex_id','광역시도 명' 컬럼이 필요합니다.")

    mode_sido = (
        trades.dropna(subset=["complex_id", "광역시도 명"])
              .groupby("complex_id")["광역시도 명"]
              .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
              .dropna()
    )
    is_capital = mode_sido.isin(SEOUL_KEYS | INCHEON_KEYS | GG_KEYS)
    capital_ids = mode_sido[is_capital].index.astype(str).unique().tolist()
    return pd.Series(capital_ids)

def aggregate_capital_price(monthly: pd.DataFrame, capital_ids: pd.Series) -> pd.DataFrame:
    sub = monthly[monthly["complex_id"].astype(str).isin(set(capital_ids))]
    if sub.empty:
        raise RuntimeError("[econ_corr] 수도권 complex_id 월별 가격 데이터가 비었습니다.")
    g = sub.groupby("ym")["ppsm_median"].median().rename("ppsm_capital")
    df = g.to_frame().sort_index()
    df.index.name = "ym"
    return df

# ----------------------------
# 상관 분석 (레벨/증감률, ±lag)
# ----------------------------
def align_and_corr(price_df: pd.DataFrame, ind_df: pd.DataFrame, max_lag: int = 12, topk: int = 50) -> pd.DataFrame:
    # 월 정렬/맞춤
    idx = pd.date_range(
        start=max(price_df.index.min(), ind_df.index.min()),
        end=min(price_df.index.max(), ind_df.index.max()),
        freq="MS",
    )
    price = price_df.reindex(idx)
    inds  = ind_df.reindex(idx)
    df    = pd.concat([price, inds], axis=1)

    # 타깃 확인
    if df["ppsm_capital"].dropna().empty:
        raise RuntimeError("[econ_corr] ppsm_capital 시계열이 비었습니다.")

    results = []
    econ_cols = [c for c in inds.columns if c not in price.columns]
    base_lvl = df["ppsm_capital"].astype("float64")
    base_pct = base_lvl.pct_change() * 100.0

    def _corr(a: pd.Series, b: pd.Series):
        a = a.astype("float64"); b = b.astype("float64")
        mask = np.isfinite(a) & np.isfinite(b)
        n = int(mask.sum())
        if n < 6: 
            return np.nan, n
        try:
            r = float(np.corrcoef(a[mask], b[mask])[0, 1])
        except Exception:
            r = np.nan
        return r, n

    for col in econ_cols:
        s_lvl = df[col].astype("float64")
        if s_lvl.dropna().empty:
            continue
        s_pct = s_lvl.pct_change() * 100.0

        best = []  # (repr, r, lag, n, r0)
        for repr_name, x, y in [("level", base_lvl, s_lvl), ("pct", base_pct, s_pct)]:
            # 0-lag 기준값
            r0, _ = _corr(x, y)
            # ±max_lag 검색
            best_r, best_lag, best_n = np.nan, 0, 0
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    r, n = _corr(x, y)
                elif lag > 0:
                    r, n = _corr(x.iloc[lag:], y.iloc[:-lag])
                else:
                    r, n = _corr(x.iloc[:lag], y.iloc[-lag:])
                if np.isfinite(r) and (not np.isfinite(best_r) or abs(r) > abs(best_r)):
                    best_r, best_lag, best_n = r, lag, n
            if np.isfinite(best_r):
                best.append((repr_name, best_r, best_lag, best_n, r0))

        if not best:
            continue
        # 절댓값 큰 r, 표본수 큰 순
        best.sort(key=lambda z: (abs(z[1]), z[3]))
        expr, br, blag, bn, r0 = best[-1]
        results.append({
            "indicator": col,
            "repr": expr,           # level / pct
            "best_r": br,
            "best_lag": blag,       # +이면 지표가 선행(지표→가격), -이면 가격이 선행
            "n_obs": bn,
            "r_at_0": r0 if np.isfinite(r0) else np.nan,
        })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values("best_r", key=lambda s: s.abs(), ascending=False)
    if topk and topk > 0:
        out = out.head(topk)
    return out.reset_index(drop=True)

# ----------------------------
# 메인
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="수도권 가격 vs 경제지표 상관분석 (레벨/증감률, ±래그)")
    ap.add_argument("--max-lag", type=int, default=12, help="래그 탐색 범위 (±개월)")
    ap.add_argument("--topk", type=int, default=50, help="상위 N개 출력/저장")
    args = ap.parse_args()

    print(f"[econ_corr] OUTDIR={OUTDIR}")
    master, monthly, trades = load_tables()

    # 수도권 단지군 구성 → 월별 중앙값(m²당)
    capital_ids = make_capital_ids_from_trades(trades)
    price_df = aggregate_capital_price(monthly, capital_ids)
    if price_df.empty or price_df.index.min() is pd.NaT:
        raise RuntimeError("[econ_corr] 수도권 가격 시계열이 비어있거나 날짜 파싱 실패.")

    start = price_df.index.min().strftime("%Y-%m")
    end   = price_df.index.max().strftime("%Y-%m")
    print(f"[econ_corr] 분석 구간: {start} ~ {end}")

    # 경제지표 (R-ONE + ECOS) 자동 수집
    ind_df = fetch_default_indicators(start=start, end=end)
    if ind_df.empty:
        print("[econ_corr] 경제지표 DF가 비었습니다. (API 키/엔드포인트/기간 확인 필요)")
        price_df.to_csv(RUNS / "capital_price_series.csv", encoding="utf-8-sig")
        return
    print(f"[econ_corr] 지표 수집 완료: cols={len(ind_df.columns)}, rows={len(ind_df)}")

    # 상관계수 계산
    corr_df = align_and_corr(price_df, ind_df, max_lag=args.max_lag, topk=args.topk)

    # 저장
    price_df.to_csv(RUNS / "capital_price_series.csv", encoding="utf-8-sig")
    ind_df.to_csv(RUNS / "econ_indicators_raw.csv",   encoding="utf-8-sig")
    if not corr_df.empty:
        corr_df.to_csv(RUNS / "econ_indicator_correlations.csv", index=False, encoding="utf-8-sig")

    # 콘솔 표시
    print("\n[econ_corr] (상관계수 절대값 기준 정렬, 상위 N)")
    if not corr_df.empty:
        print(corr_df.to_string(index=False))
    else:
        print("(상관계를 계산할 수 있는 유효한 지표가 없습니다.)")

if __name__ == "__main__":
    main()

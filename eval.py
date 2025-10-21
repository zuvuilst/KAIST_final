# eval.py
# Python 3.8 호환 / tqdm 진행률(옵션) / 기존 evaluate_ids 백워드 호환 포함

from typing import Iterable, Callable, Dict, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    # tqdm이 없어도 동작하도록 더미 정의
    def tqdm(x, **kwargs):
        return x


# ────────────────────────────────────────────────────────────
# 기본 지표
# ────────────────────────────────────────────────────────────
def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    err = y_pred[mask] - y_true[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-6)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)


# ────────────────────────────────────────────────────────────
# NOW/FUTURE 평가 (전역 마지막 시점 정의 반영)
# ────────────────────────────────────────────────────────────
def evaluate_now_at_global_last(
    monthly: pd.DataFrame,
    ids: Iterable[str],
    forecast_fn: Callable[[pd.DataFrame, str, pd.DatetimeIndex], np.ndarray],
    months_all: pd.DatetimeIndex,
    desc: str = "Eval NOW"
) -> Dict[str, float]:
    """
    NOW = months_all[-1]
    컨텍스트 = months_all[:-1]
    - 전역 NOW 시점(now_m)에 '실제 관측'이 존재하는 단지에 대해서만 평가
    - forecast_fn은 H=1 예측을 반환해야 함 (길이 1의 배열)
    """
    assert len(months_all) >= 2, "months_all 길이가 너무 짧습니다."
    now_m = months_all[-1]
    ctx_m = months_all[:-1]

    maes, rmses, mapes = [], [], []
    it = tqdm(list(ids), desc=desc, dynamic_ncols=True, mininterval=0.2)
    for cid in it:
        # 대상 단지의 시계열 중 now_m 관측이 실제로 있는지 확인
        sub = (
            monthly.loc[monthly["complex_id"].astype(str) == str(cid), ["ym", "ppsm_median"]]
            .drop_duplicates("ym")
            .set_index("ym")
        )
        if now_m not in sub.index:
            continue
        gt = float(sub.loc[now_m, "ppsm_median"])
        if not np.isfinite(gt):
            continue

        # 예측(H=1)
        try:
            pred_arr = forecast_fn(monthly, str(cid), ctx_m)
        except Exception:
            # 실패 시 진행만 계속
            continue

        if pred_arr is None or len(pred_arr) < 1 or not np.isfinite(pred_arr[0]):
            continue

        pred = float(pred_arr[0])
        m, r = mae_rmse(np.array([gt], float), np.array([pred], float))
        p = mape(np.array([gt], float), np.array([pred], float))
        maes.append(m); rmses.append(r); mapes.append(p)

        if len(maes) % 50 == 0:
            it.set_postfix(MAE=float(np.nanmean(maes)), MAPE=float(np.nanmean(mapes)))

    def _agg(v):
        v = np.asarray(v, float)
        return float(np.nanmean(v)) if len(v) else float("nan")
    return {"MAE": _agg(maes), "RMSE": _agg(rmses), "MAPE": _agg(mapes)}


def evaluate_future_from_last(
    monthly: pd.DataFrame,
    ids: Iterable[str],
    forecast_fn: Callable[[pd.DataFrame, str, pd.DatetimeIndex], np.ndarray],
    months_all: pd.DatetimeIndex,
    H: int,
    desc: str = "Eval FUTURE"
) -> Dict[str, float]:
    """
    FUTURE(H):
      - 타깃 = months_all[-H:] (마지막 H개월)
      - 컨텍스트 = months_all[:-H] 까지만 제공
      - forecast_fn은 길이 H의 예측 배열을 반환해야 함
    """
    assert H >= 1 and len(months_all) > H, "months_all 길이나 H가 부적절합니다."
    ctx_m = months_all[:-H]

    maes, rmses, mapes = [], [], []
    it = tqdm(list(ids), desc=desc, dynamic_ncols=True, mininterval=0.2)
    for cid in it:
        # GT = 마지막 H개월 (reindex + ffill/bfill)
        sub = (
            monthly.loc[monthly["complex_id"].astype(str) == str(cid), ["ym", "ppsm_median"]]
            .drop_duplicates("ym")
            .set_index("ym")
            .reindex(months_all)["ppsm_median"]
            .ffill().bfill()
        )
        s = sub.to_numpy(dtype=float)
        if len(s) < H:
            continue
        gt = s[-H:]

        try:
            pred = forecast_fn(monthly, str(cid), ctx_m)
        except Exception:
            continue

        if pred is None or len(pred) != H or not np.all(np.isfinite(pred)):
            continue

        m, r = mae_rmse(gt, pred)
        p = mape(gt, pred)
        maes.append(m); rmses.append(r); mapes.append(p)

        if len(maes) % 50 == 0:
            it.set_postfix(MAE=float(np.nanmean(maes)), MAPE=float(np.nanmean(mapes)))

    def _agg(v):
        v = np.asarray(v, float)
        return float(np.nanmean(v)) if len(v) else float("nan")
    return {"MAE": _agg(maes), "RMSE": _agg(rmses), "MAPE": _agg(mapes)}


# ────────────────────────────────────────────────────────────
# 백워드 호환: 기존 evaluate_ids (다른 모듈에서 import 하더라도 깨지지 않게)
#  - 기본 정의: 마지막 H 스텝 예측 (컨텍스트=전체 months_all) 기반 평균 지표
#  - main에서 새 함수들을 쓰더라도 retrieval.tune_hybrid_weights 같은 곳과의 호환용
# ────────────────────────────────────────────────────────────
def _series(monthly: pd.DataFrame, cid, months: pd.DatetimeIndex) -> np.ndarray:
    s = (
        monthly.loc[monthly["complex_id"].astype(str) == str(cid), ["ym", "ppsm_median"]]
        .drop_duplicates(subset=["ym"]).set_index("ym").reindex(months)["ppsm_median"]
        .ffill().bfill().astype(float).values
    )
    return s


def evaluate_ids(
    monthly: pd.DataFrame,
    ids: Iterable,
    forecast_fn: Callable[[pd.DataFrame, str, pd.DatetimeIndex], np.ndarray],
    months: pd.DatetimeIndex,
    H: int,
    progress: bool = False,   # 호환용 옵션
    desc: str = ""            # 호환용 옵션
) -> Dict[str, float]:
    """
    (레거시) months 전체를 컨텍스트로 넘겨 forecast_fn을 호출하고,
    마지막 H개 구간과 비교. 새 파이프라인에선 evaluate_now_at_global_last /
    evaluate_future_from_last 사용을 권장.
    """
    maes, rmses, mapes = [], [], []
    iterator = tqdm(list(ids), desc=desc, dynamic_ncols=True, mininterval=0.2) if progress else ids
    for cid in iterator:
        s = _series(monthly, cid, months)
        if len(s) < H:
            continue
        gt = s[-H:]
        try:
            pred = forecast_fn(monthly, str(cid), months)
        except Exception:
            continue
        if pred is None or len(pred) != H:
            continue
        m, r = mae_rmse(gt, pred)
        p = mape(gt, pred)
        maes.append(m); rmses.append(r); mapes.append(p)
    def _agg(v):
        v = np.asarray(v, float)
        return float(np.nanmean(v)) if len(v) else float("nan")
    return {"MAE": _agg(maes), "RMSE": _agg(rmses), "MAPE": _agg(mapes)}


# ────────────────────────────────────────────────────────────
# 파일 저장 유틸
# ────────────────────────────────────────────────────────────
def to_markdown_table(df: pd.DataFrame, path: str):
    text = df.to_markdown(index=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# main_quick.py
# 작은 데이터/짧은 에폭으로 스모크 테스트만 빠르게 확인하는 전용 스크립트
# - viz 의존/카카오 호출/그래프 시각화 없음
# - TRUE-COLD 제거, pseudo-COLD = HOT 전체
# - 기본값: seed=42, epochs=2, subset_n=300, months_tail=36, no_viz=True
from __future__ import annotations
import os
import argparse
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Iterable, List

from config import OUTDIR, SEED, T_IN, H_OUT, TEMP_MODEL, DEVICE, AMP_DTYPE, MIXED_PRECISION
from retrieval import meta_embed, find_neighbors, scale_align, neighbor_weights_hybrid
from temporal_models import train_temporal, forecast_with_model
from eval import evaluate_ids
from generator import build_synthetic_context

pd.options.mode.copy_on_write = True
os.makedirs(OUTDIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
# Ultra-simple progress
# ────────────────────────────────────────────────────────────
class SimpleProgress:
    def __init__(self, total_units: int):
        self.total = max(1, int(total_units))
        self.cur = 0
        self._last_len = 0
        self._t0 = None

    def _show(self, msg: str, end: str = "\r"):
        pct = int(round(100.0 * self.cur / self.total))
        bar_len = 28
        filled = int(round(pct * bar_len / 100))
        bar = "#" * filled + "." * (bar_len - filled)
        line = f"[{bar}] {pct:3d}%  {msg}"
        pad = " " * max(0, self._last_len - len(line))
        print(line + pad, end=end, flush=True)
        self._last_len = len(line)

    def start(self, msg: str):
        self._t0 = time.time()
        self._show(f"▶ {msg}")

    def end(self, msg: str = "done", step: int = 1):
        if self._t0 is None:
            self._t0 = time.time()
        dt = time.time() - self._t0
        self.cur = min(self.total, self.cur + max(1, step))
        self._show(f"✓ {msg} ({dt:.1f}s)", end="\n")
        self._t0 = None

    def tick(self, msg: str, step: int = 1):
        self.cur = min(self.total, self.cur + max(1, step))
        self._show(msg)

    def done(self):
        self.cur = self.total
        self._show("complete", end="\n")


# ────────────────────────────────────────────────────────────
# Data utils
# ────────────────────────────────────────────────────────────
def load_data():
    master_path  = os.path.join(OUTDIR, "master.csv")
    trades_path  = os.path.join(OUTDIR, "trades.csv")
    monthly_path = os.path.join(OUTDIR, "monthly.csv")
    for p in [master_path, trades_path, monthly_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"'{p}' 가 없습니다. 먼저 dataset.py를 실행하여 "
                f"{OUTDIR}에 master.csv / trades.csv / monthly.csv 를 생성하세요."
            )
    master  = pd.read_csv(master_path,  low_memory=False)
    trades  = pd.read_csv(trades_path,  parse_dates=["ym"], low_memory=False)
    monthly = pd.read_csv(monthly_path, parse_dates=["ym"], low_memory=False)
    return master, trades, monthly


def _apply_subset(master: pd.DataFrame, trades: pd.DataFrame, monthly: pd.DataFrame,
                  subset_n: Optional[int], months_tail: Optional[int]):
    # N개 단지만 사용
    if subset_n is not None and subset_n > 0:
        keep_ids = master["complex_id"].astype(str).unique()[:subset_n]
        keep_ids = set(map(str, keep_ids))
        master  = master[master["complex_id"].astype(str).isin(keep_ids)].copy()
        trades  = trades[trades["complex_id"].astype(str).isin(keep_ids)].copy()
        monthly = monthly[monthly["complex_id"].astype(str).isin(keep_ids)].copy()

    # 최근 K개월만 사용
    if months_tail is not None and months_tail > 0 and not monthly.empty:
        all_months = pd.date_range(monthly["ym"].min(), monthly["ym"].max(), freq="MS")
        cut = all_months[-months_tail] if len(all_months) >= months_tail else all_months[0]
        monthly = monthly[monthly["ym"] >= cut].copy()
        if "ym" in trades.columns:
            trades = trades[trades["ym"] >= cut].copy()

    return master, trades, monthly


# ────────────────────────────────────────────────────────────
# Build embeddings / forecasters
# ────────────────────────────────────────────────────────────
def build_embeddings(master: pd.DataFrame):
    X = meta_embed(master)
    ids = master["complex_id"].astype(str).tolist()
    return ids, X


def make_normal_forecaster(model, scalers: Optional[Dict[str, Tuple[float, float]]], H: int):
    def _f(monthly_df: pd.DataFrame, cid: str, months: pd.DatetimeIndex):
        scaler = None
        if isinstance(scalers, dict):
            scaler = scalers.get(cid) or scalers.get(str(cid))
        return forecast_with_model(monthly_df, cid, months, model, T_in=T_IN, H=H, scaler_for_id=scaler)
    return _f


def make_synthctx_forecaster(
    model,
    master: pd.DataFrame,
    monthly: pd.DataFrame,
    ids: List[str],
    X: np.ndarray,
    T_in: int,
    H: int,
):
    def _forecast(monthly_df: pd.DataFrame, cid: str, months: pd.DatetimeIndex) -> np.ndarray:
        idx_arr = master.index[master["complex_id"].astype(str) == str(cid)]
        if len(idx_arr) == 0:
            ctx = np.zeros(T_in, np.float32)
            mean_ctx = float(np.nanmean(ctx)); std_ctx = float(np.nanstd(ctx) + 1e-6)
            return forecast_with_model(monthly_df, cid, months, model,
                                       T_in=T_in, H=H, scaler_for_id=(mean_ctx, std_ctx), ctx_override=ctx)

        t_index = idx_arr[0]
        cand_idx = find_neighbors(ids, master, X, t_index)

        # neighbor_weights_hybrid: 새 버전(5개 반환)과 구버전(4개 반환) 모두 호환
        try:
            _, _, _, w, ordered_idx = neighbor_weights_hybrid(
                master=master, target_idx=t_index, cand_idx=cand_idx, X_meta=X,
                sc_emb_target=None, sc_emb_pool=None, w_meta=0.55, w_geo=0.45, w_sem=0.0
            )
        except ValueError:
            _, _, _, w = neighbor_weights_hybrid(
                master=master, target_idx=t_index, cand_idx=cand_idx, X_meta=X,
                sc_emb_target=None, sc_emb_pool=None, w_meta=0.55, w_geo=0.45, w_sem=0.0
            )
            ordered_idx = cand_idx[np.argsort(-w)]
        neigh_ids = [ids[i] for i in ordered_idx]

        scale_map = scale_align(monthly_df, cid, neigh_ids)
        neigh_monthly = monthly_df[monthly_df["complex_id"].astype(str).isin([str(i) for i in neigh_ids])].copy()
        if not neigh_monthly.empty and "ppsm_median" in neigh_monthly.columns:
            neigh_monthly["ppsm_median"] = neigh_monthly.apply(
                lambda r: r["ppsm_median"] * scale_map.get(str(r["complex_id"]), scale_map.get(r["complex_id"], 1.0)),
                axis=1
            )

        ctx = build_synthetic_context(
            target_id=cid, months=months, neigh_monthly=neigh_monthly,
            neigh_ids=neigh_ids, hybrid_weights=w, T_in=T_in,
            trend_smooth_win=3, vol_match=True, add_noise_ratio=0.03
        ).astype(np.float32)
        mean_ctx = float(np.nanmean(ctx)); std_ctx = float(np.nanstd(ctx) + 1e-6)

        return forecast_with_model(
            monthly_df, cid, months, model,
            T_in=T_IN, H=H,
            scaler_for_id=(mean_ctx, std_ctx),
            ctx_override=ctx
        )
    return _forecast


# ────────────────────────────────────────────────────────────
# Cohorts (TRUE-COLD 제거, pseudo-COLD=HOT 전체)
# ────────────────────────────────────────────────────────────
def _series_len_before_after(monthly: pd.DataFrame, cid, cutoff: pd.Timestamp) -> Tuple[int, int]:
    s = monthly.loc[monthly["complex_id"].astype(str) == str(cid), "ym"]
    pre  = int((s <  cutoff).count())
    post = int((s >= cutoff).count())
    return pre, post


def build_cohorts(monthly: pd.DataFrame,
                  ids: Iterable[str],
                  months_all: pd.DatetimeIndex,
                  H: int,
                  warm_min=1, warm_max=10, hot_k=48,
                  include_all_hot_as_pseudo: bool = True):
    pred_start = pd.Timestamp(months_all[-H])
    warm, hot, all_eval = [], [], []

    for cid in ids:
        pre, post = _series_len_before_after(monthly, cid, pred_start)
        if post >= H:
            all_eval.append(cid)
            if warm_min <= pre <= warm_max:
                warm.append(cid)
            elif pre >= hot_k:
                hot.append(cid)

    pseudo = list(hot) if include_all_hot_as_pseudo else []
    meta = {"pred_start": pred_start, "all_eval": all_eval, "warm": warm, "hot": hot, "pseudo_cold": pseudo}
    return meta


def log_cohort_counts(meta, hot_k: int):
    all_eval = meta["all_eval"]; warm = meta["warm"]; hot = meta["hot"]; pseudo = meta["pseudo_cold"]
    print(f"\n[COUNT] ALL={len(all_eval):,}, WARM(1-9)={len(warm):,}, HOT(≥{hot_k})={len(hot):,}")
    print(f"[COUNT] pseudo-COLD(HOT as COLD)={len(pseudo):,}  (~{100.0*len(pseudo)/max(1,len(all_eval)):.2f}% of ALL)")


# ────────────────────────────────────────────────────────────
# Run (single seed, quick defaults)
# ────────────────────────────────────────────────────────────
def run_once_quick(seed: int, epochs: int, hot_k: int,
                   subset_n: Optional[int], months_tail: Optional[int]):
    np.random.seed(seed)

    # TOTAL = load(1)+subset(1)+embed(1)+train(epochs)+cohort(1)+NOW(3)+FUT(3)
    total_units = 1 + 1 + 1 + epochs + 1 + 3 + 3
    prog = SimpleProgress(total_units)

    print(f"[device] DEVICE={DEVICE}, AMP={'ON' if MIXED_PRECISION else 'OFF'}, AMP_DTYPE={AMP_DTYPE}")

    prog.start("Load data")
    master, trades, monthly = load_data()
    prog.end("Load data")

    prog.start("Apply subset filters")
    master, trades, monthly = _apply_subset(master, trades, monthly, subset_n, months_tail)
    prog.end("Apply subset filters")

    prog.start("Build embeddings")
    ids, X = build_embeddings(master)
    prog.end("Build embeddings")

    months_all = pd.date_range(monthly["ym"].min(), monthly["ym"].max(), freq="MS")

    prog.start(f"Train temporal ({TEMP_MODEL}, epochs={epochs})")
    class _Reporter:
        def __init__(self, p: SimpleProgress): self.p = p; self.last = 0
        def log(self, msg: str):
            if "ep" in msg:
                try:
                    tok = msg.split("ep", 1)[1].strip().split()[0]
                    ep = int("".join(ch for ch in tok if ch.isdigit()))
                    while self.last < ep and self.last < epochs:
                        self.last += 1
                        self.p.tick(f"Training epoch {self.last}/{epochs}")
                except Exception:
                    pass
    reporter = _Reporter(prog)
    try:
        model, scalers = train_temporal(
            monthly_df=monthly, train_ids=ids, months=months_all,
            model_kind=TEMP_MODEL, T_in=T_IN, epochs=epochs, reporter=reporter, seed=seed
        )
    except TypeError:
        model, scalers = train_temporal(
            monthly_df=monthly, train_ids=ids, months=months_all,
            model_kind=TEMP_MODEL, T_in=T_IN, epochs=epochs, reporter=reporter
        )
    prog.end("Train temporal")

    # Cohorts with fallback: if HOT empty, relax hot_k to 10 automatically
    prog.start(f"Build cohorts (HOT≥{hot_k}, pseudo=HOT)")
    meta = build_cohorts(monthly, ids, months_all, H_OUT, warm_min=1, warm_max=9, hot_k=hot_k, include_all_hot_as_pseudo=True)
    if len(meta["hot"]) == 0 and hot_k > 10:
        print(f"[warn] HOT(≥{hot_k})이 비었습니다. quick-run을 위해 자동으로 HOT 기준을 10으로 낮춥니다.")
        meta = build_cohorts(monthly, ids, months_all, H_OUT, warm_min=1, warm_max=9, hot_k=10, include_all_hot_as_pseudo=True)
        hot_k = 10
    prog.end("Build cohorts")
    log_cohort_counts(meta, hot_k)

    all_eval = meta["all_eval"]; warm = meta["warm"]; hot = meta["hot"]; pseudo = meta["pseudo_cold"]

    # Forecasters
    f_now_norm = make_normal_forecaster(model, scalers, 1)
    f_fut_norm = make_normal_forecaster(model, scalers, H_OUT)
    f_now_sctx = make_synthctx_forecaster(model, master, monthly, ids, X, T_in=T_IN, H=1)
    f_fut_sctx = make_synthctx_forecaster(model, master, monthly, ids, X, T_in=T_IN, H=H_OUT)

    # NOW
    prog.start("Eval NOW: ALL/WARM/HOT")
    res_all_now  = evaluate_ids(monthly, all_eval,  f_now_norm, months_all, 1)
    res_warm_now = evaluate_ids(monthly, warm,      f_now_norm, months_all, 1)
    res_hot_now  = evaluate_ids(monthly, hot,       f_now_norm, months_all, 1)
    prog.end("Eval NOW")

    # PSEU(NOW)
    prog.start("Eval NOW: PSEU (HOT-as-COLD)")
    res_pseu_now = evaluate_ids(monthly, pseudo,    f_now_sctx, months_all, 1) if len(pseudo) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
    prog.end("Eval NOW: PSEU")

    # FUTURE
    prog.start("Eval FUT: ALL/WARM/HOT")
    res_all_fut  = evaluate_ids(monthly, all_eval,  f_fut_norm, months_all, H_OUT)
    res_warm_fut = evaluate_ids(monthly, warm,      f_fut_norm, months_all, H_OUT)
    res_hot_fut  = evaluate_ids(monthly, hot,       f_fut_norm, months_all, H_OUT)
    prog.end("Eval FUT")

    # PSEU(FUT)
    prog.start("Eval FUT: PSEU (HOT-as-COLD)")
    res_pseu_fut = evaluate_ids(monthly, pseudo,    f_fut_sctx, months_all, H_OUT) if len(pseudo) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
    prog.end("Eval FUT: PSEU")

    prog.done()

    # 요약 출력
    def fmt(d: Dict[str, float]) -> str:
        return f"MAE={d.get('MAE',np.nan):.3f}, RMSE={d.get('RMSE',np.nan):.3f}, MAPE={d.get('MAPE',np.nan):.2f}%"
    print("\n=== QUICK EVAL (NOW) ===")
    print("ALL :", fmt(res_all_now))
    print("WARM:", fmt(res_warm_now))
    print("HOT :", fmt(res_hot_now))
    print("PSEU:", fmt(res_pseu_now))

    print("\n=== QUICK EVAL (FUTURE) ===")
    print("ALL :", fmt(res_all_fut))
    print("WARM:", fmt(res_warm_fut))
    print("HOT :", fmt(res_hot_fut))
    print("PSEU:", fmt(res_pseu_fut))


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED, help="random seed (default: config.SEED)")
    ap.add_argument("--epochs", type=int, default=2, help="quick run epochs (default: 2)")
    ap.add_argument("--subset_n", type=int, default=300, help="상위 N개 단지만 사용 (default: 300)")
    ap.add_argument("--months_tail", type=int, default=36, help="최근 K개월만 사용 (default: 36)")
    ap.add_argument("--hot_k", type=int, default=60, help="HOT 임계치; quick-run에서 비면 10으로 자동 완화")
    args = ap.parse_args()

    run_once_quick(
        seed=args.seed,
        epochs=args.epochs,
        hot_k=args.hot_k,
        subset_n=args.subset_n,
        months_tail=args.months_tail,
    )

if __name__ == "__main__":
    main()

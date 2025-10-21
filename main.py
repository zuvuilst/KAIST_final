# main.py
# Single-seed run, GPU-aware prints, anchor NOW (default 2025-05),
# FUTURE = predict 6 steps from context up to anchor-6 and evaluate ONLY the last step at anchor,
# Cohorts at anchor: TRUE-COLD(pre=1), WARM(2–10), HOT(>=hot_k), pseudo-COLD=all HOT,
# WARM is evaluated twice: normal ctx vs synthetic ctx,
# Only IDs with an observation AT the anchor are evaluated (NOW & FUTURE),
# Realtime progress via tqdm, final results saved to OUTDIR.

from __future__ import annotations
import os, sys, time, threading, argparse
from typing import Optional, Dict, Tuple, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import OUTDIR, SEED, T_IN, H_OUT, TEMP_MODEL, DEVICE, MIXED_PRECISION
try:
    from config import AMP_DTYPE  # optional
except Exception:
    AMP_DTYPE = "bf16" if os.getenv("AMP_DTYPE", "").lower() in ("bf16", "bfloat16") else "fp16"

from retrieval import meta_embed, find_neighbors, scale_align, neighbor_weights_hybrid
from temporal_models import train_temporal, forecast_with_model
from generator import build_synthetic_context

# ───────────────── evaluation helpers (self-contained) ─────────────────
def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan, np.nan
    err = y_pred[mask] - y_true[mask]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-6)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)

# stdout 즉시 출력
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ───────────────── Heartbeat spinner (I/O 동안 “돌고있음” 알림) ─────────────────
class Heartbeat:
    def __init__(self, message: str = "working...", interval: float = 0.3):
        self.message = message
        self.interval = interval
        self._stop = threading.Event()
        self._t = None
    def _run(self):
        glyphs = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        i = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{glyphs[i % len(glyphs)]} {self.message}")
            sys.stdout.flush()
            time.sleep(self.interval); i += 1
        sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r"); sys.stdout.flush()
    def __enter__(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self
    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._t is not None: self._t.join()

# ───────────────────────────── Data load ─────────────────────────────
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

# ───────────── Embedding & Forecasters (정상 / 합성 컨텍스트) ─────────────
def build_embeddings(master: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    X = meta_embed(master)
    ids = master["complex_id"].astype(str).tolist()
    return ids, X

def make_normal_forecaster(model, scalers: Optional[Dict[str, Tuple[float, float]]], H: int):
    """실제 컨텍스트(months_ctx) 그대로 사용."""
    def _f(monthly_df: pd.DataFrame, cid: str, months_ctx: pd.DatetimeIndex):
        scaler = None
        if isinstance(scalers, dict):
            scaler = scalers.get(cid) or scalers.get(str(cid))
        return forecast_with_model(monthly_df, cid, months_ctx, model,
                                   T_in=T_IN, H=H, scaler_for_id=scaler)
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
    """
    합성 컨텍스트로 예측: 이웃 후보 → (정렬/필터) → scale_align → build_synthetic_context →
    forecast_with_model(ctx_override=...).
    """
    def _forecast(monthly_df: pd.DataFrame, cid: str, months_ctx: pd.DatetimeIndex) -> np.ndarray:
        idx_arr = master.index[master["complex_id"].astype(str) == str(cid)]
        if len(idx_arr) == 0:
            # 대상이 master에 없으면 zero-context fallback
            ctx = np.zeros(T_IN, np.float32)
            mean_ctx = float(np.nanmean(ctx)); std_ctx = float(np.nanstd(ctx) + 1e-6)
            return forecast_with_model(monthly_df, cid, months_ctx, model,
                                       T_in=T_IN, H=H, scaler_for_id=(mean_ctx, std_ctx),
                                       ctx_override=ctx)

        t_index = idx_arr[0]
        cand_idx = find_neighbors(ids, master, X, t_index)

        # neighbor_weights_hybrid: 4-return(구) / 5-return(신) 모두 대응
        try:
            meta_sim, geo_sim, sem_sim, w, ordered_idx = neighbor_weights_hybrid(
                master=master, target_idx=t_index, cand_idx=cand_idx, X_meta=X,
                sc_emb_target=None, sc_emb_pool=None, w_meta=0.55, w_geo=0.45, w_sem=0.0
            )
        except ValueError:
            meta_sim, geo_sim, sem_sim, w = neighbor_weights_hybrid(
                master=master, target_idx=t_index, cand_idx=cand_idx, X_meta=X,
                sc_emb_target=None, sc_emb_pool=None, w_meta=0.55, w_geo=0.45, w_sem=0.0
            )
            ordered_idx = cand_idx[np.argsort(-w)]

        all_neigh_ids = [ids[i] for i in ordered_idx]
        pairs = [(nid, float(w[k] if k < len(w) else 0.0)) for k, nid in enumerate(all_neigh_ids)]

        # 실제 월별 데이터가 있는 이웃만
        available_all = set(monthly_df["complex_id"].astype(str).unique())
        pairs = [(nid, wt) for (nid, wt) in pairs if nid in available_all and wt > 0]
        if not pairs:
            ctx = np.zeros(T_IN, np.float32)
            mean_ctx = float(np.nanmean(ctx)); std_ctx = float(np.nanstd(ctx) + 1e-6)
            return forecast_with_model(monthly_df, cid, months_ctx, model,
                                       T_in=T_IN, H=H, scaler_for_id=(mean_ctx, std_ctx),
                                       ctx_override=ctx)

        ids_kept = [p[0] for p in pairs]
        w_kept   = np.array([p[1] for p in pairs], dtype=np.float32)

        # 2차: 실제 수집된 행 기준
        neigh_monthly_raw = monthly_df[monthly_df["complex_id"].astype(str).isin(ids_kept)].copy()
        ids_in_data = set(neigh_monthly_raw["complex_id"].astype(str).unique())
        pairs = [(nid, wt) for (nid, wt) in pairs if nid in ids_in_data]
        if not pairs:
            ctx = np.zeros(T_IN, np.float32)
            mean_ctx = float(np.nanmean(ctx)); std_ctx = float(np.nanstd(ctx) + 1e-6)
            return forecast_with_model(monthly_df, cid, months_ctx, model,
                                       T_in=T_IN, H=H, scaler_for_id=(mean_ctx, std_ctx),
                                       ctx_override=ctx)

        ids_kept = [p[0] for p in pairs]
        w_kept   = np.array([p[1] for p in pairs], dtype=np.float32)
        w_kept   = w_kept / (np.sum(w_kept) + 1e-12)

        # scale-align 후 합성 컨텍스트 생성
        scale_map = scale_align(monthly_df, cid, ids_kept)
        neigh_monthly = neigh_monthly_raw.copy()
        if not neigh_monthly.empty and "ppsm_median" in neigh_monthly.columns:
            neigh_monthly["ppsm_median"] = neigh_monthly.apply(
                lambda r: r["ppsm_median"] * scale_map.get(str(r["complex_id"]), scale_map.get(r["complex_id"], 1.0)),
                axis=1
            )

        ctx = build_synthetic_context(
            target_id=cid, months=months_ctx,
            neigh_monthly=neigh_monthly, neigh_ids=ids_kept,
            hybrid_weights=w_kept, T_in=T_IN,
            trend_smooth_win=3, vol_match=True, add_noise_ratio=0.03
        ).astype(np.float32)

        mean_ctx = float(np.nanmean(ctx)); std_ctx = float(np.nanstd(ctx) + 1e-6)

        return forecast_with_model(
            monthly_df, cid, months_ctx, model,
            T_in=T_IN, H=H, scaler_for_id=(mean_ctx, std_ctx),
            ctx_override=ctx
        )
    return _forecast

# ─────────────────────── anchor & filtering helpers ───────────────────────
def pick_anchor_now(months_all: pd.DatetimeIndex, anchor_str: Optional[str]) -> pd.Timestamp:
    """원하는 anchor(YYYY-MM) 문자열이 있으면 그 달, 없으면 글로벌 마지막 달."""
    if anchor_str:
        try:
            a = pd.to_datetime(anchor_str).to_period("M").to_timestamp()
            if a not in months_all:
                cand = [m for m in months_all if m <= a]
                return cand[-1] if cand else months_all[-1]
            return a
        except Exception:
            pass
    return months_all[-1]

def ids_with_obs_at(monthly: pd.DataFrame, ids: Iterable[str], month: pd.Timestamp) -> List[str]:
    sub = monthly[["complex_id","ym"]].drop_duplicates()
    mask = sub["ym"] == month
    seen = set(sub.loc[mask, "complex_id"].astype(str).tolist())
    return [str(i) for i in ids if str(i) in seen]

# ───────────────────── Cohorts (TRUE/WARM/HOT + PSEU at anchor) ─────────────────────
def _series_len_before_after(monthly: pd.DataFrame, cid, cutoff: pd.Timestamp) -> Tuple[int, int]:
    s = monthly.loc[monthly["complex_id"].astype(str) == str(cid), "ym"]
    pre  = int((s <  cutoff).count())
    post = int((s >= cutoff).count())
    return pre, post

def build_cohorts_at_anchor(
    monthly: pd.DataFrame,
    ids: Iterable[str],
    anchor_now: pd.Timestamp,
    H: int,
    warm_min=2, warm_max=10, hot_k=60,
    include_all_hot_as_pseudo: bool = True
):
    """
    평가 anchor 기준 코호트:
    cutoff = anchor_now - (H-1)개월 (FUTURE용 컨텍스트 끝의 다음 달, 창 시작을 가늠)
    - TRUE-COLD: pre == 1
    - WARM     : pre in [2, warm_max]
    - HOT      : pre >= hot_k
    - ALL_EVAL : post >= 1 (anchor 관측 필터로 보강)
    - PSEU     : all HOT (HOT-as-COLD)
    """
    cutoff = (anchor_now.to_period("M") - (H-1)).to_timestamp()
    true_cold, warm, hot, all_eval = [], [], [], []

    for cid in ids:
        pre, post = _series_len_before_after(monthly, cid, cutoff)
        if post >= 1:
            all_eval.append(cid)
            if pre == 1:
                true_cold.append(cid)
            elif warm_min <= pre <= warm_max:
                warm.append(cid)
            elif pre >= hot_k:
                hot.append(cid)

    pseudo = list(hot) if include_all_hot_as_pseudo else []
    meta = {
        "cutoff": cutoff,
        "anchor_now": anchor_now,
        "all_eval": all_eval,
        "true_cold": true_cold,
        "warm": warm,
        "hot": hot,
        "pseudo_cold": pseudo
    }
    return meta

def log_cohort_counts(meta, hot_k: int, warm_min=2, warm_max=10):
    all_eval = meta["all_eval"]; warm = meta["warm"]; hot = meta["hot"]; pseudo = meta["pseudo_cold"]; tc = meta["true_cold"]
    print(f"\n[COUNT @anchor={meta['anchor_now'].strftime('%Y-%m')}] "
          f"ALL={len(all_eval):,}, TRUE(=1)={len(tc):,}, WARM({warm_min}-{warm_max})={len(warm):,}, HOT(≥{hot_k})={len(hot):,}")
    print(f"[COUNT] pseudo-COLD(HOT as COLD)={len(pseudo):,}  (~{100.0*len(pseudo)/max(1,len(all_eval)):.2f}% of ALL)")

# ─────────────────────────── Evaluation at anchor ───────────────────────────
def evaluate_now_at_anchor(
    monthly: pd.DataFrame,
    ids: Iterable[str],
    forecast_fn,
    months_all: pd.DatetimeIndex,
    anchor_now: pd.Timestamp,
    desc: str = "Eval NOW"
) -> Dict[str, float]:
    """
    NOW = anchor_now (1개월 예측)
    컨텍스트 = months_all[months < anchor_now]
    평가 포함: anchor_now 관측이 실제로 있는 단지들만(사전에 필터링 권장)
    """
    months_ctx = months_all[months_all < anchor_now]
    maes, rmses, mapes = [], [], []
    it = tqdm(list(ids), desc=desc, dynamic_ncols=True, mininterval=0.2)
    for cid in it:
        sub = (monthly.loc[monthly["complex_id"].astype(str) == str(cid), ["ym","ppsm_median"]]
                      .drop_duplicates("ym").set_index("ym"))
        if anchor_now not in sub.index:
            continue
        gt = float(sub.loc[anchor_now, "ppsm_median"])
        if not np.isfinite(gt):
            continue
        try:
            pred_arr = forecast_fn(monthly, str(cid), months_ctx)
        except Exception:
            it.set_postfix_str(f"skip:{cid}"); continue
        if pred_arr is None or len(pred_arr) < 1 or not np.isfinite(pred_arr[0]):
            continue
        pred = float(pred_arr[0])
        m, r = mae_rmse(np.array([gt], float), np.array([pred], float))
        p = mape(np.array([gt], float), np.array([pred], float))
        maes.append(m); rmses.append(r); mapes.append(p)
        if len(maes) % 50 == 0:
            it.set_postfix(MAE=np.nanmean(maes), MAPE=np.nanmean(mapes))
    def _agg(v): v=np.asarray(v,float); return float(np.nanmean(v)) if len(v) else np.nan
    return {"MAE": _agg(maes), "RMSE": _agg(rmses), "MAPE": _agg(mapes)}

def evaluate_future_laststep_at_anchor(
    monthly: pd.DataFrame,
    ids: Iterable[str],
    forecast_fn,
    months_all: pd.DatetimeIndex,
    anchor_now: pd.Timestamp,
    H: int,
    desc: str = "Eval FUT(last step @ anchor)"
) -> Dict[str, float]:
    """
    FUTURE(last step): 컨텍스트를 anchor-6(=anchor-H)까지 제공하고,
    H-스텝 예측 결과의 **마지막 한 점**(step H)을 **anchor 관측**과 비교.
    평가 포함: anchor 관측이 있는 단지(=NOW 대상)만.
    """
    ctx_end = (anchor_now.to_period("M") - H).to_timestamp()
    months_ctx = months_all[months_all <= ctx_end]

    maes, rmses, mapes = [], [], []
    it = tqdm(list(ids), desc=desc, dynamic_ncols=True, mininterval=0.2)
    for cid in it:
        sub = (monthly.loc[monthly["complex_id"].astype(str)==str(cid), ["ym","ppsm_median"]]
                      .drop_duplicates("ym").set_index("ym"))
        if anchor_now not in sub.index:
            continue
        gt = float(sub.loc[anchor_now, "ppsm_median"])
        if not np.isfinite(gt):
            continue

        try:
            pred = forecast_fn(monthly, str(cid), months_ctx)
        except Exception:
            it.set_postfix_str(f"skip:{cid}"); continue

        if pred is None or len(pred) != H or not np.isfinite(pred[-1]):
            continue

        last_pred = float(pred[-1])
        m, r = mae_rmse(np.array([gt], float), np.array([last_pred], float))
        p = mape(np.array([gt], float), np.array([last_pred], float))
        maes.append(m); rmses.append(r); mapes.append(p)
        if len(maes) % 50 == 0:
            it.set_postfix(MAE=np.nanmean(maes), MAPE=np.nanmean(mapes))
    def _agg(v): v=np.asarray(v,float); return float(np.nanmean(v)) if len(v) else np.nan
    return {"MAE": _agg(maes), "RMSE": _agg(rmses), "MAPE": _agg(mapes)}

# ───────────────────────────── Main run (single seed) ─────────────────────────────
def run_once(seed: int, epochs: int, hot_k: int, anchor_str: Optional[str]):
    np.random.seed(seed)
    print(f"[device] DEVICE={DEVICE}, AMP={'ON' if MIXED_PRECISION else 'OFF'}, AMP_DTYPE={AMP_DTYPE}", flush=True)

    # 1) Load
    print("→ Loading data ...", flush=True)
    t0 = time.time()
    with Heartbeat("loading data...", interval=0.3):
        master, trades, monthly = load_data()
    print(f"✓ Loaded in {time.time()-t0:.1f}s | master={len(master):,}, monthly={len(monthly):,}", flush=True)

    # 2) Embeddings
    print("→ Building meta embeddings ...", flush=True)
    t0 = time.time()
    with Heartbeat("building embeddings...", interval=0.3):
        ids, X = build_embeddings(master)
    print(f"✓ Embeddings ready in {time.time()-t0:.1f}s | items={len(ids):,}", flush=True)

    # 3) Time axis + anchor
    months_all = pd.date_range(monthly["ym"].min(), monthly["ym"].max(), freq="MS")
    anchor_now = pick_anchor_now(months_all, anchor_str)  # default e.g., 2025-05
    print(f"→ Anchor NOW set to {anchor_now.strftime('%Y-%m')}", flush=True)

    # 4) Train temporal model (tqdm epoch bar)
    print(f"→ Training temporal model: {TEMP_MODEL}, epochs={epochs}", flush=True)
    ep_bar = tqdm(total=epochs, desc="training", dynamic_ncols=True)
    class _Reporter:
        def __init__(self, bar): self.bar = bar; self.last = 0
        def log(self, msg: str):
            if "ep" in msg:
                try:
                    tok = msg.split("ep", 1)[1].strip().split()[0]
                    ep = int("".join([c for c in tok if c.isdigit()]))
                    while self.last < ep and self.last < self.bar.total:
                        self.last += 1; self.bar.update(1)
                except Exception:
                    pass
    reporter = _Reporter(ep_bar)
    try:
        model, scalers = train_temporal(
            monthly_df=monthly,
            train_ids=ids,
            months=months_all,
            model_kind=TEMP_MODEL,
            T_in=T_IN,
            epochs=epochs,
            reporter=reporter,
            seed=seed,
        )
    except TypeError:
        model, scalers = train_temporal(
            monthly_df=monthly,
            train_ids=ids,
            months=months_all,
            model_kind=TEMP_MODEL,
            T_in=T_IN,
            epochs=epochs,
            reporter=reporter,
        )
    finally:
        if ep_bar.n < ep_bar.total:
            ep_bar.update(ep_bar.total - ep_bar.n)
        ep_bar.close()
    print("✓ Training finished", flush=True)

    # 5) Cohorts @ anchor (pre=1, warm=2-10, hot>=hot_k, pseudo=HOT)
    print(f"→ Building cohorts at anchor={anchor_now.strftime('%Y-%m')} (TRUE=1, WARM=2–10, HOT≥{hot_k}, pseudo=HOT)", flush=True)
    meta = build_cohorts_at_anchor(monthly, ids, anchor_now, H_OUT,
                                   warm_min=2, warm_max=10, hot_k=hot_k,
                                   include_all_hot_as_pseudo=True)

    # NOW/FUTURE 모두 anchor 관측이 있는 단지로 제한
    ids_now = ids_with_obs_at(monthly, meta["all_eval"], anchor_now)
    true_cold = [cid for cid in meta["true_cold"] if cid in ids_now]
    warm      = [cid for cid in meta["warm"]      if cid in ids_now]
    hot       = [cid for cid in meta["hot"]       if cid in ids_now]
    pseudo    = [cid for cid in meta["pseudo_cold"] if cid in ids_now]
    all_eval  = ids_now[:]

    log_cohort_counts(
        {"anchor_now": anchor_now, "all_eval": all_eval, "true_cold": true_cold, "warm": warm, "hot": hot, "pseudo_cold": pseudo},
        hot_k, warm_min=2, warm_max=10
    )

    # 6) Forecasters
    f_now_norm = make_normal_forecaster(model, scalers, 1)
    f_fut_norm = make_normal_forecaster(model, scalers, H_OUT)
    f_now_sctx = make_synthctx_forecaster(model, master, monthly, ids, X, T_IN, 1)
    f_fut_sctx = make_synthctx_forecaster(model, master, monthly, ids, X, T_IN, H_OUT)

    # 7) Evaluate at anchor
    print("\n→ Evaluating (NOW at anchor)", flush=True)
    res_all_now   = evaluate_now_at_anchor(monthly, all_eval,  f_now_norm, months_all, anchor_now, desc="NOW: ALL")
    res_tc_now    = evaluate_now_at_anchor(monthly, true_cold, f_now_norm, months_all, anchor_now, desc="NOW: TRUE-COLD") if len(true_cold) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
    res_warmN_now = evaluate_now_at_anchor(monthly, warm,      f_now_norm, months_all, anchor_now, desc="NOW: WARM (normal)") if len(warm) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
    res_warmS_now = evaluate_now_at_anchor(monthly, warm,      f_now_sctx, months_all, anchor_now, desc="NOW: WARM (synth)")  if len(warm) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
    res_hot_now   = evaluate_now_at_anchor(monthly, hot,       f_now_norm, months_all, anchor_now, desc="NOW: HOT")           if len(hot) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
    res_pseu_now  = evaluate_now_at_anchor(monthly, pseudo,    f_now_sctx, months_all, anchor_now, desc="NOW: PSEU(HOT-as-COLD)") if len(pseudo) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}

    print("\n→ Evaluating (FUTURE last step @ anchor)", flush=True)
    res_all_fut   = evaluate_future_laststep_at_anchor(monthly, all_eval,  f_fut_norm, months_all, anchor_now, H_OUT, desc="FUT(last): ALL")
    res_tc_fut    = evaluate_future_laststep_at_anchor(monthly, true_cold, f_fut_norm, months_all, anchor_now, H_OUT, desc="FUT(last): TRUE-COLD") if len(true_cold) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
    res_warmN_fut = evaluate_future_laststep_at_anchor(monthly, warm,      f_fut_norm, months_all, anchor_now, H_OUT, desc="FUT(last): WARM (normal)") if len(warm) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
    res_warmS_fut = evaluate_future_laststep_at_anchor(monthly, warm,      f_fut_sctx, months_all, anchor_now, H_OUT, desc="FUT(last): WARM (synth)")  if len(warm) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
    res_hot_fut   = evaluate_future_laststep_at_anchor(monthly, hot,       f_fut_norm, months_all, anchor_now, H_OUT, desc="FUT(last): HOT")           if len(hot) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
    res_pseu_fut  = evaluate_future_laststep_at_anchor(monthly, pseudo,    f_fut_sctx, months_all, anchor_now, H_OUT, desc="FUT(last): PSEU(HOT-as-COLD)") if len(pseudo) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}

    # 8) Print & Save summary
    def fmt(d: Dict[str, float]) -> str:
        return f"MAE={d.get('MAE',np.nan):.3f}, RMSE={d.get('RMSE',np.nan):.3f}, MAPE={d.get('MAPE',np.nan):.2f}%"

    print("\n=== RESULTS (NOW at anchor) ===")
    print("ALL        :", fmt(res_all_now))
    print("TRUE-COLD  :", fmt(res_tc_now))
    print("WARM norm  :", fmt(res_warmN_now))
    print("WARM synth :", fmt(res_warmS_now))
    print("HOT        :", fmt(res_hot_now))
    print("PSEU(HOT→C):", fmt(res_pseu_now))

    print("\n=== RESULTS (FUTURE last step @ anchor) ===")
    print("ALL        :", fmt(res_all_fut))
    print("TRUE-COLD  :", fmt(res_tc_fut))
    print("WARM norm  :", fmt(res_warmN_fut))
    print("WARM synth :", fmt(res_warmS_fut))
    print("HOT        :", fmt(res_hot_fut))
    print("PSEU(HOT→C):", fmt(res_pseu_fut))

    # 저장
    os.makedirs(OUTDIR, exist_ok=True)
    rows = [
        {"Split":"NOW_ALL",        **res_all_now},
        {"Split":"NOW_TRUE",       **res_tc_now},
        {"Split":"NOW_WARM_NORM",  **res_warmN_now},
        {"Split":"NOW_WARM_SYNTH", **res_warmS_now},
        {"Split":"NOW_HOT",        **res_hot_now},
        {"Split":"NOW_PSEU",       **res_pseu_now},
        {"Split":"FUT_ALL",        **res_all_fut},
        {"Split":"FUT_TRUE",       **res_tc_fut},
        {"Split":"FUT_WARM_NORM",  **res_warmN_fut},
        {"Split":"FUT_WARM_SYNTH", **res_warmS_fut},
        {"Split":"FUT_HOT",        **res_hot_fut},
        {"Split":"FUT_PSEU",       **res_pseu_fut},
    ]
    df_out = pd.DataFrame(rows, columns=["Split","MAE","RMSE","MAPE"])
    suffix = f"anchor_{anchor_now.strftime('%Y%m')}_laststepH{H_OUT}"
    csv_path = os.path.join(OUTDIR, f"eval_results_single_seed_{suffix}.csv")
    json_path = os.path.join(OUTDIR, f"eval_results_single_seed_{suffix}.json")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    import json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({r["Split"]: {k: (None if not (isinstance(v,(int,float)) and np.isfinite(v)) else float(v))
                                for k, v in r.items() if k != "Split"} for r in rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {json_path}")

# ───────────────────────────────────────── CLI ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED, help="single seed (default: config.SEED, typically 42)")
    ap.add_argument("--epochs", type=int, default=10, help="training epochs")
    ap.add_argument("--hot_k", type=int, default=60, help="HOT threshold (>=60), pseudo-COLD uses all HOT")
    ap.add_argument("--anchor_now", type=str, default="2025-05", help="anchor month (YYYY-MM), e.g., 2025-05")
    args = ap.parse_args()

    run_once(seed=args.seed, epochs=args.epochs, hot_k=args.hot_k, anchor_str=args.anchor_now)

if __name__ == "__main__":
    main()

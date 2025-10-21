# main.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, time, threading, argparse
from typing import Optional, Dict, Tuple, Iterable, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from config import OUTDIR, SEED, T_IN, H_OUT, DEVICE, MIXED_PRECISION
try:
    from config import AMP_DTYPE  # optional
except Exception:
    AMP_DTYPE = "bf16" if os.getenv("AMP_DTYPE","").lower() in ("bf16","bfloat16") else "fp16"

from retrieval import meta_embed, find_neighbors, scale_align, neighbor_weights_hybrid
from temporal_models import train_temporal, forecast_with_model
from generator import build_synthetic_context
from progress_utils import print_step, Heartbeat, render_dashboard, LiveLogger, live_log

# ───────── settings ─────────
FUT_STEPS = [1,2,3,6,12,18,24,30,36]

# ───────── metrics ─────────
def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0: return np.nan, np.nan
    err = y_pred[mask] - y_true[mask]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-6)
    if mask.sum() == 0: return np.nan
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)

# stdout 즉시
os.environ["PYTHONUNBUFFERED"] = "1"
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass

# ───────── lightweight spinner ─────────
class _Spinner:
    def __init__(self, message: str = "working...", interval: float = 0.2):
        self.message = message; self.interval = interval
        self._stop = threading.Event(); self._t = None
    def _run(self):
        glyphs=["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]; i=0
        while not self._stop.is_set():
            sys.stdout.write(f"\r{glyphs[i%len(glyphs)]} {self.message}"); sys.stdout.flush()
            time.sleep(self.interval); i+=1
        sys.stdout.write("\r"+" "*(len(self.message)+4)+"\r"); sys.stdout.flush()
    def __enter__(self): self._stop.clear(); self._t=threading.Thread(target=self._run,daemon=True); self._t.start(); return self
    def __exit__(self, exc_type, exc, tb): self._stop.set()
    def join(self): 
        if self._t is not None: self._t.join()

# ───────── data load ─────────
def load_data():
    master_path  = os.path.join(OUTDIR, "master.csv")
    trades_path  = os.path.join(OUTDIR, "trades.csv")
    monthly_path = os.path.join(OUTDIR, "monthly.csv")
    for p in [master_path, trades_path, monthly_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"'{p}' 가 없습니다. dataset.py를 먼저 실행하여 {OUTDIR}에 생성하세요.")
    master  = pd.read_csv(master_path,  low_memory=False)
    trades  = pd.read_csv(trades_path,  low_memory=False)
    monthly = pd.read_csv(monthly_path, low_memory=False)
    for df in (monthly, trades):
        if "ym" in df.columns:
            df["ym"] = pd.to_datetime(df["ym"], errors="coerce").dt.to_period("M").to_timestamp()
        elif "ymd" in df.columns:
            df["ym"] = pd.to_datetime(df["ymd"], errors="coerce").dt.to_period("M").to_timestamp()
    return master, trades, monthly

# ───────── embeddings ─────────
def build_embeddings(master: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    X = meta_embed(master)    # HGNN 임베딩이 있으면 우선 사용, 없으면 메타 임베딩
    ids = master["complex_id"].astype(str).tolist()
    return ids, X

# ───────── helpers ─────────
def pick_anchor_now(months_all: pd.DatetimeIndex, anchor_str: Optional[str]) -> pd.Timestamp:
    if anchor_str:
        try:
            a = pd.to_datetime(anchor_str).to_period("M").to_timestamp()
            if a not in months_all:
                cand = [m for m in months_all if m <= a]
                return cand[-1] if cand else months_all[-1]
            return a
        except Exception: pass
    return months_all[-1]

def ids_with_obs_at(monthly: pd.DataFrame, ids: Iterable[str], month: pd.Timestamp) -> List[str]:
    sub = monthly[["complex_id","ym"]].drop_duplicates()
    seen = set(sub.loc[sub["ym"] == month, "complex_id"].astype(str).tolist())
    return [str(i) for i in ids if str(i) in seen]

def _series_len_before_after(monthly: pd.DataFrame, cid, cutoff: pd.Timestamp) -> Tuple[int,int]:
    s = monthly.loc[monthly["complex_id"].astype(str)==str(cid), "ym"]
    return int((s < cutoff).count()), int((s >= cutoff).count())

def build_cohorts_at_anchor(monthly: pd.DataFrame, ids: Iterable[str], anchor_now: pd.Timestamp, H: int,
                            warm_min=2, warm_max=10, hot_k=60, include_all_hot_as_pseudo: bool = True):
    """
    cutoff = anchor_now - (H-1)개월
    TRUE(=1), WARM(2~10), HOT(>=hot_k), PSEU(HOT as COLD)
    """
    cutoff = (anchor_now.to_period("M") - (H-1)).to_timestamp()
    true_cold, warm, hot, all_eval = [], [], [], []
    for cid in ids:
        pre, post = _series_len_before_after(monthly, cid, cutoff)
        if post >= 1:
            all_eval.append(cid)
            if pre == 1: true_cold.append(cid)
            elif warm_min <= pre <= warm_max: warm.append(cid)
            elif pre >= hot_k: hot.append(cid)
    return {"cutoff":cutoff, "anchor_now":anchor_now, "all_eval":all_eval, "true_cold":true_cold,
            "warm":warm, "hot":hot, "pseudo_cold": (list(hot) if include_all_hot_as_pseudo else [])}

def log_cohort_counts(meta, hot_k: int, warm_min=2, warm_max=10):
    all_eval = meta["all_eval"]; warm = meta["warm"]; hot = meta["hot"]; pseudo = meta["pseudo_cold"]; tc = meta["true_cold"]
    print(f"\n[COUNT @anchor={meta['anchor_now'].strftime('%Y-%m')}] ALL={len(all_eval):,}, TRUE(=1)={len(tc):,}, "
          f"WARM({warm_min}-{warm_max})={len(warm):,}, HOT(≥{hot_k})={len(hot):,}")
    print(f"[COUNT] pseudo-COLD(HOT as COLD)={len(pseudo):,}  (~{100.0*len(pseudo)/max(1,len(all_eval)):.2f}% of ALL)")

# ───────── forecasters ─────────
def make_normal_forecaster(model, scalers: Optional[Dict[str, Tuple[float, float]]], H: int, spatial_map: Dict[str,np.ndarray]):
    def _f(monthly_df: pd.DataFrame, cid: str, months_ctx: pd.DatetimeIndex, H_override: Optional[int]=None):
        H_use = H if H_override is None else H_override
        scaler = None if not isinstance(scalers, dict) else (scalers.get(cid) or scalers.get(str(cid)))
        sp = spatial_map.get(str(cid))
        return forecast_with_model(monthly_df, cid, months_ctx, model,
                                   T_in=T_IN, H=H_use, scaler_for_id=scaler, ctx_override=None, spatial_vec=sp)
    return _f

def make_synthctx_forecaster(model, master: pd.DataFrame, monthly: pd.DataFrame, trades: pd.DataFrame,
                             ids: List[str], X: np.ndarray, T_in: int, H: int, *,
                             gen_model: str = "econ_level_trend", gen_kwargs: Optional[dict] = None,
                             spatial_map: Dict[str,np.ndarray] = None):
    """
    gen_model ∈ {"basic","econ_add","econ_level_trend","mean_nosmooth","zero","radiff","csdi"}
    확산(radiff/csdi)은 generator.py에서 체크포인트/의존성 없으면 자동 폴백(econ_level_trend)
    """
    gen_kwargs = gen_kwargs or {}
    spatial_map = spatial_map or {}
    def _forecast(monthly_df: pd.DataFrame, cid: str, months_ctx: pd.DatetimeIndex, H_override: Optional[int]=None) -> np.ndarray:
        H_use = H if H_override is None else H_override
        # neighbors
        idx_arr = master.index[master["complex_id"].astype(str) == str(cid)]
        if len(idx_arr) == 0:
            ctx = np.zeros(T_IN, np.float32); mu, sd = float(np.nanmean(ctx)), float(np.nanstd(ctx)+1e-6)
            sp = spatial_map.get(str(cid))
            return forecast_with_model(monthly_df, cid, months_ctx, model, T_in=T_IN, H=H_use,
                                       scaler_for_id=(mu,sd), ctx_override=ctx, spatial_vec=sp)
        t_index = idx_arr[0]
        cand_idx = find_neighbors(ids, master, X, t_index)
        try:
            _, _, _, w, ordered_idx = neighbor_weights_hybrid(master=master, target_idx=t_index, cand_idx=cand_idx, X_meta=X,
                                                              sc_emb_target=None, sc_emb_pool=None, w_meta=0.55, w_geo=0.45, w_sem=0.0)
        except ValueError:
            _, _, _, w = neighbor_weights_hybrid(master=master, target_idx=t_index, cand_idx=cand_idx, X_meta=X,
                                                 sc_emb_target=None, sc_emb_pool=None, w_meta=0.55, w_geo=0.45, w_sem=0.0)
            ordered_idx = cand_idx[np.argsort(-w)]
        ids_all = [str(master.loc[i,"complex_id"]) for i in ordered_idx]
        avail = set(monthly_df["complex_id"].astype(str).unique())
        pairs = [(i,float(w[k] if k<len(w) else 0.0)) for k,i in enumerate(ids_all) if i in avail and (k < len(w)) and w[k]>0]
        if not pairs:
            ctx = np.zeros(T_IN, np.float32); mu, sd = float(np.nanmean(ctx)), float(np.nanstd(ctx)+1e-6)
            sp = spatial_map.get(str(cid))
            return forecast_with_model(monthly_df, cid, months_ctx, model, T_in=T_IN, H=H_use,
                                       scaler_for_id=(mu,sd), ctx_override=ctx, spatial_vec=sp)
        ids_kept = [p[0] for p in pairs]; w_kept = np.array([p[1] for p in pairs], np.float32)
        neigh_raw = monthly_df[monthly_df["complex_id"].astype(str).isin(ids_kept)].copy()
        ids_in = set(neigh_raw["complex_id"].astype(str).unique())
        pairs = [(i,w) for (i,w) in pairs if i in ids_in]
        if not pairs:
            ctx = np.zeros(T_IN, np.float32); mu, sd = float(np.nanmean(ctx)), float(np.nanstd(ctx)+1e-6)
            sp = spatial_map.get(str(cid))
            return forecast_with_model(monthly_df, cid, months_ctx, model, T_in=T_IN, H=H_use,
                                       scaler_for_id=(mu,sd), ctx_override=ctx, spatial_vec=sp)
        ids_kept = [p[0] for p in pairs]; w_kept = np.array([p[1] for p in pairs], np.float32); w_kept = w_kept/(w_kept.sum()+1e-12)
        scale_map = scale_align(monthly_df, cid, ids_kept)
        neigh = neigh_raw.copy()
        if "ppsm_median" in neigh.columns:
            neigh["ppsm_median"] = neigh.apply(lambda r: r["ppsm_median"] * scale_map.get(str(r["complex_id"]), 1.0), axis=1)

        # 합성 컨텍스트 생성
        ctx = build_synthetic_context(
            target_id=cid, months=months_ctx, neigh_monthly=neigh, neigh_ids=ids_kept,
            hybrid_weights=w_kept, T_in=T_IN,
            gen_model=gen_model,
            monthly_all=monthly, trades_all=trades,
            # generator 노브 일괄 전달
            recency_half_life=gen_kwargs.get("recency_half_life"),
            econ_corr_threshold=gen_kwargs.get("econ_corr_threshold", 0.80),
            econ_alpha_level=gen_kwargs.get("econ_alpha_level", 0.12),
            econ_beta_trend=gen_kwargs.get("econ_beta_trend", 0.20),
            econ_max_lag=gen_kwargs.get("econ_max_lag", 12),
            econ_disable_cache=gen_kwargs.get("econ_disable_cache", True),
            econ_add_strength=gen_kwargs.get("econ_add_strength", 0.25),
            diffusion_ckpt_radiff=gen_kwargs.get("diff_radiff_ckpt"),
            diffusion_ckpt_csdi=gen_kwargs.get("diff_csdi_ckpt"),
            diffusion_device=gen_kwargs.get("diff_device", "cpu"),
        ).astype(np.float32)

        mu, sd = float(np.nanmean(ctx)), float(np.nanstd(ctx)+1e-6)
        sp = spatial_map.get(str(cid))
        return forecast_with_model(monthly_df, cid, months_ctx, model, T_in=T_IN, H=H_use,
                                   scaler_for_id=(mu,sd), ctx_override=ctx, spatial_vec=sp)
    return _forecast

# ───────── evaluation ─────────
def evaluate_now(monthly: pd.DataFrame, ids: Iterable[str], forecast_fn, months_all: pd.DatetimeIndex,
                 anchor: pd.Timestamp, desc="NOW"):
    months_ctx = months_all[months_all < anchor]
    maes, rmses, mapes = [], [], []
    for cid in tqdm(list(ids), desc=desc, dynamic_ncols=True, mininterval=0.2):
        sub = (monthly.loc[monthly["complex_id"].astype(str)==str(cid), ["ym","ppsm_median"]]
               .drop_duplicates("ym").set_index("ym"))
        if anchor not in sub.index: continue
        gt = float(sub.loc[anchor,"ppsm_median"])
        try:
            pred = forecast_fn(monthly, str(cid), months_ctx, H_override=1)
        except Exception:
            continue
        if pred is None or len(pred)<1 or not np.isfinite(pred[0]): continue
        e = float(pred[0]-gt)
        maes.append(abs(e)); rmses.append(e*e); mapes.append(abs(e/gt)*100.0 if abs(gt)>1e-6 else np.nan)
    return {"MAE": float(np.nanmean(maes)) if maes else np.nan,
            "RMSE": float(np.sqrt(np.nanmean(rmses))) if rmses else np.nan,
            "MAPE": float(np.nanmean(mapes)) if mapes else np.nan}

def evaluate_future_multi(monthly: pd.DataFrame, ids: Iterable[str], forecast_fn, months_all: pd.DatetimeIndex,
                          anchor: pd.Timestamp, steps: List[int], desc="FUT(multi)"):
    """
    FUTURE 평가: 각 h에 대해 컨텍스트 끝을 anchor-h로 두고 h-스텝 예측의 마지막 값을 anchor 실측과 비교
    """
    sub_anchor = (monthly[["complex_id","ym","ppsm_median"]].drop_duplicates()
                  .pivot(index="ym", columns="complex_id", values="ppsm_median"))
    if anchor not in sub_anchor.index:
        return {h: {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan} for h in steps}
    gt_vec = sub_anchor.loc[anchor].astype(float)
    out: Dict[int, Dict[str,float]] = {}
    for h in steps:
        ctx_end = (anchor.to_period("M") - h).to_timestamp()
        months_ctx = months_all[months_all <= ctx_end]
        maes, rmses, mapes = [], [], []
        for cid in tqdm(list(ids), desc=f"{desc}:H={h}", dynamic_ncols=True, mininterval=0.2):
            if cid not in gt_vec.index: continue
            gt = float(gt_vec[cid])
            try:
                pred = forecast_fn(monthly, str(cid), months_ctx, H_override=h)
            except Exception:
                continue
            if pred is None or len(pred)!=h or not np.isfinite(pred[-1]): continue
            e = float(pred[-1]-gt)
            maes.append(abs(e)); rmses.append(e*e); mapes.append(abs(e/gt)*100.0 if abs(gt)>1e-6 else np.nan)
        out[h] = {"MAE": float(np.nanmean(maes)) if maes else np.nan,
                  "RMSE": float(np.sqrt(np.nanmean(rmses))) if rmses else np.nan,
                  "MAPE": float(np.nanmean(mapes)) if mapes else np.nan}
    return out

# ───────── main run ─────────
def run_once(seed: int, epochs: int, hot_k: int, anchor_str: Optional[str], gen_model: str, econ_off: bool,
             backbones: List[str], gen_knobs: Dict):
    logger = LiveLogger(os.path.join(OUTDIR, "live.log"))
    hb = Heartbeat(os.path.join(OUTDIR, "heartbeat.jsonl"))
    steps = ["Load+Embed+Train", "Prepare forecasters", "Evaluate NOW/FUT", "Save & Dashboard"]
    hb.write_beat(0, len(steps), tag=steps[0])

    np.random.seed(seed)
    print(f"[device] DEVICE={DEVICE}, AMP={'ON' if MIXED_PRECISION else 'OFF'}, AMP_DTYPE={AMP_DTYPE}", flush=True)

    # 1) load
    logger.section("Loading data"); live_log("start loading", path=os.path.join(OUTDIR, "live.log"), tag="load")
    with _Spinner("loading data..."): master, trades, monthly = load_data()
    print(f"✓ Data loaded | master={len(master):,}, monthly={len(monthly):,}")

    # 2) embeddings
    logger.section("Building embeddings"); live_log("start embed", path=os.path.join(OUTDIR, "live.log"), tag="embed")
    with _Spinner("building embeddings..."): ids, X = build_embeddings(master)

    # 3) time axis/anchor
    months_all = pd.date_range(monthly["ym"].min(), monthly["ym"].max(), freq="MS")
    anchor_now = pick_anchor_now(months_all, anchor_str); print(f"→ Anchor NOW set to {anchor_now:%Y-%m}")

    # spatial map (id→vector)
    sp_map: Dict[str,np.ndarray] = {}
    for i, cid in enumerate(master["complex_id"].astype(str).tolist()):
        sp_map[cid] = X[i]

    # 4) train (multi-backbone)
    logger.section("Training temporal models"); live_log(f"train backbones={backbones} epochs={epochs}", path=os.path.join(OUTDIR,"live.log"), tag="train")
    ep_bar = tqdm(total=epochs, desc="training(all)", dynamic_ncols=True)
    class _Reporter:
        def __init__(self, bar, hb, total): self.bar=bar; self.hb=hb; self.total=total; self.last=0
        def log(self, msg:str):
            if "ep" in msg:
                try:
                    k=int(''.join(c for c in msg.split("ep",1)[1].strip().split()[0] if c.isdigit()))
                    while self.last<k and self.last<self.bar.total:
                        self.last+=1; self.bar.update(1); self.hb.write_sub(self.last, self.total, tag="train")
                except Exception: pass
    reporter=_Reporter(ep_bar, hb, epochs)
    models_by_kind, scalers, sp_dim = train_temporal(
        monthly_df=monthly, train_ids=ids, months=months_all,
        model_kind=backbones, T_in=T_IN, H=H_OUT, epochs=epochs,
        reporter=reporter, seed=seed
    )
    if ep_bar.n<ep_bar.total: ep_bar.update(ep_bar.total-ep_bar.n); hb.write_sub(ep_bar.total, ep_bar.total, tag="train")
    ep_bar.close()

    # 5) cohorts @anchor
    logger.section("Build cohorts"); live_log(f"anchor={anchor_now:%Y-%m} hot_k={hot_k}", path=os.path.join(OUTDIR,"live.log"), tag="cohort")
    meta = build_cohorts_at_anchor(monthly, ids, anchor_now, H_OUT, warm_min=2, warm_max=10, hot_k=hot_k, include_all_hot_as_pseudo=True)
    ids_now = ids_with_obs_at(monthly, meta["all_eval"], anchor_now)
    true_cold = [c for c in meta["true_cold"] if c in ids_now]
    warm      = [c for c in meta["warm"]      if c in ids_now]
    hot       = [c for c in meta["hot"]       if c in ids_now]
    pseudo    = [c for c in meta["pseudo_cold"] if c in ids_now]
    all_eval  = ids_now[:]
    log_cohort_counts({"anchor_now":anchor_now,"all_eval":all_eval,"true_cold":true_cold,"warm":warm,"hot":hot,"pseudo_cold":pseudo},
                      hot_k, warm_min=2, warm_max=10)

    # 6) forecasters
    hb.write_beat(1, len(steps), tag=steps[1]); logger.section("Prepare forecasters")
    gen_used = "basic" if econ_off else (gen_model or "econ_level_trend")

    rows=[]
    hb.write_beat(2, len(steps), tag=steps[2])
    for kind, model in models_by_kind.items():
        print(f"\n=== Backbone: {kind} | gen={gen_used} ===")
        f_now_norm = make_normal_forecaster(model, scalers, 1, sp_map)
        f_fut_norm = make_normal_forecaster(model, scalers, H_OUT, sp_map)
        f_now_sctx = make_synthctx_forecaster(model, master, monthly, trades, ids, X, T_IN, 1,
                                              gen_model=gen_used, gen_kwargs=gen_knobs, spatial_map=sp_map)
        f_fut_sctx = make_synthctx_forecaster(model, master, monthly, trades, ids, X, T_IN, H_OUT,
                                              gen_model=gen_used, gen_kwargs=gen_knobs, spatial_map=sp_map)

        # NOW
        res_now_all  = evaluate_now(monthly, all_eval,  f_now_norm, months_all, anchor_now, desc=f"NOW:ALL[{kind}]")
        res_now_true = evaluate_now(monthly, true_cold, f_now_norm, months_all, anchor_now, desc=f"NOW:TRUE[{kind}]") if len(true_cold) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
        res_now_wN   = evaluate_now(monthly, warm,      f_now_norm, months_all, anchor_now, desc=f"NOW:WARM-N[{kind}]") if len(warm) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}
        res_now_wS   = evaluate_now(monthly, warm,      f_now_sctx, months_all, anchor_now, desc=f"NOW:WARM-S[{kind}]") if len(warm) else {"MAE":np.nan,"RMSE":np.nan,"MAPE":np.nan}

        # FUTURE multi-H
        fut_all_norm = evaluate_future_multi(monthly, all_eval,  f_fut_norm, months_all, anchor_now, FUT_STEPS, desc=f"FUT:ALL(norm,{kind})")
        fut_all_sctx = evaluate_future_multi(monthly, all_eval,  f_fut_sctx, months_all, anchor_now, FUT_STEPS, desc=f"FUT:ALL(sctx,{kind})")

        def push(split, d): rows.append({"Backbone":kind, "Gen":gen_used, "Split":split, **d})
        push("NOW_ALL", res_now_all); push("NOW_TRUE", res_now_true); push("NOW_WARM_NORM", res_now_wN); push("NOW_WARM_SYNTH", res_now_wS)
        for h,v in fut_all_norm.items(): push(f"FUT_ALL_NORM_H{h}", v)
        for h,v in fut_all_sctx.items(): push(f"FUT_ALL_SYNTH_H{h}", v)

    # 7) save
    hb.write_beat(3, len(steps), tag=steps[3])
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.DataFrame(rows)
    suffix = f"anchor_{anchor_now.strftime('%Y%m')}_FUTsteps_{'-'.join(map(str,FUT_STEPS))}"
    csv_path = os.path.join(OUTDIR, f"eval_results_multibackbone_{suffix}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {csv_path}")
    live_log(f"saved {os.path.basename(csv_path)}", path=os.path.join(OUTDIR,"live.log"), tag="save")

    # 8) dashboard
    try:
        beats = hb.latest(n=50, sub=False)
        dash = render_dashboard(beats, steps=steps, svg_path=os.path.join(OUTDIR, "dashboard.svg"))
        print(f"Rendered dashboard → {dash}")
    except Exception:
        pass

# ───────── CLI ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hot_k", type=int, default=60)
    ap.add_argument("--anchor_now", type=str, default="2025-05")

    # 합성 컨텍스트 선택 (확산 포함)
    ap.add_argument("--gen_model", type=str, default="econ_level_trend",
                    choices=["basic","econ_add","econ_level_trend","mean_nosmooth","zero","radiff","csdi"],
                    help="합성 컨텍스트 생성 방식")

    # 경제지표/최근성/확산 노브
    ap.add_argument("--econ_off", action="store_true", help="경제지표/확산 보정 비활성화(=basic)")
    ap.add_argument("--recency_half_life", type=int, default=12, help="이웃 최근성 half-life(개월). 0 이하이면 미사용")
    ap.add_argument("--econ_corr_threshold", type=float, default=0.80)
    ap.add_argument("--econ_alpha_level", type=float, default=0.12)
    ap.add_argument("--econ_beta_trend",  type=float, default=0.20)
    ap.add_argument("--econ_max_lag",     type=int,   default=12)
    ap.add_argument("--econ_disable_cache", type=int, choices=[0,1], default=1)
    ap.add_argument("--econ_add_strength", type=float, default=0.25)
    ap.add_argument("--diff_radiff_ckpt", type=str, default=os.path.join(OUTDIR, "radiff_ckpt.pt"))
    ap.add_argument("--diff_csdi_ckpt",   type=str, default=os.path.join(OUTDIR, "csdi_ckpt.pt"))
    ap.add_argument("--diff_device",      type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    # 백본 멀티 비교
    ap.add_argument("--backbones", type=str, default="gru,lstm,transformer,tcn",
                    help="comma-separated: gru,lstm,transformer,tcn")
    args = ap.parse_args()

    backs = [s.strip().lower() for s in args.backbones.split(",") if s.strip()]

    # generator에 전달할 knob 묶음
    gen_knobs = {
        "recency_half_life": args.recency_half_life,
        "econ_corr_threshold": args.econ_corr_threshold,
        "econ_alpha_level": args.econ_alpha_level,
        "econ_beta_trend": args.econ_beta_trend,
        "econ_max_lag": args.econ_max_lag,
        "econ_disable_cache": bool(args.econ_disable_cache),
        "econ_add_strength": args.econ_add_strength,
        "diff_radiff_ckpt": args.diff_radiff_ckpt,
        "diff_csdi_ckpt": args.diff_csdi_ckpt,
        "diff_device": args.diff_device,
    }

    run_once(seed=args.seed, epochs=args.epochs, hot_k=args.hot_k, anchor_str=args.anchor_now,
             gen_model=args.gen_model, econ_off=args.econ_off, backbones=backs, gen_knobs=gen_knobs)

if __name__ == "__main__":
    main()

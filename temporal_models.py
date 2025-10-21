# temporal_models.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import Iterable, List, Optional, Tuple, Dict, Union
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import sys, time

from config import DEVICE, MIXED_PRECISION, USE_TORCH_COMPILE, SEED, T_IN, H_OUT, AMP_DTYPE

# ───────── 기존 파일의 핵심: SlidingWindow/GRU/Transformer/AMP/ES 유지 ─────────
# (아래 기본 로직들은 원본을 보존하되, LSTM/TCN·공간융합·멀티백본 학습을 추가) :contentReference[oaicite:12]{index=12}

# ========== AMP helpers ==========
def _amp_dtype() -> torch.dtype:
    s = str(AMP_DTYPE).lower() if AMP_DTYPE is not None else "bf16"
    if s in ("bf16", "bfloat16"):
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
    return torch.float16

def _amp_enabled(dev: torch.device) -> bool:
    return bool(MIXED_PRECISION and dev.type == "cuda")

# ========== Data ==========
def build_month_index(monthly_df: pd.DataFrame, ids: Iterable[Union[str, int]]) -> pd.DatetimeIndex:
    sub = monthly_df[monthly_df["complex_id"].isin(ids)]
    if sub.empty:
        raise ValueError("monthly_df에 해당 ids가 없습니다.")
    mn, mx = pd.to_datetime(sub["ym"].min()), pd.to_datetime(sub["ym"].max())
    return pd.date_range(start=mn, end=mx, freq="MS")

def series_for_id(monthly_df: pd.DataFrame, cid, months: pd.DatetimeIndex) -> np.ndarray:
    s = (monthly_df.loc[monthly_df["complex_id"] == cid, ["ym", "ppsm_median"]]
         .drop_duplicates(subset=["ym"]).set_index("ym").reindex(months)["ppsm_median"]
         .ffill().bfill().astype(float).values).astype(np.float32)
    return s

class SlidingWindowDataset(Dataset):
    def __init__(self, monthly_df: pd.DataFrame, ids: Iterable[str|int], months: pd.DatetimeIndex,
                 T_in: int, min_points: int = 24, log_transform: bool = False, scale_per_series: bool = True):
        self.X, self.y, self.sid = [], [], []
        self.T_in = T_in; self.log_transform = log_transform; self.scale_per_series = scale_per_series
        self.scalers: Dict = {}
        min_required = max(min_points, T_in + 1)
        for cid in ids:
            v = series_for_id(monthly_df, cid, months)  # [T]
            if log_transform:
                v = np.log(np.clip(v, 1e-6, None)).astype(np.float32)
            mask = np.isfinite(v)
            if mask.sum() < min_required:
                continue
            if scale_per_series:
                mean = float(np.nanmean(v[mask])); std = float(np.nanstd(v[mask]) + 1e-6)
                v_norm = (v - mean) / (std + 1e-6); self.scalers[cid] = (mean, std)
            else:
                v_norm = v
            L = len(v_norm)
            for t in range(T_in, L):
                x = v_norm[t-T_in:t]; y = v_norm[t]
                if np.isfinite(y) and np.all(np.isfinite(x)):
                    self.X.append(x[:,None]); self.y.append(y); self.sid.append(cid)
        if len(self.X) == 0:
            raise ValueError("슬라이딩 윈도우 표본이 생성되지 않았습니다.")
        self.X = np.stack(self.X, axis=0).astype(np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): 
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.float32), self.sid[i]

# ========== Models (with optional spatial fusion) ==========
class _Fuse(nn.Module):
    """ST-RAP식 간단 게이트: concat 전에 h⊙s_head의 코사인으로 게이트."""
    def forward(self, h: torch.Tensor, s: Optional[torch.Tensor]) -> torch.Tensor:
        if s is None or s.numel() == 0: 
            return h
        s_head = s[:, :h.shape[-1]]
        hn = h / (h.norm(dim=-1, keepdim=True)+1e-6)
        sn = s_head / (s_head.norm(dim=-1, keepdim=True)+1e-6)
        g = torch.sigmoid((hn*sn).sum(-1, keepdim=True))
        return torch.cat([g*h, (1.0-g)*s], dim=-1)

class GRUReg(nn.Module):
    def __init__(self, sp_dim=0, hidden=64, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, batch_first=True, dropout=(dropout if 1>1 else 0.0))
        self.fuse = _Fuse()
        self.head = nn.Sequential(nn.Linear(hidden+sp_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x, sp: Optional[torch.Tensor]=None):
        out,_ = self.rnn(x); h = out[:,-1,:]
        h = self.fuse(h, sp); y = self.head(h)
        return y.squeeze(-1)

class LSTMReg(nn.Module):
    def __init__(self, sp_dim=0, hidden=64, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, batch_first=True, dropout=(dropout if 1>1 else 0.0))
        self.fuse = _Fuse()
        self.head = nn.Sequential(nn.Linear(hidden+sp_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x, sp=None):
        out,_ = self.rnn(x); h = out[:,-1,:]
        h = self.fuse(h, sp); y = self.head(h)
        return y.squeeze(-1)

class _TCNBlock(nn.Module):
    def __init__(self, ci, co, k=3, d=1):
        super().__init__()
        pad = (k-1)*d
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, k, padding=pad, dilation=d), nn.ReLU(),
            nn.Conv1d(co, co, k, padding=pad, dilation=d), nn.ReLU(),
        )
    def forward(self, x):  # (B,C,T)
        y = self.net(x); return y[...,:x.size(-1)]

class TCNReg(nn.Module):
    def __init__(self, sp_dim=0, hidden=64):
        super().__init__()
        self.tcn = nn.Sequential(_TCNBlock(1, hidden, 3, 1), _TCNBlock(hidden, hidden, 3, 2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fuse = _Fuse()
        self.head = nn.Sequential(nn.Linear(hidden+sp_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x, sp=None):
        z = self.tcn(x.transpose(1,2)); h = self.pool(z).squeeze(-1)  # (B,H)
        h = self.fuse(h, sp); y = self.head(h)
        return y.squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    def forward(self, x):
        T = x.size(1); return x + self.pe[:,:T,:]

class TFReg(nn.Module):
    def __init__(self, sp_dim=0, d_model=64, nhead=4, nlayers=2, ff=128, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(1, d_model); self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff,
                                         dropout=dropout, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fuse = _Fuse()
        self.head = nn.Sequential(nn.Linear(d_model+sp_dim, d_model), nn.GELU(), nn.Linear(d_model, 1))
    def forward(self, x, sp=None):
        z = self.pos(self.inp(x)); z = self.enc(z)
        h = self.pool(z.transpose(1,2)).squeeze(-1)
        h = self.fuse(h, sp); y = self.head(h)
        return y.squeeze(-1)

def _build(kind: str, sp_dim: int):
    k = kind.lower()
    if k == "gru": return GRUReg(sp_dim=sp_dim)
    if k == "lstm": return LSTMReg(sp_dim=sp_dim)
    if k == "tcn": return TCNReg(sp_dim=sp_dim)
    if k == "transformer": return TFReg(sp_dim=sp_dim)
    raise ValueError(f"Unknown model kind: {kind}")

# ========== Train (single or multi-backbone) ==========
def train_temporal(
    monthly_df: pd.DataFrame,
    train_ids,
    months: Optional[pd.DatetimeIndex] = None,
    model_kind: Union[str, List[str]] = "gru",
    T_in: int = T_IN,
    H: int = H_OUT,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    reporter=None,
    log_transform: bool = False,
    scale_per_series: bool = True,
    seed: Optional[int] = None,
    val_split: float = 0.10,
    early_stop_patience: int = 5,
    min_delta: float = 1e-4,
    spatial_map: Optional[Dict[str, np.ndarray]] = None  # 미사용(학습때는 마지막 한점 예측), 추후 확장 가능
) -> Tuple[Dict[str, nn.Module], Dict[str, Tuple[float,float]], int]:
    """
    반환: (models_by_kind, scalers_per_id, sp_dim)
    - model_kind가 문자열이면 [kind]로 처리.
    """
    if months is None:
        months = pd.date_range(monthly_df["ym"].min(), monthly_df["ym"].max(), freq="MS")

    kinds = [model_kind] if isinstance(model_kind, str) else list(model_kind)
    ds = SlidingWindowDataset(monthly_df, list(train_ids), months, T_in,
                              min_points=max(24,T_in+6), log_transform=log_transform, scale_per_series=scale_per_series)

    # (간단 랜덤 분할)
    do_val = (val_split and len(ds) >= 10)
    if do_val:
        from torch.utils.data import random_split
        n_total = len(ds); n_val = max(1, int(round(n_total*float(val_split)))); n_tr = max(1, n_total - n_val)
        g = torch.Generator(); g.manual_seed(SEED if seed is None else seed)
        ds_tr, ds_val = random_split(ds, [n_tr, n_val], generator=g)
    else:
        ds_tr, ds_val = ds, None

    device = torch.device(DEVICE)
    dl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2, pin_memory=(device.type=="cuda"))
    dl_val = None
    if ds_val is not None:
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2, pin_memory=(device.type=="cuda"))

    # 스케일러(ID별)
    scalers: Dict[str, Tuple[float,float]] = getattr(ds, "scalers", {})

    sp_dim = 0  # 추후 학습 입력에 공간벡터 포함하려면 여기 확장

    models_by_kind: Dict[str, nn.Module] = {}
    for kind in kinds:
        net = _build(kind, sp_dim=sp_dim).to(device)
        if hasattr(torch, "compile") and USE_TORCH_COMPILE:
            try: net = torch.compile(net)
            except Exception: pass
        opt = Adam(net.parameters(), lr=lr)
        loss_fn = nn.L1Loss()
        scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled(device))

        best = float("inf"); best_state=None; patience=early_stop_patience
        for ep in range(1, epochs+1):
            net.train(); losses=[]
            last_tick=[0.0]; tick=[0]; spinner="|/-\\"
            def pulse(b, total):
                now=time.time()
                if now-last_tick[0] >= 0.5:
                    pct= (b/max(1,total))*100.0
                    sys.stdout.write("\r{} {} ep {}/{} batch {}/{} {:5.1f}%".format(kind, spinner[tick[0]%4], ep, epochs, b, total, pct))
                    sys.stdout.flush(); tick[0]+=1; last_tick[0]=now
            total_batches=len(dl)
            for bi,(xb,yb,_) in enumerate(dl,1):
                pulse(bi,total_batches)
                xb=xb.to(device); yb=yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=_amp_dtype(), enabled=_amp_enabled(device)):
                    yhat=net(xb, None)  # 학습은 순수 시계열로
                    loss=loss_fn(yhat, yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                losses.append(float(loss.detach().item()))
            sys.stdout.write(f"\r✓ {kind} ep {ep}/{epochs} train MAE={np.mean(losses):.4f}       \n")

            # val
            val_mae = np.mean(losses)
            if dl_val is not None:
                net.eval(); v=[]
                with torch.no_grad():
                    for xb,yb,_ in dl_val:
                        xb=xb.to(device); yb=yb.to(device)
                        with torch.amp.autocast("cuda", dtype=_amp_dtype(), enabled=_amp_enabled(device)):
                            vloss = loss_fn(net(xb, None), yb)
                        v.append(float(vloss.detach().item()))
                val_mae = float(np.mean(v)) if v else float("inf")
                print(f"   → val MAE={val_mae:.4f}")
            if reporter is not None and hasattr(reporter,"log"):
                reporter.log(f"[TEMP-{kind}] ep {ep:03d} train_MAE={float(np.mean(losses)):.4f} val_MAE={val_mae:.4f}")

            # early stop
            if (best - val_mae) > float(min_delta):
                best = val_mae; best_state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                patience = early_stop_patience; print("   ✓ best updated")
            else:
                patience -= 1; print(f"   no improvement (patience left: {patience})")
                if patience <= 0:
                    print(f"→ Early stopping {kind} at ep {ep} (best={best:.4f})"); break
        if best_state is not None:
            net.load_state_dict(best_state)
        models_by_kind[kind] = net.eval()

    return models_by_kind, scalers, sp_dim

# ========== Forecast ==========
@torch.no_grad()
def forecast_with_model(
    monthly_df: pd.DataFrame,
    cid,
    months: pd.DatetimeIndex,
    model: nn.Module,
    T_in: int = T_IN,
    H: int = H_OUT,
    scaler_for_id: Optional[Tuple[float, float]] = None,
    log_transform: bool = False,
    ctx_override: Optional[np.ndarray] = None,   # 합성 컨텍스트
    spatial_vec: Optional[np.ndarray] = None     # (옵션) 공간벡터 → 모델이 fuse 처리
) -> np.ndarray:
    """
    (기존 함수 유지 + spatial_vec 인자 추가)
    재귀로 H-스텝 예측, 각 스텝에서 입력 꼬리를 갱신.
    """
    device = next(model.parameters()).device
    s = (monthly_df.loc[monthly_df["complex_id"] == cid, ["ym","ppsm_median"]]
         .drop_duplicates("ym").set_index("ym").reindex(months)["ppsm_median"]
         .ffill().bfill().astype(float).values).astype(np.float32)

    if log_transform:
        s = np.log(np.clip(s, 1e-6, None)).astype(np.float32)

    if scaler_for_id is None:
        tail = s[-T_in:] if len(s) >= T_in else s
        mean = float(np.nanmean(tail)); std = float(np.nanstd(tail) + 1e-6)
    else:
        mean, std = scaler_for_id

    if ctx_override is not None:
        ctx = ctx_override.astype(np.float32)
        if log_transform: ctx = np.log(np.clip(ctx, 1e-6, None)).astype(np.float32)
        ctx = (ctx - mean)/(std+1e-6)
        if len(ctx) != T_in:
            raise ValueError(f"ctx_override length must be {T_in}, got {len(ctx)}")
    else:
        s_norm = (s - mean)/(std+1e-6)
        if len(s_norm) < T_in:
            pad = np.repeat(s_norm[:1], T_in - len(s_norm))
            ctx = np.concatenate([pad, s_norm], axis=0).astype(np.float32)
        else:
            ctx = s_norm[-T_in:].astype(np.float32)

    sp_t = None if spatial_vec is None else torch.from_numpy(spatial_vec.reshape(1,-1)).to(device=device, dtype=torch.float32)
    preds_norm=[]
    for _ in range(H):
        xb = torch.from_numpy(ctx.reshape(1, T_in, 1)).to(device=device, dtype=torch.float32)
        with torch.amp.autocast("cuda", dtype=_amp_dtype(), enabled=_amp_enabled(device)):
            yhat = model(xb, sp_t)
        yhat = float(yhat.squeeze(0).detach().cpu().item())
        preds_norm.append(yhat)
        ctx = np.concatenate([ctx[1:], np.array([yhat], dtype=np.float32)], axis=0)
    preds = np.asarray(preds_norm, np.float32)*(std+1e-6) + mean
    if log_transform: preds = np.exp(preds)
    return preds

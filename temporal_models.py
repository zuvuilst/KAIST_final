# temporal_models.py
from __future__ import annotations
import math
from typing import Iterable, List, Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys, time
from progress_utils import Heartbeat, print_step

from config import (
    DEVICE, MIXED_PRECISION, USE_TORCH_COMPILE, SEED,
    T_IN, H_OUT, AMP_DTYPE
)
try:
    from config import AMP_DTYPE
except Exception:
    AMP_DTYPE = "bf16"

def _amp_dtype() -> torch.dtype:
    s = str(AMP_DTYPE).lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16
# =========================
# Utilities / AMP helpers
# =========================
def set_seed(seed: int = 42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _log(msg: str, reporter=None):
    if reporter is not None and hasattr(reporter, "log"):
        reporter.log(msg)
    else:
        print(msg)

def _get_device() -> torch.device:
    try:
        return torch.device(DEVICE)
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _amp_enabled(dev: torch.device) -> bool:
    # AMP는 CUDA에서만 동작
    return bool(MIXED_PRECISION and dev.type == "cuda")

def _amp_dtype() -> torch.dtype:
    # bf16 선호(지원되면), 아니면 fp16
    dt = str(AMP_DTYPE).lower() if AMP_DTYPE is not None else "bf16"
    if dt in ("bf16", "bfloat16"):
        if torch.cuda.is_available():
            # torch 2.0+는 is_bf16_supported() 있음
            ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            if ok:
                return torch.bfloat16
    return torch.float16

def _scaler_for(dev: torch.device) -> "torch.amp.GradScaler":
    # bf16에서는 스케일러를 켜면 의미가 없음 → disabled
    dtype = _amp_dtype()
    use_scaler = _amp_enabled(dev) and (dtype is torch.float16)
    return torch.amp.GradScaler("cuda", enabled=use_scaler)


# =========================
# Data utilities
# =========================
def build_month_index(monthly_df: pd.DataFrame, ids: Iterable[Union[str, int]]) -> pd.DatetimeIndex:
    sub = monthly_df[monthly_df["complex_id"].isin(ids)]
    if sub.empty:
        raise ValueError("monthly_df에 해당 ids가 없습니다.")
    mn, mx = sub["ym"].min(), sub["ym"].max()
    mn = pd.to_datetime(mn); mx = pd.to_datetime(mx)
    return pd.date_range(start=mn, end=mx, freq="MS")

def series_for_id(monthly_df: pd.DataFrame, cid, months: pd.DatetimeIndex) -> np.ndarray:
    s = (
        monthly_df.loc[monthly_df["complex_id"] == cid, ["ym", "ppsm_median"]]
        .drop_duplicates(subset=["ym"])
        .set_index("ym")
        .reindex(months)["ppsm_median"]
        .ffill()
        .bfill()
        .astype(float)
        .values
    )
    return s


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        monthly_df: pd.DataFrame,
        ids: Iterable[str | int],
        months: pd.DatetimeIndex,
        T_in: int,
        min_points: int = 24,
        log_transform: bool = False,
        scale_per_series: bool = True,
    ):
        self.X, self.y, self.sid = [], [], []
        self.T_in = T_in
        self.log_transform = log_transform
        self.scale_per_series = scale_per_series
        self.scalers: Dict = {}

        min_required = max(min_points, T_in + 1)

        for cid in ids:
            v = series_for_id(monthly_df, cid, months)  # [T_all]
            v = np.asarray(v, dtype=np.float32)

            if self.log_transform:
                v = np.log(np.clip(v, 1e-6, None)).astype(np.float32)

            finite_mask = np.isfinite(v)
            if finite_mask.sum() < min_required:
                continue

            if scale_per_series:
                mean = float(np.nanmean(v[finite_mask]))
                std = float(np.nanstd(v[finite_mask]))
                if not np.isfinite(mean):
                    continue
                if not np.isfinite(std) or std == 0.0:
                    std = 1e-6
                v_norm = (v - mean) / std
                self.scalers[cid] = (mean, std)
            else:
                v_norm = v

            L = len(v_norm)
            for t in range(T_in, L):
                x = v_norm[t - T_in: t]
                y = v_norm[t]
                if not (np.isfinite(y) and np.all(np.isfinite(x))):
                    continue
                self.X.append(x[:, None])
                self.y.append(y)
                self.sid.append(cid)

        if len(self.X) == 0:
            raise ValueError(
                "슬라이딩 윈도우 표본이 생성되지 않았습니다. "
                "원인: (1) ids의 유효 월별 값이 부족, (2) T_in/min_points 과대, (3) 결측 과다."
            )

        self.X = np.stack(self.X, axis=0).astype(np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            self.sid[idx]
        )


# =========================
# Models
# =========================
class GRURegressor(nn.Module):
    def __init__(self, input_dim=1, hidden=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden, num_layers=num_layers,
                          batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        h_last = out[:, -1, :]
        y = self.head(h_last)
        return y.squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=2048)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
    def forward(self, x):
        z = self.input_proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        h_last = z[:, -1, :]
        y = self.head(h_last)
        return y.squeeze(-1)

def build_temporal_model(kind: str = "gru", **kwargs) -> nn.Module:
    if kind.lower() == "gru":
        model = GRURegressor(**kwargs)
    elif kind.lower() == "transformer":
        model = TransformerRegressor(**kwargs)
    else:
        raise ValueError(f"Unknown temporal model kind: {kind}")
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass
    return model


# =========================
# Training
# =========================
def train_temporal(
    monthly_df: pd.DataFrame,
    train_ids,
    months: Optional[pd.DatetimeIndex] = None,
    model_kind: str = "gru",
    T_in: int = T_IN,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    reporter=None,
    log_transform: bool = False,
    scale_per_series: bool = True,
    seed: Optional[int] = None,
    # ▼ Early Stopping 옵션
    val_split: float = 0.10,           # 전체 표본 중 검증 비율
    early_stop_patience: int = 5,      # 개선 없을 때 허용 에폭 수
    min_delta: float = 1e-4,           # “개선”으로 인정할 최소 감소량
):
    # --- 시드 ---
    def set_seed(seed_: int = 42):
        import random, os
        random.seed(seed_)
        np.random.seed(seed_)
        torch.manual_seed(seed_)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_)
    set_seed(SEED if seed is None else seed)

    if months is None:
        months = pd.date_range(monthly_df["ym"].min(), monthly_df["ym"].max(), freq="MS")

    # ▶ 슬라이딩 윈도우 데이터셋
    ds = SlidingWindowDataset(
        monthly_df=monthly_df, ids=list(train_ids), months=months, T_in=T_in,
        min_points=max(24, T_in + 6), log_transform=log_transform, scale_per_series=scale_per_series
    )

    # ▶ Train/Val 분할 (표본 기준 무작위 분할)
    do_val = (val_split is not None) and (val_split > 0.0) and (len(ds) >= 10)
    if do_val:
        from torch.utils.data import random_split
        n_total = len(ds)
        n_val = max(1, int(round(n_total * float(val_split))))
        n_tr  = max(1, n_total - n_val)
        g = torch.Generator()
        g.manual_seed(SEED if seed is None else seed)
        ds_tr, ds_val = random_split(ds, [n_tr, n_val], generator=g)
    else:
        ds_tr, ds_val = ds, None

    # DataLoader
    device = torch.device(DEVICE)
    dl = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    dl_val = None
    if ds_val is not None:
        dl_val = DataLoader(
            ds_val, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=2, pin_memory=(device.type == "cuda")
        )

    # --- 모델 ---
    if model_kind.lower() == "gru":
        model = build_temporal_model("gru", input_dim=1, hidden=64, num_layers=1, dropout=0.1)
        tag = "TEMP-gru"
    else:
        model = build_temporal_model("transformer", input_dim=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1)
        tag = "TEMP-tf"

    model = model.to(device)
    loss_fn = nn.L1Loss()
    opt = Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(MIXED_PRECISION and device.type == "cuda"))

    # --- 하트비트/스피너 (간단)
    import sys, time
    spinner = "|/-\\"
    def pulse(ep, b, total_b, last_tick, tick_idx):
        now = time.time()
        if now - last_tick[0] >= 0.5:
            pct = (b / max(1, total_b)) * 100.0
            sys.stdout.write(
                "\rtraining ep {}/{} [{}] batch {}/{}  {:5.1f}%".format(
                    ep, epochs, spinner[tick_idx[0] % len(spinner)], b, total_b, pct
                )
            )
            sys.stdout.flush()
            tick_idx[0] += 1
            last_tick[0] = now

    # --- Early Stopping 상태
    best_val = float("inf")
    best_state = None
    patience_left = int(early_stop_patience)

    # --- 학습 루프 ---
    for ep in range(1, epochs + 1):
        model.train()
        mae_meter = []
        total_batches = len(dl)
        last_tick = [0.0]; tick_idx = [0]
        t0 = time.time()

        # 에폭 시작 알림
        print(f"\n→ training ep {ep}/{epochs} (batches={total_batches}, val={do_val})", flush=True)

        for b_idx, (xb, yb, _) in enumerate(dl, 1):
            pulse(ep, b_idx, total_batches, last_tick, tick_idx)

            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=_amp_dtype(), enabled=(MIXED_PRECISION and device.type == "cuda")):
                pred = model(xb)
                loss = loss_fn(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            mae_meter.append(loss.detach().float().item())
            pulse(ep, b_idx, total_batches, last_tick, tick_idx)

        # Train 요약
        dt = time.time() - t0
        tr_mae = float(np.mean(mae_meter)) if mae_meter else float("nan")
        sys.stdout.write(f"\r✓ ep {ep}/{epochs} train done in {dt:.1f}s | MAE={tr_mae:.4f}            \n")
        sys.stdout.flush()

        # --- Validation MAE ---
        val_mae = tr_mae
        if do_val and dl_val is not None:
            model.eval()
            v_losses = []
            with torch.no_grad():
                for xb, yb, _ in dl_val:
                    xb = xb.to(device=device, dtype=torch.float32)
                    yb = yb.to(device=device, dtype=torch.float32)
                    with torch.amp.autocast("cuda", dtype=_amp_dtype(), enabled=(MIXED_PRECISION and device.type == "cuda")):
                        yhat = model(xb)
                        vloss = loss_fn(yhat, yb)
                    v_losses.append(vloss.detach().float().item())
            val_mae = float(np.mean(v_losses)) if v_losses else float("inf")
            print(f"   → val MAE={val_mae:.4f}")

        # Reporter 로그
        if reporter is not None and hasattr(reporter, "log"):
            reporter.log(f"[{tag}] ep {ep:03d} train_MAE={tr_mae:.4f} val_MAE={val_mae:.4f}")

        # --- Early Stopping 판정 (검증 우선, 없으면 train 기준) ---
        metric = val_mae if do_val else tr_mae
        improved = (best_val - metric) > float(min_delta)
        if improved:
            best_val = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(early_stop_patience)
            print(f"   ✓ best updated → metric={best_val:.4f}")
        else:
            patience_left -= 1
            print(f"   no improvement (patience left: {patience_left})")
            if patience_left <= 0:
                print(f"→ Early stopping at epoch {ep} (best={best_val:.4f}).")
                break

    # --- 가장 좋은 가중치로 복원 ---
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"✓ Loaded best weights (best metric={best_val:.4f})")

    if hasattr(reporter, "log"):
        reporter.log(f"[temporal] model trained with early stopping: {model_kind}")
    else:
        print(f"[temporal] model trained with early stopping: {model_kind}", flush=True)
    return model, (ds.scalers if hasattr(ds, 'scalers') else {})


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
    ctx_override: Optional[np.ndarray] = None,  # 합성 히스토리 직접 주입
) -> np.ndarray:
    # 모델의 실제 device 가져오기
    device = next(model.parameters()).device

    # 시계열 불러오기
    s = (
        monthly_df.loc[monthly_df["complex_id"] == cid, ["ym", "ppsm_median"]]
        .drop_duplicates(subset=["ym"]).set_index("ym").reindex(months)["ppsm_median"]
        .ffill().bfill().astype(float).values
    ).astype(np.float32)

    # 로그 변환
    if log_transform:
        s = np.log(np.clip(s, 1e-6, None)).astype(np.float32)

    # 스케일러 결정
    if scaler_for_id is None:
        tail = s[-T_in:] if len(s) >= T_in else s
        mean = float(np.nanmean(tail))
        std  = float(np.nanstd(tail) + 1e-6)
    else:
        mean, std = scaler_for_id

    # 컨텍스트 구성
    if ctx_override is not None:
        ctx = ctx_override.astype(np.float32)
        if log_transform:
            ctx = np.log(np.clip(ctx, 1e-6, None)).astype(np.float32)
        ctx = (ctx - mean) / (std + 1e-6)
        if len(ctx) != T_in:
            raise ValueError(f"ctx_override length must be {T_in}, got {len(ctx)}")
    else:
        s_norm = (s - mean) / (std + 1e-6)
        if len(s_norm) < T_in:
            pad = np.repeat(s_norm[:1], T_in - len(s_norm))
            ctx = np.concatenate([pad, s_norm], axis=0).astype(np.float32)
        else:
            ctx = s_norm[-T_in:].astype(np.float32)

    # 재귀 예측
    preds_norm = []
    model.eval()
    for _ in range(H):
        xb = torch.from_numpy(ctx.reshape(1, T_in, 1)).to(device=device, dtype=torch.float32)
        with torch.amp.autocast(
            "cuda",
            dtype=_amp_dtype(),
            enabled=(MIXED_PRECISION and device.type == "cuda")
        ):
            yhat = model(xb)
        yhat = float(yhat.squeeze(0).detach().cpu().item())
        preds_norm.append(yhat)
        ctx = np.concatenate([ctx[1:], np.array([yhat], dtype=np.float32)], axis=0)

    preds_norm = np.asarray(preds_norm, dtype=np.float32)
    preds = preds_norm * (std + 1e-6) + mean
    if log_transform:
        preds = np.exp(preds)
    return preds

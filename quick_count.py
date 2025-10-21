# quick_count_now.py
# monthly.csv에서 "NOW=전역 마지막에서 offset개월 전"에 관측이 있는 매물 수와
# 코호트( TRUE-COLD=1, WARM=2~10, HOT>=60 ) 분포를 즉시 계산.

import os
import sys
import argparse
import numpy as np
import pandas as pd

try:
    from config import OUTDIR
except Exception:
    OUTDIR = os.getenv("APT_OUTDIR", r"C:\KAIST_APT_Trade\out")

MONTHLY_CSV = os.path.join(OUTDIR, "monthly.csv")

def pick_now_month(monthly: pd.DataFrame, offset: int = 1) -> pd.Timestamp:
    """NOW = 전역 마지막에서 offset개월 전 (offset>=1).
       월이 충분치 않으면 가능한 가장 이른 값으로 fallback."""
    uniq = (
        monthly["ym"]
        .drop_duplicates()
        .sort_values()
        .to_list()
    )
    if not uniq:
        raise ValueError("monthly.csv에 유효한 ym 값이 없습니다.")
    # offset이 너무 크면 맨 앞월, 너무 작으면 맨 끝월로 보정
    offset = max(0, int(offset))
    if offset == 0:
        # offset=0이면 마지막 달(기본적으로 권장하지 않음)
        return pd.to_datetime(uniq[-1])
    if len(uniq) >= (offset + 1):
        return pd.to_datetime(uniq[-(offset + 1)])
    # 월이 부족하면 맨 앞/뒤에서 가능한 값으로
    return pd.to_datetime(uniq[0])

def main(offset: int, show_ids: int = 0, save_list: bool = True):
    if not os.path.exists(MONTHLY_CSV):
        print(f"[ERR] monthly.csv를 찾지 못했습니다: {MONTHLY_CSV}")
        print("→ 먼저 dataset.py를 실행해 OUTDIR에 master.csv / trades.csv / monthly.csv를 생성하세요.")
        sys.exit(1)

    # 데이터 로드
    monthly = pd.read_csv(MONTHLY_CSV, parse_dates=["ym"], low_memory=False)
    monthly["ppsm_median"] = pd.to_numeric(monthly["ppsm_median"], errors="coerce")
    monthly["complex_id"]  = monthly["complex_id"].astype(str)

    # NOW = 전역 마지막에서 offset개월 전 (기본 offset=1 → '마지막 바로 전 달')
    now_m = pick_now_month(monthly, offset=offset)

    # NOW 시점의 실제 관측(유효 숫자) 보유 매물들
    now_slice = monthly[(monthly["ym"] == now_m)].copy()
    now_slice = now_slice[
        now_slice["ppsm_median"]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
    ]
    ids_now = now_slice["complex_id"].drop_duplicates().astype(str).tolist()

    # 코호트 계산 (NOW 이전의 관측 개수 = pre)
    monthly_unique = monthly[["complex_id", "ym", "ppsm_median"]].drop_duplicates(["complex_id","ym"])
    pre_counts = (
        monthly_unique[monthly_unique["ym"] < now_m]
        .groupby("complex_id")["ym"].count().astype(int)
    )
    pre_now = pre_counts[pre_counts.index.isin(ids_now)].copy()

    # 규칙: TRUE-COLD=1, WARM=2~10, HOT>=60
    n_all  = len(ids_now)
    n_true = int((pre_now == 1).sum())
    n_warm = int(((pre_now >= 2) & (pre_now <= 10)).sum())
    n_hot  = int((pre_now >= 60).sum())

    print(f"\n[OUTDIR] {OUTDIR}")
    print(f"[NOW=penultimate] 전역 기준월 = {now_m:%Y-%m}  (offset={offset})")
    print(f"[COUNT] NOW 관측 보유 매물 수 = {n_all:,}")
    print(f"  - TRUE-COLD (pre=1)  : {n_true:,}")
    print(f"  - WARM (2~10)        : {n_warm:,}")
    print(f"  - HOT (>=60)         : {n_hot:,}")

    if show_ids > 0:
        print("\n[IDs] NOW 관측 보유 매물 예시:")
        for cid in ids_now[:show_ids]:
            pre_val = int(pre_now.get(cid, 0))
            print(f"  - {cid}  (pre={pre_val})")

    if save_list:
        os.makedirs(OUTDIR, exist_ok=True)
        out_txt = os.path.join(OUTDIR, f"ids_with_now_offset{offset}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for cid in ids_now:
                f.write(str(cid) + "\n")
        print(f"\n[Saved] NOW(ID) 목록 → {out_txt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=1, help="NOW = 전역 마지막에서 offset개월 전 (기본=1 → 바로 전 달)")
    ap.add_argument("--show_ids", type=int, default=20, help="예시로 보여줄 ID 개수")
    ap.add_argument("--no_save", action="store_true", help="ID 리스트 파일 저장하지 않음")
    args = ap.parse_args()

    main(offset=args.offset, show_ids=args.show_ids, save_list=(not args.no_save))

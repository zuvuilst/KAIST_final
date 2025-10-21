# dataset.py
import os, json, time, requests
import numpy as np, pandas as pd
from tqdm import tqdm
from config import DEFAULT_CSVS, ADMIN_XLSX, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, OUTDIR

NAVER_GEOCODE_URL = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"

def geocode_batch(addresses, client_id, client_secret, sleep=0.12):
    headers = {"X-NCP-APIGW-API-KEY-ID": client_id, "X-NCP-APIGW-API-KEY": client_secret}
    out = {}
    for a in tqdm(addresses, desc="[geocode] 주소→좌표", unit="addr"):
        if not isinstance(a, str) or not a.strip():
            out[a] = (np.nan, np.nan); continue
        try:
            r = requests.get(NAVER_GEOCODE_URL, headers=headers, params={"query": a}, timeout=6)
            if r.status_code == 200:
                d = r.json()
                if d.get("addresses"):
                    x = d["addresses"][0].get("x"); y = d["addresses"][0].get("y")
                    out[a] = (float(y), float(x))
                else:
                    out[a] = (np.nan, np.nan)
            else:
                out[a] = (np.nan, np.nan)
        except Exception:
            out[a] = (np.nan, np.nan)
        time.sleep(sleep)
    return out

def build_and_save(outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    print(f"[data] loading raw CSVs:")
    dfs = []
    for p in DEFAULT_CSVS:
        print("  -", p)
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)

    keep = ['legal_district_code','legal_town_code','road_name','legal_district_name',
            'apartment_complex_name','parcel_number','exclusive_use_area','transaction_amount',
            'floor_number','construction_year','apartment_complex_sequence','contract_date']
    df = df[keep].copy()

    print(f"[data] merge admin: {ADMIN_XLSX}")
    df_k = pd.read_excel(ADMIN_XLSX)[['광역시도 명','시군구 코드','시군구 명']].drop_duplicates()
    merged = pd.merge(df, df_k, left_on='legal_district_code', right_on='시군구 코드', how='left')
    df = merged[['legal_district_code','legal_town_code','road_name','legal_district_name',
                 'apartment_complex_name','parcel_number','exclusive_use_area','transaction_amount',
                 'floor_number','construction_year','apartment_complex_sequence','contract_date',
                 '광역시도 명','시군구 명']].copy()

    df['지번주소'] = (
        df['광역시도 명'].astype(str) + ' ' +
        df['시군구 명'].astype(str)   + ' ' +
        df['legal_district_name'].astype(str) + ' ' +
        df['parcel_number'].astype(str)
    )

    addr_unique = df['지번주소'].dropna().astype(str).unique().tolist()

    cache_path = os.path.join(outdir, "geocode_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, "r", encoding="utf-8"))

    to_query = [a for a in addr_unique if a not in cache]
    if to_query:
        print(f"[data] geocoding {len(to_query)} / {len(addr_unique)} (나머지는 캐시)")
        new = geocode_batch(to_query, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
        cache.update(new)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    coords = df['지번주소'].astype(str).map(cache)
    df["위도"] = coords.map(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)
    df["경도"] = coords.map(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)
    df = df.dropna(subset=["위도", "경도"]).reset_index(drop=True)

    # master
    master = (df.groupby(["apartment_complex_sequence","apartment_complex_name","지번주소"])
                .agg(lat=("위도","median"),
                    lon=("경도","median"),
                    construction_year=("construction_year","median"))
                .reset_index()
                .rename(columns={"apartment_complex_sequence":"complex_id"}))

    # trades
    trades = df.rename(columns={"apartment_complex_sequence":"complex_id", "contract_date":"ymd"})
    trades["ym"] = pd.to_datetime(trades["ymd"]).dt.to_period("M").dt.to_timestamp()

    # monthly (median price per sqm)
    def to_ppsm(r):
        try:
            a = float(r["exclusive_use_area"]); p = float(r["transaction_amount"])
            return p / max(a, 1e-6)
        except Exception:
            return np.nan
    trades["ppsm"] = trades.apply(to_ppsm, axis=1)
    monthly = (trades.groupby(["complex_id","ym"])
                    .agg(ppsm_median=("ppsm","median"))
                    .reset_index())

    # ===== CSV로 저장 =====
    master_path  = os.path.join(outdir, "master.csv")
    trades_path  = os.path.join(outdir, "trades.csv")
    monthly_path = os.path.join(outdir, "monthly.csv")
    master.to_csv(master_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")

    print(f"[data] saved CSV ->")
    print(f"  - {master_path}")
    print(f"  - {trades_path}")
    print(f"  - {monthly_path}")
    print(f"[data] master={len(master):,}, monthly={len(monthly):,}, trades={len(trades):,}")
    return master, monthly, trades

if __name__ == "__main__":
    build_and_save()

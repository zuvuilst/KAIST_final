# econ_api.py
from __future__ import annotations
import os, json, hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import requests
from pathlib import Path


# 외부 config (네 코드 유지)
from config import OUTDIR, CACHE_DIR

# =========================================================
# 설정/키 (환경변수로 덮어쓰기 가능)
# =========================================================
ECOS_API_KEY = os.getenv("ECOS_API_KEY", "P1AS9UY4C9ATQRVMBNWZ").strip()
RONE_API_KEY = os.getenv("RONE_API_KEY", "fb5f352e07b34e49ba46a8d8bb7628eb").strip()
RONE_BASE_URL = os.getenv("RONE_BASE_URL", "https://www.reb.or.kr/r-one/openapi").rstrip("/")

USER_AGENT = "APT-Trade-Econ/1.0 (analytics)"
TIMEOUT    = 30

ECON_CACHE = Path(CACHE_DIR) / "econ"
ECON_CACHE.mkdir(parents=True, exist_ok=True)
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

# R-ONE는 헤더가 민감함 (Referer 필수)
RONE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json,*/*",
    "Referer": "https://www.reb.or.kr/",
    "Connection": "close",
}

def _cache_path(prefix: str, params: Dict) -> Path:
    key = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h   = hashlib.md5(key).hexdigest()
    return ECON_CACHE / f"{prefix}_{h}.json"

def _http_get_json(url: str, params: Dict=None, headers: Dict=None) -> Optional[Dict]:
    pa = params or {}
    hd = {"User-Agent": USER_AGENT}
    if headers:
        hd.update(headers)
    try:
        # 네트워크/인증 이슈 우회 위해 verify=False
        r = requests.get(url, params=pa, headers=hd, timeout=TIMEOUT, verify=False)
        r.raise_for_status()
        txt = r.text.strip()
        if not txt:
            return None
        try:
            return r.json()
        except Exception:
            return json.loads(txt)
    except requests.exceptions.RequestException as e:
        print(f"[오류] ❌ API 네트워크 오류: {e}")
        return None
    except json.JSONDecodeError:
        print(f"[오류] ❌ JSON 파싱 실패 (앞 200자): {r.text[:200]}...")
        return None

# =========================================================
# 1) 한국은행 ECOS API
# =========================================================
@dataclass
class EcosSeries:
    stat_code: str
    item_code1: str = ""
    item_code2: str = ""
    cycle: str = "M"
    start: str = "2010-01"
    end: str   = "2025-12"
    col_name: str = ""

def _ecos_find_rows(payload: Dict) -> List[Dict]:
    """ECOS 응답에서 row 리스트를 유연하게 추출"""
    if not isinstance(payload, dict):
        return []
    # 보통: {"StatisticSearch":[{...},{ "row":[...] }]}
    block = payload.get("StatisticSearch")
    if isinstance(block, list):
        for b in block:
            if isinstance(b, dict) and isinstance(b.get("row"), list):
                return b["row"]
    if isinstance(block, dict) and isinstance(block.get("row"), list):
        return block["row"]
    # 혹시 다른 키 계층에 있을 수도 있음
    for v in payload.values():
        if isinstance(v, list):
            for b in v:
                if isinstance(b, dict) and isinstance(b.get("row"), list):
                    return b["row"]
    return []

def ecos_fetch(series: List[EcosSeries]) -> pd.DataFrame:
    if not ECOS_API_KEY:
        print("[econ][ECOS] ECOS_API_KEY 없음 → ECOS 스킵.")
        return pd.DataFrame(index=pd.DatetimeIndex([], name="ym"))

    base = "https://ecos.bok.or.kr/api/StatisticSearch"
    frames = []
    for s in series:
        params = dict(
            api_key=ECOS_API_KEY, stat_code=s.stat_code, item_code1=s.item_code1,
            item_code2=s.item_code2, cycle=s.cycle, start=s.start, end=s.end
        )
        cache = _cache_path("ecos", params)
        use_cache = os.getenv("ECON_DISABLE_CACHE", "") != "1"
        if use_cache and cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                cache.unlink(missing_ok=True); data = None
        else:
            bits = [ECOS_API_KEY, "json", "kr", "1", "100000", s.stat_code, s.cycle,
                    s.start.replace("-", ""), s.end.replace("-", "")]
            if s.item_code1: bits.append(s.item_code1)
            if s.item_code2: bits.append(s.item_code2)
            url = "/".join([base] + bits)
            data = _http_get_json(url)
            if use_cache and data:
                cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        rows = _ecos_find_rows(data or {})
        if not rows:
            # print(f"[econ][ECOS] 빈 응답 stat={s.stat_code} item={s.item_code1}")
            continue

        recs = []
        for r in rows:
            t = str(r.get("TIME", "")).strip()  # YYYYMM
            if not t or len(t) < 6: 
                continue
            try:
                ym = pd.to_datetime(t[:6], format="%Y%m")
            except Exception:
                continue
            v_raw = str(r.get("DATA_VALUE", "")).replace(",", "")
            v = pd.to_numeric(v_raw, errors="coerce")
            recs.append({"ym": ym, "value": v})

        if not recs:
            continue
        df = pd.DataFrame(recs).dropna().drop_duplicates("ym").set_index("ym").sort_index()
        cname = s.col_name or f"{s.stat_code}.{s.item_code1 or 'ALL'}.{s.item_code2 or ''}".strip(".")
        df.columns = [cname]
        frames.append(df)

    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="ym"))
    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "ym"
    return out

def fetch_ecos_default(start="2010-01", end="2025-12") -> pd.DataFrame:
    ecos_list = [
        EcosSeries(stat_code="722Y001", item_code1="0101000", start=start, end=end, col_name="ecos_base_rate"),
        EcosSeries(stat_code="901Y009", item_code1="0",       start=start, end=end, col_name="ecos_cpi_2020"),
        EcosSeries(stat_code="901Y067", item_code1="I16A",    start=start, end=end, col_name="ecos_ci_2020"),
        EcosSeries(stat_code="901Y104", item_code1="I48A",    start=start, end=end, col_name="ecos_construction_put"),
        EcosSeries(stat_code="901Y020", item_code1="I42A",    start=start, end=end, col_name="ecos_construction_orders"),
        EcosSeries(stat_code="901Y074", item_code1="I410R",   start=start, end=end, col_name="ecos_unsold"),
    ]
    return ecos_fetch(ecos_list)

# =========================================================
# 2) R-ONE(부동산통계정보) API
# =========================================================
@dataclass
class ROneSeries:
    dataset_code: str
    region_code: Optional[str] = None
    start: str = "201001"   # YYYYMM
    end: str   = "202512"   # YYYYMM
    col_name: str = ""

def _rone_rows(payload: Dict) -> List[Dict]:
    """R-ONE 응답에서 row 리스트 추출 (테이블마다 키 구조가 다름)"""
    if not isinstance(payload, dict):
        return []
    # 대표 3개 블록 우선
    for key in ("SttsApiTblData", "SttsApiTbl", "SttsApiTblItm"):
        blk = payload.get(key)
        if isinstance(blk, list):
            # 보통 [ {헤더}, { "row":[...] } ] 형태
            for part in blk:
                if isinstance(part, dict) and isinstance(part.get("row"), list):
                    return part["row"]
        elif isinstance(blk, dict) and isinstance(blk.get("row"), list):
            return blk["row"]
    # 혹시 다른 키에 숨어있는 경우도 대비
    for v in payload.values():
        if isinstance(v, list):
            for b in v:
                if isinstance(b, dict) and isinstance(b.get("row"), list):
                    return b["row"]
    return []

def _rone_fetch_one(s: ROneSeries) -> pd.DataFrame:
    if not RONE_API_KEY:
        print("[econ][RONE] RONE_API_KEY 없음 → R-ONE 스킵.")
        return pd.DataFrame()

    params = {"dataset": s.dataset_code, "region": s.region_code or "nation", "start": s.start, "end": s.end}
    cache = _cache_path("rone_v20_hdr_parse_debug", params)
    use_cache = os.getenv("ECON_DISABLE_CACHE", "") != "1"
    if use_cache and cache.exists():
        try:
            df_cached = pd.read_json(cache, orient='split')
            df_cached['ym'] = pd.to_datetime(df_cached['ym'])
            return df_cached.set_index('ym')
        except Exception:
            cache.unlink(missing_ok=True)

    url = f"{RONE_BASE_URL}/SttsApiTblData.do"
    all_recs: List[Dict] = []

    ys, ye = int(s.start[:4]), int(s.end[:4])
    for year in range(ys, ye + 1):
        st = f"{year}{'01' if year>ys else s.start[4:]}"
        ed = f"{year}{'12' if year<ye else s.end[4:]}"
        page = 1
        while True:
            q = {
                "KEY": RONE_API_KEY,
                "STATBL_ID": s.dataset_code,
                "DTACYCLE_CD": "MM",
                "START_WRTTIME": st,
                "END_WRTTIME": ed,
                "Type": "json",
                "pIndex": page,
                "pSize": 1000,
            }
            if s.region_code:
                q["CLS_ID"] = s.region_code

            data = _http_get_json(url, q, headers=RONE_HEADERS)
            rows = _rone_rows(data or {})
            if not rows:
                break

            # 날짜/값 컬럼 자동 탐색
            date_candidates = ("WRTTIME", "WRITNG_YM", "STD_MT", "WRT_YM", "YYMM", "WRTTIME_IDTFR_ID")
            value_candidates = ("DTVAL_CO", "DTVAL", "VALUE", "PRC", "IDX", "AMT", "DTA_VAL")

            parsed_cnt = 0
            for r in rows:
                # 날짜
                t = None
                for dc in date_candidates:
                    if dc in r and r[dc]:
                        t = str(r[dc]); break
                if not t or len(t) < 6:
                    continue
                try:
                    ym = pd.to_datetime(t[:6], format="%Y%m")
                except Exception:
                    continue
                # 값
                v = None
                for vc in value_candidates:
                    if vc in r and r[vc] not in (None, ""):
                        try:
                            v = float(str(r[vc]).replace(",", ""))
                            break
                        except Exception:
                            pass
                if v is None:
                    continue

                all_recs.append({"ym": ym, "value": v})
                parsed_cnt += 1

            # 디버그: row는 있는데 파싱을 하나도 못 했을 때 샘플 키를 보여줌
            if parsed_cnt == 0 and rows:
                sample = rows[0]
                print(f"[econ][RONE][DEBUG] 파싱 실패 stat={s.dataset_code} rows>0 "
                      f"sample_keys={list(sample.keys())[:15]} sample={str(sample)[:180]}")

            if len(rows) < q["pSize"]:
                break
            page += 1

    if not all_recs:
        print(f"[econ][RONE] '{s.col_name or s.dataset_code}' 데이터 없음.")
        return pd.DataFrame(index=pd.DatetimeIndex([], name="ym"))

    out = pd.DataFrame(all_recs).dropna().drop_duplicates("ym").set_index("ym").sort_index()
    out.columns = [s.col_name or s.dataset_code]
    if use_cache:
        try:
            out.reset_index().to_json(cache, orient='split', force_ascii=False)
        except Exception:
            pass
    return out

def rone_fetch(series: List[ROneSeries]) -> pd.DataFrame:
    frames = [df for s in series if not (df := _rone_fetch_one(s)).empty]
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="ym"))
    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "ym"
    return out

def fetch_rone_default(start="201001", end="202512") -> pd.DataFrame:
    NATION_CODE = "500001"         # 전국
    CAPITAL_REGION_CODE = "500002" # 수도권
    ser = [
        ROneSeries("A_2024_00901", NATION_CODE,         start, end, "rone_land_price_idx_nation"),
        ROneSeries("A_2024_00016", CAPITAL_REGION_CODE, start, end, "rone_house_price_idx_capital"),
        ROneSeries("A_2024_00018", CAPITAL_REGION_CODE, start, end, "rone_rent_integrated_idx_capital"),
        ROneSeries("A_2024_00027", CAPITAL_REGION_CODE, start, end, "rone_median_sale_price_capital"),
        ROneSeries("A_2024_00045", CAPITAL_REGION_CODE, start, end, "rone_house_price_idx_apt_capital"),
        ROneSeries("A_2024_00076", CAPITAL_REGION_CODE, start, end, "rone_market_supply_demand_apt_capital"),
        ROneSeries("A_2024_00060", CAPITAL_REGION_CODE, start, end, "rone_avg_sale_price_apt_capital"),
        ROneSeries("A_2024_00062", CAPITAL_REGION_CODE, start, end, "rone_median_sale_price_apt_capital"),
        ROneSeries("A_2024_00069", CAPITAL_REGION_CODE, start, end, "rone_avg_rent_price_apt_capital"),
    ]
    return rone_fetch(ser)

# =========================================================
# 3) 통합 프리셋
# =========================================================
def fetch_default_indicators(start="2010-01", end="2025-12") -> pd.DataFrame:
    # ECOS (YYYY-MM)
    df_ecos = fetch_ecos_default(start=start, end=end)
    # R-ONE (YYYYMM)
    s2, e2 = start.replace("-", ""), end.replace("-", "")
    df_rone = fetch_rone_default(start=s2, end=e2)

    if df_ecos.empty and df_rone.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="ym"))

    df = pd.concat([df_ecos, df_rone], axis=1)

    # 월단위 인덱스 정규화
    df = df.groupby(df.index.to_period("M")).mean()
    df.index = df.index.to_timestamp()
    df.index.name = "ym"
    return df.sort_index()

# =========================================================
# 4) 실행 진입점: 그냥 실행 → CSV 저장
# =========================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="R-ONE + ECOS 통합 수집 (헤더/파서 보강)")
    ap.add_argument("--start", default="2010-01")
    ap.add_argument("--end",   default="2025-12")
    ap.add_argument("--csv",   default=str(Path(OUTDIR) / "econ_combined.csv"))
    args = ap.parse_args()

    df = fetch_default_indicators(start=args.start, end=args.end)
    if df.empty:
        print("[warn] 수집된 데이터가 없습니다.")
    else:
        out = df.reset_index().rename(columns={"ym": "date"})
        out.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"[저장] {args.csv}  shape={df.shape}")
        print("열 예시:", [c for c in out.columns if c != "date"][:12])

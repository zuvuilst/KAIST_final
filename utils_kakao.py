#utils_kakao.py
import os
import math
import time
from typing import Dict, List, Optional, Callable
from viz import viz
import requests
import pandas as pd

from config import KAKAO_REST_API_KEY, CACHE_DIR, USE_CACHE

R_EARTH = 6378137.0
KAKAO_CACHE_DIR = os.path.join(CACHE_DIR, "kakao_poi")
os.makedirs(KAKAO_CACHE_DIR, exist_ok=True)

class KakaoMapClient:
    BASE_URL = "https://dapi.kakao.com/v2/local"

    def __init__(self, rest_api_key: Optional[str] = None, pause: float = 0.1, timeout: float = 5.0):
        key = rest_api_key or KAKAO_REST_API_KEY
        if not key:
            raise ValueError("Kakao REST API Key가 필요합니다.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"KakaoAK {key}"})
        self.pause = pause
        self.timeout = timeout

    def _request(self, path: str, params: Dict) -> Dict:
        url = f"{self.BASE_URL}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {"documents": [], "meta": {"is_end": True}}

    def search_all_pages(self, endpoint: str, base_params: Dict) -> List[Dict]:
        rows = []; pages = 0
        for page in range(1, 46):
            params = base_params.copy(); params["page"] = page
            data = self._request(endpoint, params)
            docs = data.get("documents", [])
            if not docs: break
            rows.extend(docs); pages += 1
            if data.get("meta", {}).get("is_end", True): break
            time.sleep(self.pause)
        viz.log(f"[kakao] endpoint={endpoint.split('/')[-1]} pages={pages} rows={len(rows)}")
        return rows

def tile_centers_wgs84(lon: float, lat: float, tile_count_per_axis: int = 3, tile_step_m: int = 800):
    centers = []
    offset = (tile_count_per_axis - 1) // 2
    for i in range(tile_count_per_axis):
        for j in range(tile_count_per_axis):
            dx = (i - offset) * tile_step_m
            dy = (j - offset) * tile_step_m
            dlon = (dx / (R_EARTH * math.cos(math.radians(lat)))) * (180.0 / math.pi)
            dlat = (dy / R_EARTH) * (180.0 / math.pi)
            centers.append((lon + dlon, lat + dlat))
    return centers

def haversine_m(lon1, lat1, lon2, lat2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R_EARTH*c

def process_tiled_results(
    docs: List[Dict], center_lon: float, center_lat: float, radius_m: int,
    filter_func: Optional[Callable[[Dict], bool]] = None
):
    seen = set()
    uniq = []
    for d in docs:
        did = d.get("id")
        if did and did not in seen:
            seen.add(did)
            uniq.append(d)

    if filter_func:
        uniq = [d for d in uniq if filter_func(d)]

    final_list = []
    for d in uniq:
        try:
            lon = float(d["x"]); lat = float(d["y"])
            dist = haversine_m(center_lon, center_lat, lon, lat)
            if dist <= radius_m:
                d["distance"] = str(int(dist))
                final_list.append(d)
        except Exception:
            continue

    final_list.sort(key=lambda z: int(z.get("distance", "999999")))
    return final_list

def _cache_key(center_lon: float, center_lat: float, radius_m: int, cfg_name: str) -> str:
    lonr = round(center_lon, 5); latr = round(center_lat, 5)
    return os.path.join(KAKAO_CACHE_DIR, f"{cfg_name}_{lonr}_{latr}_{radius_m}.json")

class AmenityFinder:
    def __init__(self, client: KakaoMapClient, center_lon: float, center_lat: float, radius_m: int,
                 tile_cnt: int = 3, tile_step_m: int = 800, inner_radius: int = 1200):
        self.client = client
        self.lon = center_lon
        self.lat = center_lat
        self.radius = radius_m
        self.tiles = tile_centers_wgs84(center_lon, center_lat, tile_cnt, tile_step_m)
        self.inner_radius = inner_radius

    def collect(self, config: Dict) -> List[Dict]:
        cache_fp = _cache_key(self.lon, self.lat, self.radius, config["name"])
        if USE_CACHE and os.path.exists(cache_fp):
            try:
                return pd.read_json(cache_fp, orient="records", dtype=False).to_dict(orient="records")
            except Exception:
                pass

        endpoint = "/search/keyword.json" if config["type"] == "KEYWORD" else "/search/category.json"
        all_docs = []
        for tx, ty in self.tiles:
            params = {"x": tx, "y": ty, "radius": self.inner_radius, "size": 15, "sort": "accuracy"}
            if config["type"] == "KEYWORD":
                params["query"] = config["code_or_keyword"]
            else:
                params["category_group_code"] = config["code_or_keyword"]
            docs_tile = self.client.search_all_pages(endpoint, params)
            all_docs.extend(docs_tile)

        final_list = process_tiled_results(all_docs, self.lon, self.lat, self.radius, config.get("filter_func"))

        if USE_CACHE:
            try:
                pd.DataFrame(final_list).to_json(cache_fp, orient="records", force_ascii=False)
            except Exception:
                pass

        return final_list

    def collect_many(self, configs: List[Dict]) -> Dict[str, List[Dict]]:
        out = {}
        for cfg in configs:
            out[cfg["name"]] = self.collect(cfg)
        return out

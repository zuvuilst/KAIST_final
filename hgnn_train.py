# hgnn_train.py
# 좌표 반경 입력 → 카카오 편의시설 수집 → HeteroData → HGT 임베딩 학습 → 슈퍼카테고리 임베딩
from __future__ import annotations

import sys
import types
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from config import (
    SEARCH_RADIUS_M, DEVICE, SEED
)

# ────────────────────────────────────────────────────────────
# viz.py가 없어도 utils_kakao가 동작하도록 'viz' 더미를 주입
# (utils_kakao 내부에서 viz.viz.log를 참조하는 경우가 있어 안전장치)
# ────────────────────────────────────────────────────────────
try:
    from utils_kakao import KakaoMapClient, AmenityFinder
except Exception:
    # viz 더미 모듈 주입
    dummy_viz_mod = types.ModuleType("viz")

    class _VizDummy:
        def log(self, msg: str, **kv):
            # 조용히 무시하거나, 원하면 print(msg)로 바꿔도 됨
            print(str(msg))

    dummy_viz_mod.viz = _VizDummy()
    sys.modules["viz"] = dummy_viz_mod
    # 재시도
    from utils_kakao import KakaoMapClient, AmenityFinder

from utils_graph import GraphBuilder, HGTModel, sanitize_heterodata


# 수집 대상 카테고리 구성 (기존과 동일)
FACILITIES_CONFIG = [
    {"name": "유치원", "type": "CATEGORY", "code_or_keyword": "PS3"},
    {"name": "초등학교", "type": "CATEGORY", "code_or_keyword": "SC4",
     "filter_func": lambda d: "초등학교" in d.get("category_name", "")},
    {"name": "중학교", "type": "CATEGORY", "code_or_keyword": "SC4",
     "filter_func": lambda d: "중학교" in d.get("category_name", "")},
    {"name": "고등학교", "type": "CATEGORY", "code_or_keyword": "SC4",
     "filter_func": lambda d: "고등학교" in d.get("category_name", "")},
    {"name": "대학교", "type": "CATEGORY", "code_or_keyword": "SC4",
     "filter_func": lambda d: "대학교" in d.get("category_name", "")},
    {"name": "학원_국영수보습", "type": "CATEGORY", "code_or_keyword": "AC5",
     "filter_func": lambda d: any(k in d.get("place_name", "") for k in ["보습","입시","영어","수학","국어","논술","과학"])},
    {"name": "학원_예체능", "type": "CATEGORY", "code_or_keyword": "AC5",
     "filter_func": lambda d: any(k in d.get("place_name", "") for k in ["미술","음악","피아노","체육","태권도","발레","무용","댄스","요가","필라테스"])},
    {"name": "지하철역", "type": "CATEGORY", "code_or_keyword": "SW8"},
    {"name": "버스정류장", "type": "KEYWORD",  "code_or_keyword": "버스정류장"},
    {"name": "고속도로IC", "type": "KEYWORD",  "code_or_keyword": "IC"},
    {"name": "주차장", "type": "CATEGORY", "code_or_keyword": "PK6"},
    {"name": "관광명소", "type": "CATEGORY", "code_or_keyword": "AT4"},
    {"name": "공공기관", "type": "CATEGORY", "code_or_keyword": "PO3"},
    {"name": "부동산중개소", "type": "CATEGORY", "code_or_keyword": "AG2"},
    {"name": "마트", "type": "CATEGORY", "code_or_keyword": "MT1"},
    {"name": "편의점", "type": "CATEGORY", "code_or_keyword": "CS2"},
    {"name": "종합상급병원", "type": "CATEGORY", "code_or_keyword": "HP8",
     "filter_func": lambda d: any(k in d.get("place_name","") for k in ["종합병원","대학병원","대학교병원","상급종합병원"])},
    {"name": "약국", "type": "CATEGORY", "code_or_keyword": "PM9"},
    {"name": "문화시설", "type": "CATEGORY", "code_or_keyword": "CT1"},
    {"name": "은행", "type": "CATEGORY", "code_or_keyword": "BK9"},
    {"name": "식당", "type": "CATEGORY", "code_or_keyword": "FD6"},
    {"name": "카페", "type": "CATEGORY", "code_or_keyword": "CE7"},
    {"name": "주유소", "type": "CATEGORY", "code_or_keyword": "OL7"},
]


def build_graph_from_kakao(lon: float, lat: float, radius_m: int = SEARCH_RADIUS_M):
    """
    주어진 (lon, lat) 반경 내 POI를 카카오에서 수집 → HeteroData 그래프 구성
    반환: (data: HeteroData, complex_df, poi_df, edge_df)
    """
    # 지연 import (torch_geometric 없을 때 파일 import 단계 오류 방지)
    from torch_geometric.data import HeteroData

    client = KakaoMapClient()
    finder = AmenityFinder(client, center_lon=lon, center_lat=lat, radius_m=radius_m)
    blocks: Dict[str, List[dict]] = {}

    print("[HGNN] collecting POIs ...")
    for cfg in tqdm(FACILITIES_CONFIG, ncols=100, desc="[Kakao] categories"):
        docs = finder.collect(cfg)
        print(f"[HGNN] collected {cfg['name']}: {len(docs)}")
        blocks[cfg["name"]] = docs

    # 수집된 POI가 하나도 없으면 최소 그래프 반환
    total_poi = sum(len(v) for v in blocks.values())
    if total_poi == 0:
        data = HeteroData()
        data["complex"].x = torch.zeros((1, 1), dtype=torch.float32)   # bias만
        data["poi"].x     = torch.zeros((0, 3), dtype=torch.float32)   # super-cat 3차원 가정
        data["complex","near_by","poi"].edge_index = torch.empty((2,0), dtype=torch.long)
        cdf = pd.DataFrame([{"complex_id":"TARGET","lon":lon,"lat":lat,"bias":1.0}])
        return data, cdf, pd.DataFrame(), pd.DataFrame()

    # 표 생성
    gb = GraphBuilder()
    complex_row = gb.add_complex("TARGET", lon, lat)
    for name, docs in blocks.items():
        gb.ingest_block(complex_row, name, docs)

    poi_df, edge_df = gb.build_tables()
    complex_df = pd.DataFrame([complex_row])

    data = gb.to_heterodata(complex_df, poi_df, edge_df)
    data = sanitize_heterodata(data)
    return data, complex_df, poi_df, edge_df


def train_hgnn_for_target(
    lon: float,
    lat: float,
    radius_m: int = SEARCH_RADIUS_M,
    epochs: int = 40,
    hidden: int = 64
) -> Dict[str, np.ndarray]:
    """
    하나의 타깃 단지 주변 그래프를 구축하고 HGT로 임베딩을 학습,
    슈퍼카테고리(교육/교통/편의시설) 평균 임베딩을 반환.
    """
    torch.manual_seed(SEED)

    data, cdf, pdf, edf = build_graph_from_kakao(lon, lat, radius_m)

    n_complex = (data["complex"].x.size(0) if "complex" in data.node_types else 0)
    n_poi     = (data["poi"].x.size(0)     if "poi"     in data.node_types else 0)
    n_edge    = (data["complex","near_by","poi"].edge_index.size(1)
                 if ("complex","near_by","poi") in data.edge_types else 0)

    print(f"[HGNN] summary: complex={n_complex}, poi={n_poi}, edges={n_edge}")
    if n_complex == 0 or n_poi == 0 or n_edge == 0:
        print("[HGNN] empty graph → return zero embeddings")
        z = np.zeros(hidden, dtype=np.float32)
        return {"교육": z, "교통": z, "편의시설": z}

    # 모델
    metadata = (list(data.node_types), list(data.edge_types))
    model = HGTModel(
        in_dim_complex=data["complex"].x.shape[1],
        in_dim_poi=data["poi"].x.shape[1],
        hidden=hidden, heads=2, layers=2, metadata=metadata
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 학습 데이터 준비
    edge = data["complex","near_by","poi"].edge_index
    pos_dst = edge[1].to(torch.long)
    n_poi = data["poi"].x.shape[0]

    try:
        model.train()
        for ep in range(1, epochs + 1):
            opt.zero_grad(set_to_none=True)
            out = model(data.to(DEVICE))
            c_emb = out["complex"].x[0]   # [H]
            p_emb = out["poi"].x          # [N, H]

            pos_vec = p_emb[pos_dst]                    # 양성(실제 연결)
            pos_score = torch.sum(c_emb * pos_vec, dim=1)

            neg_idx = torch.randint(0, n_poi, (len(pos_dst),), device=p_emb.device)
            neg_vec = p_emb[neg_idx]                    # 음성(무작위)
            neg_score = torch.sum(c_emb * neg_vec, dim=1)

            # margin ranking (hinge)
            loss = torch.relu(1.0 - pos_score + neg_score).mean()
            loss.backward()
            opt.step()

            if ep % 10 == 0 or ep == 1 or ep == epochs:
                print(f"[HGNN] ep {ep:03d} loss={float(loss):.4f}")

        # 슈퍼카테고리 평균 임베딩 산출
        with torch.no_grad():
            out = model(data.to(DEVICE))
            p_emb = out["poi"].x.detach().cpu().numpy()  # [N, H]
            super_names = ["교육", "교통", "편의시설"]
            sc_vec: Dict[str, List[np.ndarray]] = {s: [] for s in super_names}
            if not pdf.empty and "super_cat" in pdf.columns:
                for i, s in enumerate(pdf["super_cat"].tolist()):
                    if s in sc_vec:
                        sc_vec[s].append(p_emb[i])
            sc_emb = {
                k: (np.stack(v, axis=0).mean(axis=0) if len(v) > 0 else np.zeros(p_emb.shape[1], dtype=np.float32))
                for k, v in sc_vec.items()
            }
        return sc_emb

    except Exception as e:
        print(f"[HGNN] training failed → {e}\n[HGNN] fallback to zero embeddings")
        z = np.zeros(hidden, dtype=np.float32)
        return {"교육": z, "교통": z, "편의시설": z}

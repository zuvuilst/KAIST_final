#utils_graph.py
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import HeteroData

from config import SUPER_CAT_MAP

def distance_bucket(d: float) -> int:
    if d < 300: return 0
    if d < 500: return 1
    if d < 1000: return 2
    if d < 2000: return 3
    return 4

class GraphBuilder:
    def __init__(self):
        self.poi_rows: List[Dict] = []
        self.edge_rows: List[Dict] = []
        self.poi_id_map: Dict[str, int] = {}

    def add_complex(self, complex_id: str, lon: float, lat: float) -> Dict:
        return {"complex_id": complex_id, "lon": lon, "lat": lat, "bias": 1.0}

    def ingest_block(self, complex_row: Dict, fac_name: str, docs: List[Dict]):
        for d in docs:
            pid = d.get("id")
            if pid is None:
                continue
            if pid not in self.poi_id_map:
                self.poi_id_map[pid] = len(self.poi_rows)
                self.poi_rows.append({
                    "poi_id": pid,
                    "name": d.get("place_name", ""),
                    "super_cat": SUPER_CAT_MAP.get(fac_name, "편의시설"),
                    "raw_cat": d.get("category_name", ""),
                    "lon": float(d.get("x", 0.0)),
                    "lat": float(d.get("y", 0.0)),
                })
            dist_m = float(d.get("distance", "999999"))
            self.edge_rows.append({
                "complex_id": complex_row["complex_id"],
                "poi_id": pid,
                "dist_m": dist_m,
                "bucket": distance_bucket(dist_m),
                "fac_name": fac_name
            })

    def build_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        poi_df = pd.DataFrame(self.poi_rows) if self.poi_rows else pd.DataFrame(
            columns=["poi_id","name","super_cat","raw_cat","lon","lat"]
        )
        edge_df = pd.DataFrame(self.edge_rows) if self.edge_rows else pd.DataFrame(
            columns=["complex_id","poi_id","dist_m","bucket","fac_name"]
        )
        return poi_df, edge_df

    def to_heterodata(self, complex_df: pd.DataFrame, poi_df: pd.DataFrame, edge_df: pd.DataFrame) -> HeteroData:
        super_cats = sorted(set(SUPER_CAT_MAP.values()))
        sc2i = {s:i for i,s in enumerate(super_cats)}

        cx = complex_df[["bias"]].astype(np.float32).to_numpy() if not complex_df.empty else np.zeros((0,1),np.float32)
        px = np.zeros((len(poi_df), len(super_cats)), dtype=np.float32)
        if not poi_df.empty:
            for i, s in enumerate(poi_df["super_cat"].fillna("편의시설").tolist()):
                if s in sc2i: px[i, sc2i[s]] = 1.0

        cid2i = {cid:i for i,cid in enumerate(complex_df["complex_id"].tolist())}
        pid2i = {pid:i for i,pid in enumerate(poi_df["poi_id"].tolist())}

        src = [cid2i[c] for c in edge_df["complex_id"].tolist()] if not edge_df.empty else []
        dst = [pid2i[p] for p in edge_df["poi_id"].tolist()] if not edge_df.empty else []

        data = HeteroData()
        data["complex"].x = torch.tensor(cx, dtype=torch.float32)
        data["poi"].x = torch.tensor(px, dtype=torch.float32)

        if len(src) > 0 and len(dst) > 0:
            ei = torch.tensor([src, dst], dtype=torch.long)
        else:
            ei = torch.empty((2,0), dtype=torch.long)

        data["complex","near_by","poi"].edge_index = ei
        if not edge_df.empty:
            data["complex","near_by","poi"].edge_bucket = torch.tensor(edge_df["bucket"].astype(int).values, dtype=torch.long)
        else:
            data["complex","near_by","poi"].edge_bucket = torch.empty((0,), dtype=torch.long)

        return data

def sanitize_heterodata(data: HeteroData) -> HeteroData:
    to_delete = []
    for rel in list(data.edge_index_dict.keys()):
        ei = data.edge_index_dict[rel]
        src_t, _, dst_t = rel
        if ei.numel() == 0:
            to_delete.append(rel)
            continue
        if data[src_t].x.size(0) == 0 or data[dst_t].x.size(0) == 0:
            to_delete.append(rel)
    for rel in to_delete:
        del data[rel]
    if len(list(data.node_types)) == 0:
        return data
    return data

class HGTModel(nn.Module):
    def __init__(self, in_dim_complex: int, in_dim_poi: int, hidden: int = 64, heads: int = 2, layers: int = 2, metadata=None):
        super().__init__()
        from torch_geometric.nn import HGTConv
        self.map_c = nn.Linear(in_dim_complex, hidden)
        self.map_p = nn.Linear(in_dim_poi, hidden)
        self._layers = layers
        self._heads = heads
        self._hidden = hidden
        self._convs = None
        self._norms = None
        self._metadata = metadata

    def _ensure_convs(self, metadata):
        from torch_geometric.nn import HGTConv
        if (self._convs is not None) and (self._metadata == metadata):
            return
        self._metadata = metadata
        self._convs = nn.ModuleList([HGTConv(in_channels=self._hidden, out_channels=self._hidden, metadata=metadata, heads=self._heads) for _ in range(self._layers)])
        self._norms = nn.ModuleList([nn.LayerNorm(self._hidden) for _ in range(self._layers)])

    def forward(self, data: HeteroData) -> HeteroData:
        dev = next(self.parameters()).device
        # 입력 매핑 + 장치 정렬
        x_c_in = data["complex"].x if "complex" in data.node_types else torch.zeros((0,1), dtype=torch.float32, device=dev)
        x_p_in = data["poi"].x     if "poi"     in data.node_types else torch.zeros((0,1), dtype=torch.float32, device=dev)
        x_c = self.map_c(x_c_in.to(dev))
        x_p = self.map_p(x_p_in.to(dev))

        out = HeteroData()
        has_rel = ("complex","near_by","poi") in data.edge_types and data["complex","near_by","poi"].edge_index.numel()>0

        if ("complex" not in data.node_types) or ("poi" not in data.node_types) or (not has_rel):
            out["complex"].x = x_c
            out["poi"].x     = x_p
            return out

        metadata = (list(data.node_types), list(data.edge_types))
        self._ensure_convs(metadata)

        # edge dict을 장치에 맞게 이동
        edge_index_dict = {}
        for rel, ei in data.edge_index_dict.items():
            edge_index_dict[rel] = ei.to(dev, non_blocking=True)

        x = {"complex": x_c, "poi": x_p}
        for conv, norm in zip(self._convs, self._norms):
            x = conv(x, edge_index_dict)
            x = {k: norm(torch.relu(v)) for k, v in x.items()}

        out["complex"].x = x["complex"]
        out["poi"].x     = x["poi"]
        return out

#config.py
import os
import torch

# ========= Paths =========
OUTDIR = os.getenv("APT_OUTDIR", r"D:\KAIST\Thesis_Data")
CACHE_DIR = os.path.join(OUTDIR, "cache")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ========= External Keys =========
KAKAO_REST_API_KEY  = os.getenv("KAKAO_REST_API_KEY",  "af8e92d2b69c862990818f7e3fe46747")
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID",     "zomg3g7bbu")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "XHMX9y87BVSem7AIweqEdYSz9H4qmbkO698gRiIN")

# ========= Raw data defaults =========
DEFAULT_CSVS = [
    r"D:\KAIST\Data\Thesis\AptTradeReportDetail.csv",
    r"D:\KAIST\Data\Thesis\AptTradeReportDetailGG.csv",
]
ADMIN_XLSX = r"D:\KAIST\투신2024\행정구역코드 정리.xlsx"

# ========= Runtime / GPU =========
# 우선순위: 환경변수 DEVICE > CUDA 가용성
_device_env = os.getenv("DEVICE", "").strip().lower()
if _device_env in {"cuda", "gpu"} and torch.cuda.is_available():
    DEVICE = "cuda"
elif _device_env == "cpu":
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_THREADS = int(os.getenv("NUM_THREADS", "1"))
torch.set_num_threads(max(1, NUM_THREADS))

# matmul 정밀도 및 cudnn 설정
try:
    torch.set_float32_matmul_precision(os.getenv("MATMUL_PREC", "high"))
except Exception:
    pass

import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = os.getenv("CUDNN_BENCHMARK", "1") == "1"
cudnn.deterministic = os.getenv("CUDNN_DETERMINISTIC", "0") == "1"

# AMP / compile
MIXED_PRECISION = os.getenv("AMP", "1") == "1"
AMP_DTYPE = os.getenv("AMP_DTYPE", "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16")
USE_TORCH_COMPILE = os.getenv("TORCH_COMPILE", "0") == "1"
TORCH_COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", "default")  # 'default' | 'reduce-overhead' | 'max-autotune'

SEED = int(os.getenv("SEED", "42"))
USE_CACHE = os.getenv("USE_CACHE", "1") == "1"

# ========= Retrieval / Graph =========
SEARCH_RADIUS_M    = int(os.getenv("SEARCH_RADIUS_M", "1000"))
NEIGHBOR_TOPK      = int(os.getenv("NEIGHBOR_TOPK", "10"))
GEO_NEAREST_N      = int(os.getenv("GEO_NEAREST_N", "1"))
GEO_MIX_BETA       = float(os.getenv("GEO_MIX_BETA", "0.3"))
USE_FAISS          = os.getenv("USE_FAISS", "1") == "1"
LOCAL_RADIUS_M     = int(os.getenv("LOCAL_RADIUS_M", "1500"))
SCALE_ALIGN_WINDOW = int(os.getenv("SCALE_ALIGN_WINDOW", "6"))

# ========= Temporal =========
T_IN        = int(os.getenv("T_IN", "12"))
H_OUT       = int(os.getenv("H_OUT", "6"))
TEMP_MODEL  = os.getenv("TEMP_MODEL", "transformer")

# ========= Diffusion (옵션 보일러플레이트) =========
USE_DIFFUSION = os.getenv("USE_DIFFUSION", "1") == "1"
DIFF_STEPS    = int(os.getenv("DIFF_STEPS", "100"))
DIFF_EPOCHS   = int(os.getenv("DIFF_EPOCHS", "50"))
DIFF_LR       = float(os.getenv("DIFF_LR", "2e-4"))
TS_LENGTH     = int(os.getenv("TS_LENGTH", "24"))

# ========= HGNN 카테고리 =========
HGNN_FAC_NAMES = [
    "유치원","초등학교","중학교","고등학교","대학교",
    "학원_국영수보습","학원_예체능",
    "지하철역","버스정류장","고속도로IC","주차장",
    "관광명소","공공기관","부동산중개소",
    "마트","편의점","종합상급병원","약국","문화시설","은행","식당","카페","주유소",
]

SUPER_CAT_MAP = {
    "유치원":"교육","초등학교":"교육","중학교":"교육","고등학교":"교육","대학교":"교육",
    "학원_국영수보습":"교육","학원_예체능":"교육",
    "지하철역":"교통","버스정류장":"교통","고속도로IC":"교통","주차장":"교통",
    "관광명소":"편의시설","공공기관":"편의시설","부동산중개소":"편의시설",
    "마트":"편의시설","편의점":"편의시설","종합상급병원":"편의시설","약국":"편의시설",
    "문화시설":"편의시설","은행":"편의시설","식당":"편의시설","카페":"편의시설","주유소":"편의시설",
}

# ========= Cohort thresholds =========
WARM_MAX_PRE = int(os.getenv("WARM_MAX_PRE", "10"))
HOT_MIN_PRE  = int(os.getenv("HOT_MIN_PRE",  "12"))

# ========= Logging =========
USE_WANDB     = os.getenv("USE_WANDB", "0") == "1"
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "GEN-ST-RAP")
WANDB_ENTITY  = os.getenv("WANDB_ENTITY", None)

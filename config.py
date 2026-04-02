PROJECT_ROOT  = "/Data1/qwh/Front120-VideoScreening"
MODEL_PATH    = "/Data1/qwh/Qwen3.5-35B-A3B"
DATA_BASE_DIR = "/desay_h3c/CX_Collection"
OUTPUT_DIR    = f"{PROJECT_ROOT}/outputs"

VLLM_URLS = [
    "http://127.0.0.1:8000/v1/chat/completions",
    "http://127.0.0.1:8001/v1/chat/completions",
    "http://127.0.0.1:8002/v1/chat/completions",
    "http://127.0.0.1:8003/v1/chat/completions",
]

CONCURRENCY_PER_INST = 16
SAMPLE_INTERVAL_SEC  = 2.0
EARLY_STOP           = True

CONF_HIGH = 80
CONF_LOW  = 30

LOG_LEVEL = "INFO"

# ── 任务类型，由命令行参数覆盖 ──
# "gate"  → 仅检测道闸
# "truck" → 仅检测重型货车
TASK_TYPE: str = "gate"

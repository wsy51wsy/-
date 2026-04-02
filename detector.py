import base64
import copy
import itertools
import json
import re

import aiohttp

import config

# ════════════════════════════════════════════════
#  道闸  prompt + schema
# ════════════════════════════════════════════════
_SYSTEM_PROMPT_GATE = """\
你是交通场景视觉检测系统，分析道路监控画面，判断是否存在道闸，仅输出指定JSON。

【道闸】
安装于机动车出入口的自动管控设备，通过栏杆升降实现车辆进出权限管理。
关键特征：① 可升降的闸杆/栏杆；② 立柱式机箱；③ 安装于车道出入口。
排除：道路护栏、固定栏杆、围栏、围墙、普通门、行人/地铁闸机。

【判定规则】
- 目标主体完整、清晰可辨、符合所有特征时判"是"；模糊、遮挡、局部可见一律判"否"
- 给出0-100整数置信度，表示对该目标判断的确信程度

【输出】
仅输出一行合法JSON，无任何其他内容：
{"道闸":"是/否","道闸置信度":整数}"""

_SCHEMA_GATE = {
    "type": "object",
    "properties": {
        "道闸":       {"type": "string",  "enum": ["是", "否"]},
        "道闸置信度":  {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["道闸", "道闸置信度"],
}

# ════════════════════════════════════════════════
#  重型货车  prompt + schema
# ════════════════════════════════════════════════
_SYSTEM_PROMPT_TRUCK = """\
你是交通场景视觉检测系统，分析道路监控画面，判断是否存在重型货车，仅输出指定JSON。

【重型货车】
总质量大、载重性能强的货运车辆，用于长途物流、大宗物资、集装箱、工程物料运输。
关键特征：① 车身体积远超普通乘用车；② 多轴（通常≥3轴）；③ 车头高大。
排除：轻型货车（蓝牌）、皮卡、面包车、轿车、SUV、公交车、客车、中型货车。

【判定规则】
- 目标主体完整、清晰可辨、符合所有特征时判"是"；模糊、遮挡、局部可见一律判"否"
- 给出0-100整数置信度，表示对该目标判断的确信程度

【输出】
仅输出一行合法JSON，无任何其他内容：
{"重型货车":"是/否","重型货车置信度":整数}"""

_SCHEMA_TRUCK = {
    "type": "object",
    "properties": {
        "重型货车":      {"type": "string",  "enum": ["是", "否"]},
        "重型货车置信度": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["重型货车", "重型货车置信度"],
}

# ════════════════════════════════════════════════
#  消息模板（共用）
# ════════════════════════════════════════════════
_MSG_TEMPLATE = [
    {"role": "system", "content": ""},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": ""}},
            {"type": "text", "text": "请分析这张图片。"},
        ],
    },
]

_url_cycle = itertools.cycle(config.VLLM_URLS)


def _next_url() -> str:
    return next(_url_cycle)


def _build_payload(jpeg_bytes: bytes, task_type: str) -> dict:
    if task_type == "gate":
        system_prompt = _SYSTEM_PROMPT_GATE
        schema = _SCHEMA_GATE
    elif task_type == "truck":
        system_prompt = _SYSTEM_PROMPT_TRUCK
        schema = _SCHEMA_TRUCK
    else:
        raise ValueError(f"未知 task_type: {task_type!r}，仅支持 'gate' 或 'truck'")

    b64 = base64.b64encode(jpeg_bytes).decode()
    msgs = copy.deepcopy(_MSG_TEMPLATE)
    msgs[0]["content"] = system_prompt
    msgs[1]["content"][0]["image_url"]["url"] = f"data:image/jpeg;base64,{b64}"
    return {
        "model": config.MODEL_PATH,
        "messages": msgs,
        "max_tokens": 2048,
        "temperature": 0,
        "extra_body": {
            "guided_json": schema,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def _extract_json(content: str) -> dict:
    """从模型输出中提取 JSON，兼容思考链（<think>...</think>）前缀。"""
    matches = re.findall(r'\{[^{}]+\}', content, re.DOTALL)
    if matches:
        return json.loads(matches[-1])
    raise ValueError(f"模型输出中未找到 JSON：{content!r}")


async def infer(session: aiohttp.ClientSession,
                jpeg_bytes: bytes,
                task_type: str) -> tuple[dict, str]:
    """返回 (解析后的结构化结果, 模型原始回答字符串)"""
    payload = _build_payload(jpeg_bytes, task_type)
    async with session.post(_next_url(), json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    content = data["choices"][0]["message"]["content"]
    return _extract_json(content), content

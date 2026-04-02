import os

import config


def pack_video_path(pack: str, filename: str) -> str:
    parts = pack.split("_")
    pack_path = os.path.join(config.DATA_BASE_DIR, f"{parts[-2]}_{parts[-1]}", pack)
    for sub in sorted(os.listdir(pack_path)):
        if sub.startswith("CX"):
            candidate = os.path.join(pack_path, sub, "Front120_enc", filename)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(f"找不到视频: {pack}/{filename}")


def scan_videos(pack_names: list[str]) -> list[tuple[str, str]]:
    tasks = []
    for pack in pack_names:
        parts = pack.split("_")
        pack_path = os.path.join(config.DATA_BASE_DIR, f"{parts[-2]}_{parts[-1]}", pack)
        if not os.path.isdir(pack_path):
            continue
        for sub in sorted(os.listdir(pack_path)):
            if not sub.startswith("CX"):
                continue
            front_dir = os.path.join(pack_path, sub, "Front120_enc")
            if os.path.isdir(front_dir):
                for f in sorted(f for f in os.listdir(front_dir) if f.endswith(".mp4")):
                    tasks.append((pack, os.path.join(front_dir, f)))
    return tasks

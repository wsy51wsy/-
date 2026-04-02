import asyncio
import dataclasses
import json
import logging
import os
import pathlib
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import cv2

import config
import detector
from utils import scan_videos

PACK_NAMES = ["20260304_215717_CX_WL397","20260304_225627_CX_WL268","20260305_213918_CX_WL397","20260306_121934_CX WL165","20260306_181747_CX_WL232","20260306_193443_CX_WL463","20260306_232312_CX_WL170","20260306_214115_CX_WL397","20260306_231555_CX_WL397","20260307_011022_CX_WL170","20260307_004936_CX_WL397","20260307_025411_CX_WL323","20260307_081643_CX_WL413","20260307_213154_CX_WL268","20260307_214117_CX_WL397","20260307_230935_CX_WL397","20260308_025438_CX_WL268","20260308_024922_CX_WL323","20260308_055043_CX_WL323","20260308_082140_CX_WL413","20260308_214313_CX_WL397","20260308_220737_CX_WL352","20260308_231044_CX_WL397","20260309_004557_CX_WL397","20260309_055837_CX_WL323","20260309_063625_CX_WL268","20260309_100326_CX_WL261","20260309_113001_CX_WL165","20260309_130535_CX_WL165","20260309_093925_CX_WL413","20260309_141650_CX_WL413","20260309_150813_CX_WL170","20260309_173301_CX_WL170","20260309_214622_CX_WL261","20260309_231602_CX_WL261","20260310_004426_CX_WL261","20260310_020326_CX_WL268","20260310_024328_CX_WL303","20260310_090809_CX_WL261","20260310_093238_CX_WL165","20260310_102347_CX_WL165","20260310_093250_CX_WL170","20260310_114839_CX_WL165","20260310_124631_CX_WL165","20260310_142943_CX_WL303","20260310_155446_CX_WL463","20260310_185115_CX_WL463","20260311_004617_CX_WL261","20260311_024637_CX_WL303","20260311_124649_CX_WL463","20260311_153932_CX_WL165","20260311_152208_CX_WL323","20260311_174827_CX_WL165","20260311_185133_CX_WL165","20260311_105145_CX_WL397","20260311_120413_CX_WL397","20260311_214452_CX_WL261","20260312_024441_CX_WL303","20260312_064707_CX_WL413","20260312_092848_CX_WL463","20260312_093917_CX_WL261","20260312_103806_CX_WL323","20260312_112220_CX_WL463","20260312_125143_CX_WL463","20260312_144450_CX_WL303","20260312_153502_CX_WL165","20260312_172510_CX_WL165","20260312 182812_CX_WL165","20260312_222040_CX_WL170","20260312_214420_CX_WL261","20260313_024358_CX_WL303","20260313_092803_CX_WL463","20260313_124820_CX_WL463","20260313_142425_CX_WL303","20260313_152223_CX_WL165","20260313_181818_CX_WL165","20260313_214515_CX_WL261","20260314_022904_CX_WL303","20260314_091948_CX_WL261","20260314_111751_CX_WL261","20260314_213511_CX_WL261","20260315_022951_CX_WL303","20260315_091825_CX_WL261","20260315_112319_CX_WL261","20260315_092515_CX_WL165","20260315_092542_CX_WL413","20260315_112246_CX_WL413","20260315_124754_CX_WL165","20260315_145004_CX_WL248","20260315_144652_CX_WL303","20260315_153611_CX_WL463","20260315_180103_CX_WL248","20260315_185036_CX_WL463","20260315_213715_CX_WL261","20260314_114442_CX_WL397","20260316_022716_CX_WL303","20260316_093108_CX_WL165","20260316_093405_CX_WL261","20260316_130422_CX_WL165","20260316_144557_CX_WL303","20260316_154306_CX_WL232","20260316_155306_CX_WL463","20260316_165907_CX_WL170","20260316_174427_CX_WL463","20260316_175232_CX_WL232","20260316_200834_CX_WL463","20260316_213019_CX_WL261","20260316_143904_CX_WL323","20260317_022947_CX_WL303","20260317_092220_CX_WL165","20260317_091857_CX_WL261","20260317_112837_CX_WL165","20260317_123258_CX_WL165","20260317_145726_CX_WL463","20260317_150932_CX_WL303","20260317_151014_CX_WL463","20260317_180213_CX_WL463","20260317_181319_CX_WL303","20260317_183349_CX_WL248","20260317_144603_CX_WL323","20260317_213039_CX_WL261","20260318_022102_CX_WL303","20260318_093132_CX_WL165","20260318_092404_CX_WL261","20260318_112438_CX_WL413","20260318_124807_CX_WL413","20260318_150420_CX_WL303","20260318_160318_CX_WL170","20260318_181455_CX_WL463","20260318_212903_CX_WL261","20260319_003003_CX_WL426","20260319_023455_CX_WL303","20260319_150752_CX_WL303","20260319_213734_CX_WL261","20260320_022852_CX_WL303","20260320_092734_CX_WL261","20260320_150200_CX_WL303","20260320_160203_CX_WL170","20260320_212638_CX_WL261","20260321_021210_CX_WL303","20260321_092945_CX_WL261","20260321_150858_CX_WL303","20260321_212544_CX_WL261","20260321_213631_CX_WL529","20260322_022023_CX_WL303","20260322_023544_CX_WL170","20260322_034337_CX_WL170","20260322_055510_CX_WL170","20260322_092846_CX_WL261","20260322_150002_CX_WL303","20260322_214007_CX_WL529","20260323_023920_CX_WL170","20260323_211744_CX_WL529","20260323_212352_CX_WL261","20260324_014325_CX_WL170","20260324_150916_CX_WL170","20260324_210622_CX_WL529","20260325_023819_CX_WL170","20260325_152901_CX_WL170","20260325_150546_CX_WL165","20260325_150543_CX_WL426","20260325_211035_CX_WL529","20260326_013620_CX_WL170","20260326_213006_CX_WL529","20260326_214923_CX_WL323","20260327_023021_CX_WL170","20260327_034654_CX_WL397","20260327_210737_CX_WL529","20260328_014734_CX_WL170","20260328_020741_CX_WL397","20260328_020755_CX_WL306","20260328_211431_CX_WL529","20260329_041506_CX_WL397","20260329_211418_CX _WL529","20260330_013724_CX_WL170","20260330_020437_CX_WL323","20260331_014531_CX_WL323"]



@dataclasses.dataclass
class InferContext:
    session: aiohttp.ClientSession
    sem: asyncio.Semaphore
    executor: ThreadPoolExecutor


def setup_logger() -> logging.Logger:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger = logging.getLogger("main")
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(config.OUTPUT_DIR, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _sample_frames(video_path: str) -> list[tuple[int, float, bytes]]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(fps * config.SAMPLE_INTERVAL_SEC))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                frames.append((idx, round(idx / fps, 2), buf.tobytes()))
        idx += 1
    cap.release()
    return frames


def _is_hit(result: dict) -> bool:
    return result["道闸"] == "是" or result["重型货车"] == "是"


def _save_hit_frame(pack: str, video_path: str, frame_idx: int, time_sec: float,
                    jpeg: bytes, raw_content: str, result: dict) -> None:
    """将命中帧的图片和模型完整回答（含 think）保存到磁盘。"""
    video_stem = pathlib.Path(video_path).stem
    save_dir = pathlib.Path(config.OUTPUT_DIR) / "hit_frames" / pack / video_stem
    save_dir.mkdir(parents=True, exist_ok=True)
    base = f"frame_{frame_idx}_{time_sec}s"
    (save_dir / f"{base}.jpg").write_bytes(jpeg)
    text = (
        f"video : {video_path}\n"
        f"frame : {frame_idx}  time: {time_sec}s\n"
        f"result: {json.dumps(result, ensure_ascii=False)}\n"
        f"{'='*60}\n"
        f"{raw_content}\n"
    )
    (save_dir / f"{base}.txt").write_text(text, encoding="utf-8")


async def process_video(ctx: InferContext, video_path: str, logger: logging.Logger,
                        pack: str = "") -> list[dict]:
    loop = asyncio.get_running_loop()
    frames = await loop.run_in_executor(ctx.executor, _sample_frames, video_path)

    hits = []
    for i, (frame_idx, time_sec, jpeg) in enumerate(frames):
        async with ctx.sem:
            result, raw_content = await detector.infer(ctx.session, jpeg)

        result["frame_idx"] = frame_idx
        result["time_sec"] = time_sec

        logger.debug(f"  frame {frame_idx} ({time_sec}s): 道闸={result['道闸']}({result['道闸置信度']}) 货车={result['重型货车']}({result['重型货车置信度']})")

        if _is_hit(result):
            hits.append(result)
            _save_hit_frame(pack, video_path, frame_idx, time_sec, jpeg, raw_content, result)
            if config.EARLY_STOP:
                conf = max(result["道闸置信度"] if result["道闸"] == "是" else 0,
                           result["重型货车置信度"] if result["重型货车"] == "是" else 0)
                if conf >= config.CONF_HIGH:
                    break
                elif conf >= config.CONF_LOW:
                    for fi, ts, jp in frames[i + 1 : i + 3]:
                        async with ctx.sem:
                            r2, rc2 = await detector.infer(ctx.session, jp)
                        r2["frame_idx"] = fi
                        r2["time_sec"] = ts
                        if _is_hit(r2):
                            hits.append(r2)
                            _save_hit_frame(pack, video_path, fi, ts, jp, rc2, r2)
                    break

    return hits


async def run(tasks: list[tuple[str, str]], logger: logging.Logger) -> list[dict]:
    sem = asyncio.Semaphore(len(config.VLLM_URLS) * config.CONCURRENCY_PER_INST)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        with ThreadPoolExecutor(max_workers=min(8, len(tasks) or 1)) as executor:
            ctx = InferContext(session=session, sem=sem, executor=executor)

            async def _wrap(pack: str, fname: str, path: str) -> dict:
                try:
                    hits = await process_video(ctx, path, logger, pack=pack)
                    return {"pack": pack, "filename": fname, "hits": hits}
                except Exception:
                    logger.exception(f"{pack}/{fname} failed")
                    raise


            futs = [_wrap(pack, os.path.basename(path), path) for pack, path in tasks]
            results = []
            for i, fut in enumerate(asyncio.as_completed(futs), 1):
                try:
                    item = await fut
                    results.append(item)
                    status = f"HIT({len(item['hits'])})" if item["hits"] else "miss"
                    logger.info(f"[{i}/{len(tasks)}] {item['pack']}/{item['filename']} -> {status}")
                except Exception:
                    logger.exception(f"[{i}/{len(tasks)}] task failed")
    return results



def build_output(results: list[dict]) -> tuple[dict, dict]:
    simple: dict[str, list] = defaultdict(list)
    detail: dict[str, dict] = defaultdict(dict)
    for item in results:
        if item["hits"]:
            simple[item["pack"]].append(item["filename"])
            detail[item["pack"]][item["filename"]] = {"detections": item["hits"]}
    for pack in simple:
        simple[pack].sort()
    return dict(simple), dict(detail)


def main():
    logger = setup_logger()
    tasks = scan_videos(PACK_NAMES)
    logger.info(f"Total videos: {len(tasks)}")

    t0 = time.time()
    results = asyncio.run(run(tasks, logger))
    elapsed = time.time() - t0

    simple, detail = build_output(results)
    for pack in PACK_NAMES:
        simple.setdefault(pack, [])

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(simple, f, ensure_ascii=False, indent=2)
    with open(os.path.join(config.OUTPUT_DIR, "results_detail.json"), "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    total_hits = sum(len(v) for v in simple.values())
    logger.info(f"Done in {elapsed:.1f}s, hits={total_hits}, saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Run external MCITrack-L384 for MyLabel in an isolated Python process.

Only this adapter is bundled with MyLabel. The official MCITrack repository,
checkpoint and Python environment are selected by the user and remain external.
One UTF-8 JSON object is written per line for the Qt front-end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def emit(kind: str, **payload: Any) -> None:
    print(json.dumps({"type": kind, **payload}, ensure_ascii=False), flush=True)


def load_request(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class FrameProvider:
    def __init__(self, request: dict[str, Any]) -> None:
        import cv2

        self.cv2 = cv2
        self.source_type = str(request.get("source_type", "images"))
        self.total_frames = int(request.get("total_frames", 0))
        self.paths = [Path(item) for item in request.get("frame_paths", [])]
        self.capture = None
        self.cache_index = -1
        self.cache_image = None
        if self.source_type == "images":
            if len(self.paths) != self.total_frames:
                raise ValueError("图片数量与项目总帧数不一致。")
        elif self.source_type == "video":
            source_path = Path(request["source_path"])
            if not source_path.is_file():
                raise FileNotFoundError(f"视频文件不存在：{source_path}")
            self.capture = cv2.VideoCapture(str(source_path))
            if not self.capture.isOpened():
                raise RuntimeError(f"算法环境无法打开视频：{source_path}")
        else:
            raise ValueError(f"不支持的数据源类型：{self.source_type}")

    def get_rgb(self, index: int) -> Any:
        import numpy as np

        if index == self.cache_index and self.cache_image is not None:
            return self.cache_image.copy()
        if not 0 <= index < self.total_frames:
            raise IndexError(f"帧序号超出范围：{index + 1}")
        if self.source_type == "images":
            try:
                raw = np.fromfile(str(self.paths[index]), dtype=np.uint8)
                bgr = self.cv2.imdecode(raw, self.cv2.IMREAD_COLOR)
            except (OSError, ValueError):
                bgr = None
            if bgr is None:
                raise RuntimeError(f"算法环境无法读取图片：{self.paths[index]}")
        else:
            assert self.capture is not None
            if index != self.cache_index + 1:
                self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = self.capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f"算法环境无法读取第 {index + 1} 帧。")
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        self.cache_index = index
        self.cache_image = rgb.copy()
        return rgb

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def configure_official_cuda_calls(torch: Any, device: Any) -> None:
    """Route the official tracker's hard-coded .cuda() calls to our device."""

    def tensor_cuda(tensor: Any, *_args: Any, **_kwargs: Any) -> Any:
        return tensor.to(device)

    def module_cuda(module: Any, *_args: Any, **_kwargs: Any) -> Any:
        return module.to(device)

    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.cuda = module_cuda


def validate_l384_checkpoint(torch: Any, checkpoint: Any) -> dict[str, Any]:
    state = checkpoint.get("net") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("模型权重缺少 net 参数，无法作为官方 MCITrack 检查点加载。")
    position = state.get("encoder.body.pos_embed")
    neck = state.get("neck.layers.0.mixer.in_proj.weight")
    if position is None or neck is None:
        raise ValueError("模型权重缺少 MCITrack-L384 的关键参数。")
    if tuple(position.shape) != (1, 720, 768) or tuple(neck.shape) != (3072, 768):
        raise ValueError(
            "所选权重不是当前代码所需的 MCITrack-L384（384输入）权重；"
            f"检测到 pos_embed={tuple(position.shape)}、neck={tuple(neck.shape)}。"
        )
    for name, value in state.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"模型权重包含无效数值：{name}")
    return state


def normalize_box(box: Any, width: int, height: int) -> list[int]:
    import numpy as np

    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise RuntimeError("MCITrack 输出了无效目标框。")
    x, y, w, h = (float(item) for item in values)
    x = min(max(x, 0.0), max(0.0, width - 1.0))
    y = min(max(y, 0.0), max(0.0, height - 1.0))
    w = min(max(w, 1.0), max(1.0, width - x))
    h = min(max(h, 1.0), max(1.0, height - y))
    return [round(x), round(y), max(1, round(w)), max(1, round(h))]


def run_mcitrack(request: dict[str, Any], device_name: str) -> None:
    code_path = Path(request["code_path"]).resolve()
    weight_path = Path(request["model_path"]).resolve()
    if request.get("algorithm") != "mcitrack_l384":
        raise ValueError(f"不支持的跟踪算法：{request.get('algorithm')}")
    if not code_path.is_dir():
        raise FileNotFoundError(f"MCITrack代码目录不存在：{code_path}")
    if not (code_path / "lib" / "models" / "mcitrack").is_dir():
        raise ValueError(f"所选目录不是有效的MCITrack官方代码目录：{code_path}")
    config_path = code_path / "experiments" / "mcitrack" / "mcitrack_l384.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少MCITrack-L384配置文件：{config_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"模型权重不存在：{weight_path}")

    sys.path.insert(0, str(code_path))
    os.chdir(code_path)
    import torch

    device = torch.device(device_name)
    configure_official_cuda_calls(torch, device)

    from lib.config.mcitrack.config import cfg, update_config_from_file
    update_config_from_file(str(config_path))
    if int(cfg.TEST.SEARCH_SIZE) != 384 or str(cfg.MODEL.ENCODER.TYPE).lower() != "fastitpnl":
        raise RuntimeError("mcitrack_l384.yaml配置与官方L384结构不一致。")

    # Avoid loading the training-time Fast-iTPN pretrain file. The selected
    # tracking checkpoint already contains the complete encoder parameters.
    import lib.models.mcitrack.encoder as encoder_module
    encoder_module.is_main_process = lambda: False

    try:
        checkpoint = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(weight_path), map_location="cpu")
    state = validate_l384_checkpoint(torch, checkpoint)

    from lib.models.mcitrack import build_mcitrack
    network = build_mcitrack(cfg)
    network.load_state_dict(state, strict=True)
    network.to(device).eval()

    from lib.test.tracker.mcitrack import MCITRACK
    params = SimpleNamespace(
        cfg=cfg,
        checkpoint=str(weight_path),
        template_factor=cfg.TEST.TEMPLATE_FACTOR,
        template_size=cfg.TEST.TEMPLATE_SIZE,
        search_factor=cfg.TEST.SEARCH_FACTOR,
        search_size=cfg.TEST.SEARCH_SIZE,
        yaml_name="mcitrack_l384",
        debug=0,
        save_all_boxes=False,
    )
    # Reuse the official track and update implementation, while supplying the
    # already validated network so the 1.4 GB checkpoint is not loaded twice.
    tracker = MCITRACK.__new__(MCITRACK)
    from lib.test.tracker.basetracker import BaseTracker
    BaseTracker.__init__(tracker, params)
    tracker.cfg = cfg
    tracker.network = network
    from lib.test.tracker.utils import Preprocessor
    tracker.preprocessor = Preprocessor()
    tracker.state = None
    tracker.fx_sz = cfg.TEST.SEARCH_SIZE // cfg.MODEL.ENCODER.STRIDE
    from lib.test.utils.hann import hann2d
    tracker.output_window = hann2d(
        torch.tensor([tracker.fx_sz, tracker.fx_sz], dtype=torch.long), centered=True
    ).to(device)
    tracker.num_template = cfg.TEST.NUM_TEMPLATES
    tracker.debug = 0
    tracker.frame_id = 0
    tracker.h_state = [None] * cfg.MODEL.NECK.N_LAYERS
    # Generic user videos are closest to the official long-video TRACKINGNET
    # evaluation preset. DEFAULT=1 would discard nearly every hidden-state
    # update and effectively disable MCITrack's contextual propagation.
    tracker.update_threshold = cfg.TEST.UPT.TRACKINGNET
    tracker.update_h_t = cfg.TEST.UPH.TRACKINGNET
    tracker.update_intervals = cfg.TEST.INTER.TRACKINGNET
    tracker.memory_bank = cfg.TEST.MB.TRACKINGNET

    total_frames = int(request.get("total_frames", 0))
    start = int(request["start_index"])
    if total_frames <= 0 or not 0 <= start < total_frames:
        raise ValueError("跟踪帧范围无效。")
    init_box = [float(value) for value in request["box"]]
    if len(init_box) != 4 or init_box[2] <= 0 or init_box[3] <= 0:
        raise ValueError("初始目标框的宽和高必须大于0。")

    frames = FrameProvider(request)
    try:
        initial_image = frames.get_rgb(start)
        tracker.initialize(initial_image, {"init_bbox": init_box, "seq_name": "MyLabel"})
        emit("ready", device=device_name, config="mcitrack_l384")
        for index in range(start + 1, total_frames):
            image = frames.get_rgb(index)
            with torch.inference_mode():
                result = tracker.track(image)
            box = normalize_box(result["target_bbox"], image.shape[1], image.shape[0])
            emit("box", index=index, box=box)
    finally:
        frames.close()
        del tracker, network, checkpoint, state
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args()
    request = load_request(Path(args.request))
    emit("environment", python=sys.executable, prefix=sys.prefix)
    preferred = "cuda:0" if request.get("prefer_gpu", True) else "cpu"
    try:
        import torch
        if preferred.startswith("cuda") and not torch.cuda.is_available():
            emit("fallback", message="当前算法环境未启用CUDA，正在使用CPU运行MCITrack。")
            preferred = "cpu"
    except Exception:
        pass
    try:
        run_mcitrack(request, preferred)
    except Exception as first_error:
        if not preferred.startswith("cuda"):
            emit("error", message=str(first_error) or first_error.__class__.__name__, details=traceback.format_exc())
            return 1
        emit("fallback", message=f"GPU启动失败，正在改用CPU：{first_error}")
        try:
            run_mcitrack(request, "cpu")
        except Exception as second_error:
            emit("error", message=str(second_error) or second_error.__class__.__name__, details=traceback.format_exc())
            return 1
    emit("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

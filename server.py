"""HTTP inference server for MyLabel remote automatic annotation.

Run from the copied project directory on Linux:
    python server.py --config model.json

The model registry in model.json is intentionally external to the client EXE.
Add future algorithms by registering another factory in ALGORITHM_FACTORIES and
adding its paths/options to the models object in model.json.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import secrets
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取服务配置 {path}：{exc}") from exc
    if not isinstance(config.get("server"), dict) or not isinstance(config.get("models"), dict):
        raise ValueError("model.json 必须包含 server 和 models 对象。")
    return config


def resolve_path(config_file: Path, value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = config_file.parent / path
    return path.resolve()


def decode_image(value: Any, max_bytes: int) -> np.ndarray:
    if not isinstance(value, str) or not value:
        raise ValueError("请求缺少 image_base64。")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("image_base64 不是有效的Base64图像。") from exc
    if not raw or len(raw) > max_bytes:
        raise ValueError(f"图像大小必须在1到{max_bytes}字节之间。")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码客户端图像。")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_box(box: Any, width: int, height: int) -> list[int]:
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise RuntimeError("跟踪器输出了无效目标框。")
    x, y, w, h = (float(item) for item in values)
    x = min(max(x, 0.0), max(0.0, width - 1.0))
    y = min(max(y, 0.0), max(0.0, height - 1.0))
    w = min(max(w, 1.0), max(1.0, width - x))
    h = min(max(h, 1.0), max(1.0, height - y))
    return [round(x), round(y), max(1, round(w)), max(1, round(h))]


class MCITrackL384Model:
    """One cached L384 network; each client session gets independent track state."""

    def __init__(self, name: str, options: dict[str, Any], config_file: Path) -> None:
        self.name = name
        self.options = options
        self.code_path = resolve_path(config_file, str(options["code_path"]))
        self.model_path = resolve_path(config_file, str(options["model_path"]))
        self.device_name = str(options.get("device", "auto"))
        self.loaded = False
        self.load_lock = threading.Lock()
        self.inference_lock = threading.RLock()
        self.network = None
        self.cfg = None
        self.device = None
        self.tracker_class = None
        self.base_tracker_class = None
        self.preprocessor_class = None
        self.hann2d = None

    def load(self) -> None:
        if self.loaded:
            return
        with self.load_lock:
            if self.loaded:
                return
            if not (self.code_path / "lib" / "models" / "mcitrack").is_dir():
                raise FileNotFoundError(f"MCITrack代码目录无效：{self.code_path}")
            if not self.model_path.is_file():
                raise FileNotFoundError(f"MCITrack权重不存在：{self.model_path}")
            config_name = str(self.options.get("config", "mcitrack_l384"))
            yaml_path = self.code_path / "experiments" / "mcitrack" / f"{config_name}.yaml"
            if not yaml_path.is_file():
                raise FileNotFoundError(f"MCITrack配置不存在：{yaml_path}")
            if str(self.code_path) not in sys.path:
                sys.path.insert(0, str(self.code_path))

            import torch
            if self.device_name == "auto":
                self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
            if self.device_name.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("model.json要求使用CUDA，但当前PyTorch无法使用CUDA。")
            self.device = torch.device(self.device_name)

            # Official MCITrack uses .cuda() in preprocessing/head creation.
            # Route those calls to the configured device, including CPU fallback.
            device = self.device
            torch.Tensor.cuda = lambda tensor, *_a, **_kw: tensor.to(device)
            torch.nn.Module.cuda = lambda module, *_a, **_kw: module.to(device)

            from lib.config.mcitrack.config import cfg, update_config_from_file
            update_config_from_file(str(yaml_path))
            if int(cfg.TEST.SEARCH_SIZE) != 384 or str(cfg.MODEL.ENCODER.TYPE).lower() != "fastitpnl":
                raise RuntimeError("服务端当前仅适配官方MCITrack-L384结构。")
            import lib.models.mcitrack.encoder as encoder_module
            encoder_module.is_main_process = lambda: False
            try:
                checkpoint = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(str(self.model_path), map_location="cpu")
            state = checkpoint.get("net") if isinstance(checkpoint, dict) else None
            if not isinstance(state, dict):
                raise ValueError("权重缺少net参数。")
            pos = state.get("encoder.body.pos_embed")
            neck = state.get("neck.layers.0.mixer.in_proj.weight")
            if pos is None or neck is None or tuple(pos.shape) != (1, 720, 768) or tuple(neck.shape) != (3072, 768):
                raise ValueError("权重不是与当前服务配置匹配的MCITrack-L384权重。")
            from lib.models.mcitrack import build_mcitrack
            network = build_mcitrack(cfg)
            network.load_state_dict(state, strict=True)
            network.to(self.device).eval()

            from lib.test.tracker.mcitrack import MCITRACK
            from lib.test.tracker.basetracker import BaseTracker
            from lib.test.tracker.utils import Preprocessor
            from lib.test.utils.hann import hann2d
            self.cfg = cfg
            self.network = network
            self.tracker_class = MCITRACK
            self.base_tracker_class = BaseTracker
            self.preprocessor_class = Preprocessor
            self.hann2d = hann2d
            self.loaded = True
            del checkpoint, state

    def new_tracker(self, image: np.ndarray, box: list[float]) -> Any:
        self.load()
        import torch
        cfg = self.cfg
        params = SimpleNamespace(
            cfg=cfg,
            checkpoint=str(self.model_path),
            template_factor=cfg.TEST.TEMPLATE_FACTOR,
            template_size=cfg.TEST.TEMPLATE_SIZE,
            search_factor=cfg.TEST.SEARCH_FACTOR,
            search_size=cfg.TEST.SEARCH_SIZE,
            yaml_name="mcitrack_l384",
            debug=0,
            save_all_boxes=False,
        )
        tracker = self.tracker_class.__new__(self.tracker_class)
        self.base_tracker_class.__init__(tracker, params)
        tracker.cfg = cfg
        tracker.network = self.network
        tracker.preprocessor = self.preprocessor_class()
        tracker.state = None
        tracker.fx_sz = cfg.TEST.SEARCH_SIZE // cfg.MODEL.ENCODER.STRIDE
        tracker.output_window = self.hann2d(
            torch.tensor([tracker.fx_sz, tracker.fx_sz], dtype=torch.long), centered=True
        ).to(self.device)
        tracker.num_template = cfg.TEST.NUM_TEMPLATES
        tracker.debug = 0
        tracker.frame_id = 0
        tracker.h_state = [None] * cfg.MODEL.NECK.N_LAYERS
        tracker.update_threshold = cfg.TEST.UPT.TRACKINGNET
        tracker.update_h_t = cfg.TEST.UPH.TRACKINGNET
        tracker.update_intervals = cfg.TEST.INTER.TRACKINGNET
        tracker.memory_bank = cfg.TEST.MB.TRACKINGNET
        with self.inference_lock:
            tracker.initialize(image, {"init_bbox": box, "seq_name": "MyLabelRemote"})
        return tracker

    def track(self, tracker: Any, image: np.ndarray) -> list[int]:
        import torch
        with self.inference_lock, torch.inference_mode():
            result = tracker.track(image)
        return normalize_box(result["target_bbox"], image.shape[1], image.shape[0])

    def status(self) -> dict[str, Any]:
        return {"type": "mcitrack_l384", "loaded": self.loaded, "device": self.device_name}


# Future algorithm example:
#   class NewAlgorithmModel: load(), new_tracker(), track(), status()
#   ALGORITHM_FACTORIES["new_type"] = NewAlgorithmModel
ALGORITHM_FACTORIES: dict[str, type[Any]] = {"mcitrack_l384": MCITrackL384Model}


@dataclass
class TrackingSession:
    algorithm: str
    model: Any
    tracker: Any
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


class ModelService:
    def __init__(self, config: dict[str, Any], config_file: Path) -> None:
        self.server_options = config["server"]
        self.max_image_bytes = int(self.server_options.get("max_image_bytes", 20 * 1024 * 1024))
        self.session_timeout = max(60, int(self.server_options.get("session_timeout_seconds", 1800)))
        configured_token = os.environ.get("MYLABEL_SERVER_TOKEN", str(self.server_options.get("access_token", "")))
        self.access_token = configured_token.strip()
        self.models: dict[str, Any] = {}
        for name, options in config["models"].items():
            if not isinstance(options, dict) or not options.get("enabled", True):
                continue
            model_type = str(options.get("type", name))
            factory = ALGORITHM_FACTORIES.get(model_type)
            if factory is None:
                raise ValueError(f"model.json中的算法类型尚未注册：{model_type}")
            self.models[str(name)] = factory(str(name), options, config_file)
        if not self.models:
            raise ValueError("model.json中没有启用任何算法。")
        self.sessions: dict[str, TrackingSession] = {}
        self.sessions_lock = threading.RLock()

    def authorized(self, header: str) -> bool:
        if not self.access_token:
            return True
        prefix = "Bearer "
        supplied = header[len(prefix):] if header.startswith(prefix) else ""
        return hmac.compare_digest(supplied, self.access_token)

    def create_session(self, algorithm: str, image: np.ndarray, box: Any) -> tuple[str, Any]:
        model = self.models.get(algorithm)
        if model is None:
            raise ValueError(f"服务端没有启用算法：{algorithm}")
        values = np.asarray(box, dtype=np.float64).reshape(-1)
        if values.size != 4 or not np.isfinite(values).all() or values[2] <= 0 or values[3] <= 0:
            raise ValueError("初始框必须是有效的[x,y,w,h]。")
        tracker = model.new_tracker(image, values.tolist())
        session_id = secrets.token_urlsafe(24)
        with self.sessions_lock:
            self.sessions[session_id] = TrackingSession(algorithm, model, tracker)
        return session_id, model

    def get_session(self, session_id: str) -> TrackingSession:
        with self.sessions_lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("跟踪会话不存在或已过期。")
        return session

    def delete_session(self, session_id: str) -> bool:
        with self.sessions_lock:
            return self.sessions.pop(session_id, None) is not None

    def cleanup_expired(self) -> int:
        cutoff = time.time() - self.session_timeout
        with self.sessions_lock:
            expired = [key for key, value in self.sessions.items() if value.last_used < cutoff]
            for key in expired:
                del self.sessions[key]
        return len(expired)


def create_app(config_file: Path) -> tuple[Flask, ModelService, dict[str, Any]]:
    config_file = config_file.resolve()
    config = load_config(config_file)
    service = ModelService(config, config_file)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = service.max_image_bytes * 2 + 1024 * 1024

    @app.before_request
    def check_auth() -> Any:
        if request.path == "/api/v1/health" and not service.access_token:
            return None
        if not service.authorized(request.headers.get("Authorization", "")):
            return jsonify({"ok": False, "error": "访问令牌无效。"}), 401
        return None

    @app.get("/api/v1/health")
    def health() -> Any:
        service.cleanup_expired()
        with service.sessions_lock:
            session_count = len(service.sessions)
        return jsonify({
            "ok": True,
            "service": "MyLabel Tracking Server",
            "algorithms": {name: model.status() for name, model in service.models.items()},
            "active_sessions": session_count,
        })

    @app.post("/api/v1/sessions")
    def start_session() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("请求正文必须是JSON对象。")
        image = decode_image(payload.get("image_base64"), service.max_image_bytes)
        session_id, model = service.create_session(str(payload.get("algorithm", "")), image, payload.get("box"))
        return jsonify({"ok": True, "session_id": session_id, "algorithm": model.name, "device": model.device_name})

    @app.post("/api/v1/sessions/<session_id>/track")
    def track(session_id: str) -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("请求正文必须是JSON对象。")
        image = decode_image(payload.get("image_base64"), service.max_image_bytes)
        session = service.get_session(session_id)
        with session.lock:
            box = session.model.track(session.tracker, image)
            session.last_used = time.time()
        return jsonify({"ok": True, "box": box, "frame_index": payload.get("frame_index")})

    @app.delete("/api/v1/sessions/<session_id>")
    def stop_session(session_id: str) -> Any:
        return jsonify({"ok": True, "deleted": service.delete_session(session_id)})

    @app.errorhandler(Exception)
    def handle_error(exc: Exception) -> Any:
        status = 404 if isinstance(exc, KeyError) else 400 if isinstance(exc, ValueError) else 500
        message = str(exc.args[0] if isinstance(exc, KeyError) and exc.args else exc) or exc.__class__.__name__
        if status == 500:
            traceback.print_exc()
        return jsonify({"ok": False, "error": message}), status

    def cleanup_loop() -> None:
        while True:
            time.sleep(min(60, max(10, service.session_timeout // 4)))
            removed = service.cleanup_expired()
            if removed:
                print(f"已清理 {removed} 个过期跟踪会话。", flush=True)

    threading.Thread(target=cleanup_loop, name="session-cleanup", daemon=True).start()
    return app, service, config


def main() -> int:
    parser = argparse.ArgumentParser(description="MyLabel远程自动标注服务")
    parser.add_argument("--config", default="model.json", help="模型与服务配置文件")
    parser.add_argument("--host", help="覆盖model.json中的监听地址")
    parser.add_argument("--port", type=int, help="覆盖model.json中的监听端口")
    args = parser.parse_args()
    app, service, config = create_app(Path(args.config))
    server_options = config["server"]
    host = args.host or str(server_options.get("host", "0.0.0.0"))
    port = args.port or int(server_options.get("port", 8000))
    print(f"MyLabel跟踪服务启动：http://{host}:{port}", flush=True)
    print(f"已配置算法：{', '.join(service.models)}", flush=True)
    print("访问令牌：" + ("已启用" if service.access_token else "未启用（仅建议可信内网使用）"), flush=True)
    app.run(host=host, port=port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

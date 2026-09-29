"""检查 MyLabel 服务端的 MCITrack-L384 环境、权重和真实推理。

常用命令：
  python test_mcitrack_env.py --config model.json --require-cuda
  python test_mcitrack_env.py --config model.json --video demo.mp4 --box 100,80,200,300
  python test_mcitrack_env.py --config model.json --image first.jpg --next-image second.jpg --box 100,80,200,300
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any


CHECK = "[检查]"
PASS = "[通过]"
WARN = "[警告]"
FAIL = "[失败]"


def print_item(prefix: str, message: str) -> None:
    print(f"{prefix} {message}", flush=True)


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "未知"


def check_imports() -> dict[str, Any]:
    requirements = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python/opencv-python-headless"),
        ("yaml", "PyYAML"),
        ("easydict", "easydict"),
        ("timm", "timm"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("flask", "Flask"),
    ]
    modules: dict[str, Any] = {}
    missing: list[str] = []
    print_item(CHECK, "检查 Python 依赖……")
    for module_name, display_name in requirements:
        try:
            module = importlib.import_module(module_name)
            modules[module_name] = module
            distribution = "opencv-python-headless" if module_name == "cv2" else display_name
            version = package_version(distribution) if module_name == "flask" else (
                getattr(module, "__version__", "") or package_version(distribution)
            )
            print_item(PASS, f"{display_name}: {version}")
        except Exception as exc:
            missing.append(display_name)
            print_item(FAIL, f"{display_name}: {exc}")
    if missing:
        raise RuntimeError("缺少必要依赖：" + "、".join(missing))
    return modules


def parse_box(value: str) -> list[float]:
    try:
        box = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("目标框必须是 x,y,w,h 四个数字。") from exc
    if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
        raise argparse.ArgumentTypeError("目标框必须是有效的 x,y,w,h，且宽高大于0。")
    return box


def synchronize(torch: Any, device: Any) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def load_rgb_image(cv2: Any, path: Path) -> Any:
    import numpy as np

    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        bgr = None
    if bgr is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_test_frames(cv2: Any, args: argparse.Namespace) -> tuple[Any, Any] | None:
    if args.video:
        video = Path(args.video).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"测试视频不存在：{video}")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV无法打开测试视频：{video}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
            ok1, first = capture.read()
            ok2, second = capture.read()
        finally:
            capture.release()
        if not ok1 or first is None or not ok2 or second is None:
            raise RuntimeError(f"无法读取视频第 {args.start_frame + 1}、{args.start_frame + 2} 帧。")
        return cv2.cvtColor(first, cv2.COLOR_BGR2RGB), cv2.cvtColor(second, cv2.COLOR_BGR2RGB)
    if args.image or args.next_image:
        if not args.image or not args.next_image:
            raise ValueError("图片推理测试必须同时提供 --image 和 --next-image。")
        return (
            load_rgb_image(cv2, Path(args.image).expanduser().resolve()),
            load_rgb_image(cv2, Path(args.next_image).expanduser().resolve()),
        )
    return None


def show_cuda(torch: Any, require_cuda: bool) -> None:
    print_item(CHECK, "检查 PyTorch 与 GPU……")
    print_item(PASS, f"PyTorch: {torch.__version__}")
    cuda_available = bool(torch.cuda.is_available())
    print_item(PASS if cuda_available else WARN, f"CUDA可用：{cuda_available}")
    print_item(CHECK, f"PyTorch编译CUDA版本：{torch.version.cuda or '无（CPU版PyTorch）'}")
    if cuda_available:
        count = torch.cuda.device_count()
        print_item(PASS, f"检测到 {count} 张CUDA GPU")
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            total_gib = properties.total_memory / 1024**3
            print_item(PASS, f"GPU {index}: {properties.name}，显存 {total_gib:.2f} GiB，计算能力 {properties.major}.{properties.minor}")
        try:
            free, total = torch.cuda.mem_get_info(0)
            print_item(CHECK, f"GPU 0 当前空闲/总显存：{free / 1024**3:.2f}/{total / 1024**3:.2f} GiB")
        except Exception as exc:
            print_item(WARN, f"无法读取GPU空闲显存：{exc}")
    elif require_cuda:
        raise RuntimeError("要求CUDA测试，但当前环境无法使用CUDA。请检查NVIDIA驱动和GPU版PyTorch。")
    else:
        print_item(WARN, "当前将使用CPU完成模型测试；服务器加速需要安装CUDA版PyTorch。")


def main() -> int:
    parser = argparse.ArgumentParser(description="MCITrack-L384 服务端环境与推理测试")
    parser.add_argument("--config", default="model.json", help="服务端model.json路径（默认：model.json）")
    parser.add_argument("--algorithm", default="mcitrack_l384", help="model.json中的算法名称")
    parser.add_argument("--require-cuda", action="store_true", help="CUDA不可用时测试失败（Linux GPU服务器推荐）")
    parser.add_argument("--video", help="可选：用于两帧真实推理的测试视频")
    parser.add_argument("--start-frame", type=int, default=0, help="测试视频起始帧，0表示第一帧")
    parser.add_argument("--image", help="可选：初始化图片")
    parser.add_argument("--next-image", help="可选：初始化图片的下一帧")
    parser.add_argument("--box", type=parse_box, help="真实推理初始框：x,y,w,h")
    parser.add_argument("--skip-model-load", action="store_true", help="只检查基础环境、路径和CUDA，不加载1.4GB权重")
    args = parser.parse_args()

    if args.start_frame < 0:
        parser.error("--start-frame不能小于0。")
    wants_frames = bool(args.video or args.image or args.next_image)
    if wants_frames and args.box is None:
        parser.error("真实推理测试必须提供 --box x,y,w,h。")
    if args.skip_model_load and wants_frames:
        parser.error("--skip-model-load不能与真实推理参数同时使用。")

    started = time.perf_counter()
    print("=" * 68)
    print("MCITrack-L384 环境测试")
    print("=" * 68)
    print_item(CHECK, f"系统：{platform.platform()}")
    print_item(CHECK, f"Python：{sys.version.split()[0]} ({sys.executable})")
    print_item(CHECK, f"工作目录：{Path.cwd()}")

    try:
        modules = check_imports()
        torch = modules["torch"]
        cv2 = modules["cv2"]
        show_cuda(torch, args.require_cuda)

        config_file = Path(args.config).expanduser().resolve()
        print_item(CHECK, f"读取配置：{config_file}")
        try:
            config = json.loads(config_file.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"无法读取model.json：{exc}") from exc
        options = config.get("models", {}).get(args.algorithm)
        if not isinstance(options, dict) or not options.get("enabled", True):
            raise ValueError(f"model.json中没有启用算法：{args.algorithm}")

        from server import ALGORITHM_FACTORIES, resolve_path
        model_type = str(options.get("type", args.algorithm))
        factory = ALGORITHM_FACTORIES.get(model_type)
        if factory is None:
            raise ValueError(f"server.py尚未注册算法类型：{model_type}")
        code_path = resolve_path(config_file, str(options["code_path"]))
        weight_path = resolve_path(config_file, str(options["model_path"]))
        yaml_name = str(options.get("config", "mcitrack_l384"))
        yaml_path = code_path / "experiments" / "mcitrack" / f"{yaml_name}.yaml"
        for label, path, expected in (
            ("MCITrack代码目录", code_path, "dir"),
            ("L384配置", yaml_path, "file"),
            ("模型权重", weight_path, "file"),
        ):
            exists = path.is_dir() if expected == "dir" else path.is_file()
            print_item(PASS if exists else FAIL, f"{label}：{path}")
            if not exists:
                raise FileNotFoundError(f"{label}不存在：{path}")
        size_gib = weight_path.stat().st_size / 1024**3
        print_item(CHECK, f"权重大小：{size_gib:.3f} GiB")

        if args.skip_model_load:
            print_item(WARN, "已按参数跳过权重加载和真实推理。")
        else:
            print_item(CHECK, "开始严格加载MCITrack-L384完整权重……")
            model = factory(args.algorithm, options, config_file)
            load_started = time.perf_counter()
            model.load()
            synchronize(torch, model.device)
            load_seconds = time.perf_counter() - load_started
            print_item(PASS, f"模型严格加载成功，设备={model.device_name}，耗时={load_seconds:.3f}秒")
            if model.device.type == "cuda":
                allocated = torch.cuda.memory_allocated(model.device) / 1024**3
                reserved = torch.cuda.memory_reserved(model.device) / 1024**3
                print_item(CHECK, f"模型加载后显存：已分配 {allocated:.2f} GiB，已保留 {reserved:.2f} GiB")

            frames = load_test_frames(cv2, args)
            if frames is None:
                print_item(WARN, "未提供视频或连续图片；已验证模型完整加载，但未执行逐帧推理。")
                print_item(CHECK, "如需真实推理：--video <视频> --box x,y,w,h")
            else:
                first, second = frames
                print_item(CHECK, f"测试画面：{first.shape[1]}x{first.shape[0]}，初始框={args.box}")
                init_started = time.perf_counter()
                tracker = model.new_tracker(first, args.box)
                synchronize(torch, model.device)
                init_seconds = time.perf_counter() - init_started
                infer_started = time.perf_counter()
                result_box = model.track(tracker, second)
                synchronize(torch, model.device)
                infer_seconds = time.perf_counter() - infer_started
                print_item(PASS, f"首帧初始化成功，耗时={init_seconds:.3f}秒")
                print_item(PASS, f"下一帧真实推理成功，输出框={result_box}，耗时={infer_seconds:.3f}秒")
                if infer_seconds > 0:
                    print_item(CHECK, f"单会话实测速度：{1.0 / infer_seconds:.2f} FPS（不含网络传输和JPEG编解码）")

        elapsed = time.perf_counter() - started
        print("=" * 68)
        print_item(PASS, f"MCITrack环境测试完成，总耗时 {elapsed:.3f} 秒。")
        if not torch.cuda.is_available():
            print_item(WARN, "测试虽通过，但当前是CPU模式，Linux GPU服务器请使用 --require-cuda 再测一次。")
        return 0
    except Exception as exc:
        print("=" * 68)
        print_item(FAIL, str(exc) or exc.__class__.__name__)
        if os.environ.get("MYLABEL_TEST_TRACEBACK", "").casefold() in {"1", "true", "yes"}:
            traceback.print_exc()
        print_item(FAIL, "MCITrack环境测试未通过。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
